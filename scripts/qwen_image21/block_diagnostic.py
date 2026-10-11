#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Complete native block-0 operations with actual pinned BF16 weights.
REPO CHECKPOINT PLAN_JSON NATIVE_CIS_JSON NEW_OUTPUT SEED. CUDA kernels perform every
native operation; Torch allocates buffers and runs the independent reference.
"""
import ctypes,hashlib,inspect,json,pathlib,subprocess,sys
import torch
from safetensors import safe_open
import diffusers.models.transformers.transformer_qwenimage21 as reference
repo,model,planfile,cisfile,out=map(pathlib.Path,sys.argv[1:6]);seed=int(sys.argv[6]);out.mkdir(parents=True,exist_ok=False)
refsha=hashlib.sha256(pathlib.Path(inspect.getfile(reference)).read_bytes()).hexdigest();assert refsha=='0eb0555e21ca93195e1fe9389113cafbf0e8822f6454c3001cebb3d21521ebd7'
sources=[]
for name in ['nllb_encoder.cu','image_modulation.cu','dense_gemm_bf16.cu','embed_from_argmax.cu','rms_norm_vanilla.cu']:
 p=out/name;p.write_bytes((repo/'kernels/gb10/common'/name).read_bytes());sources.append(p)
wrapper=out/'wrapper.cu';wrapper.write_text(''.join('#include "'+str(p.resolve())+'"\n' for p in sources)+r'''
extern "C" int linear(void*x,void*w,void*y,unsigned m,unsigned n,unsigned k){dense_gemm_bf16_pipelined<<<dim3((n+127)/128,(m+127)/128),256>>>((__nv_bfloat16*)x,(__nv_bfloat16*)w,(__nv_bfloat16*)y,m,n,k);return cudaDeviceSynchronize();}
extern "C" int layer_norm(void*x,void*one,void*zero,void*y,unsigned rows){nllb_layernorm_oop_bf16<<<rows,256,1024>>>((__nv_bfloat16*)x,(__nv_bfloat16*)y,(__nv_bfloat16*)one,(__nv_bfloat16*)zero,rows,4096,1e-6f);return cudaDeviceSynchronize();}
extern "C" int scale(void*x,void*mod,void*sel,void*y,unsigned rows,unsigned component){image_modulation_scale_bf16<<<(rows*4096+255)/256,256>>>((__nv_bfloat16*)x,(__nv_bfloat16*)mod,(uint32_t*)sel,(__nv_bfloat16*)y,rows,4096,16384,component*4096);return cudaDeviceSynchronize();}
extern "C" int residual(void*x,void*branch,void*mod,void*sel,void*y,unsigned rows,unsigned component){image_modulation_residual_bf16<<<(rows*4096+255)/256,256>>>((__nv_bfloat16*)x,(__nv_bfloat16*)branch,(__nv_bfloat16*)mod,(uint32_t*)sel,(__nv_bfloat16*)y,rows,4096,16384,component*4096);return cudaDeviceSynchronize();}
extern "C" int headnorm(void*x,void*one,void*w,void*tmp,unsigned rows){rms_norm_vanilla<<<rows,128>>>((__nv_bfloat16*)x,(__nv_bfloat16*)one,(__nv_bfloat16*)tmp,128,1e-6f);image_head_weight_bf16<<<(rows*128+255)/256,256>>>((__nv_bfloat16*)tmp,(__nv_bfloat16*)w,(__nv_bfloat16*)x,rows,128);return cudaDeviceSynchronize();}
extern "C" int rotate(void*x,void*cis,unsigned samples,unsigned sequence){image_rope_complex_bf16<<<(samples*sequence*32*64+255)/256,256>>>((__nv_bfloat16*)x,(float*)cis,(__nv_bfloat16*)x,samples,sequence,32);return cudaDeviceSynchronize();}
extern "C" int gather(void*ids,void*x,void*y,unsigned rows){batched_embed<<<rows,256>>>((unsigned*)ids,(__nv_bfloat16*)x,(__nv_bfloat16*)y,4096);return cudaDeviceSynchronize();}
extern "C" int attend(void*q,void*k,void*v,void*y,unsigned keys){nllb_attn_kv_bf16<<<32,128,(keys+128)*4>>>((__nv_bfloat16*)q,(__nv_bfloat16*)k,(__nv_bfloat16*)v,(__nv_bfloat16*)y,1,keys,32,128,1.0f/sqrtf(128.0f),0);return cudaDeviceSynchronize();}
extern "C" int activate(void*g,void*u,void*y,unsigned n){image_silu_staged_mul_bf16<<<(n+255)/256,256>>>((__nv_bfloat16*)g,(__nv_bfloat16*)u,(__nv_bfloat16*)y,n);return cudaDeviceSynchronize();}
''')
command=['nvcc','-shared','-Xcompiler','-fPIC','-O3','--fmad=false','-arch=sm_121',str(wrapper),'-o',str(out/'block.so')]
built=subprocess.run(command,capture_output=True,text=True);(out/'build.log').write_text(built.stdout+built.stderr);built.check_returncode()
lib=ctypes.CDLL(str(out/'block.so'))
for name,pointers,ints in [('linear',3,3),('layer_norm',4,1),('scale',4,2),('residual',5,2),('headnorm',4,1),('rotate',2,2),('gather',3,1),('attend',4,1),('activate',3,1)]:
 fn=getattr(lib,name);fn.argtypes=[ctypes.c_void_p]*pointers+[ctypes.c_uint]*ints;fn.restype=ctypes.c_int
plan=json.loads(planfile.read_text());table=json.loads(cisfile.read_text());(out/'plan.json').write_bytes(planfile.read_bytes());(out/'cis.json').write_bytes(cisfile.read_bytes())
assert plan['image_ids']==[-1,0,-1] and plan['key_valid']==[[True]*3]*2
component=model/'transformer';index=json.loads((component/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(seed)
x=torch.randn(2,3,4096,device='cuda',dtype=torch.bfloat16);mod=torch.randn(3,16384,device='cuda',dtype=torch.bfloat16)
mask=torch.tensor([False,True,True],device='cuda');selected=torch.tensor([2,0,0,2,1,1],device='cuda',dtype=torch.uint32)
weights={};weight_hashes={}
for name in ['attn.to_q','attn.to_k','attn.to_v','attn.to_out.0','attn.norm_q','attn.norm_k','img_mlp.gate_layer','img_mlp.proj','img_mlp.out']:
 key=f'transformer_blocks.0.{name}.weight'
 with safe_open(component/index[key],framework='pt',device='cpu') as f: cpu=f.get_tensor(key)
 assert cpu.dtype==torch.bfloat16
 weight_hashes[name]=hashlib.sha256(cpu.view(torch.uint16).numpy().tobytes()).hexdigest();weights[name]=cpu.cuda()
with torch.device('meta'): ref=reference.QwenImage21TransformerBlock(4096,32,128,3,1e-6)
ref.load_state_dict({name+'.weight':w for name,w in weights.items()},assign=True);ref.eval()
ref_cis=reference.QwenImage21Rope(10000,[16,56,56])([[1,1,1]],torch.tensor([False,True,False]),torch.device('cpu')).cuda()
cis=torch.tensor(table['cis_f32_bits'],dtype=torch.uint32).view(torch.float32).cuda();one=torch.ones(4096,device='cuda',dtype=torch.bfloat16);zero=torch.zeros_like(one)
results=[]
def metric(a,b):
 finite=torch.isfinite(a)&torch.isfinite(b);af,bf=a[finite].double(),b[finite].double();error=af-bf
 return dict(elements=a.numel(),bf16_bit_mismatches=int((a.view(torch.int16)!=b.view(torch.int16)).sum()),nonfinite_pairs=int((~finite).sum()),nonfinite_bit_mismatches=int(((a.view(torch.int16)!=b.view(torch.int16))&~finite).sum()),max_abs_error_finite=float(error.abs().max()) if error.numel() else 0.0,relative_l2_finite=float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(bf).clamp_min(1e-30)))
def record(name,a,b):
 results.append(dict(stage=name,**metric(a,b)))
 for label,t in [('native',a),('reference',b)]: (out/f'{name}-{label}.bf16').write_bytes(t.detach().cpu().contiguous().view(torch.uint16).numpy().tobytes())
def call(name,*args): assert getattr(lib,name)(*(a.data_ptr() if isinstance(a,torch.Tensor) else a for a in args))==0
for label,t in [('input',x),('modulation',mod)]: (out/f'{label}.bf16').write_bytes(t.cpu().view(torch.uint16).numpy().tobytes())
norm1=torch.empty_like(x);call('layer_norm',x,one,zero,norm1,6);record('norm1',norm1,torch.nn.functional.layer_norm(x,(4096,),eps=1e-6))
def selected_param(component): return reference._select_modulation_rows(mod[:,component*4096:(component+1)*4096],mask)
scaled1=torch.empty_like(x);call('scale',norm1,mod,selected,scaled1,6,0);record('scale1',scaled1,norm1*(1+selected_param(0)))
qkv=[]
for name in ['q','k','v']:
 a=torch.empty_like(x);call('linear',scaled1,weights[f'attn.to_{name}'],a,6,4096,4096);record(name,a,torch.nn.functional.linear(scaled1,weights[f'attn.to_{name}']))
 if name!='v':
  shaped=a.reshape(2,3,32,128);expected=getattr(ref.attn,f'norm_{name}')(shaped).detach();tmp=torch.empty_like(a)
  call('headnorm',a,one,weights[f'attn.norm_{name}'],tmp,192);record(name+'norm',shaped,expected)
  expected=reference.apply_rotary_emb_qwen(shaped,ref_cis,use_real=False);call('rotate',a,cis,2,3);record(name+'rope',shaped,expected)
 qkv.append(a.reshape(2,3,32,128))
q,k,v=qkv;ids=torch.tensor(plan['gather'],device='cuda',dtype=torch.uint32);ck,cv=torch.empty_like(k),torch.empty_like(v)
call('gather',ids,k,ck,6);call('gather',ids,v,cv,6);attended=torch.zeros_like(q)
for query,(start,count) in enumerate(plan['spans']):
 if count: call('attend',q.data_ptr()+query*8192,ck.data_ptr()+start*8192,cv.data_ptr()+start*8192,attended.data_ptr()+query*8192,count)
visibility=torch.tril(torch.ones(3,3,device='cuda',dtype=torch.bool))[None,None]
expected_attn=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask=visibility).transpose(1,2).contiguous();record('attention',attended,expected_attn)
branch=torch.empty_like(x);call('linear',attended,weights['attn.to_out.0'],branch,6,4096,4096);record('output_projection',branch,torch.nn.functional.linear(attended.flatten(2),weights['attn.to_out.0']))
r1=torch.empty_like(x);call('residual',x,branch,mod,selected,r1,6,1);record('residual1',r1,x+selected_param(1).tanh()*branch)
norm2=torch.empty_like(x);call('layer_norm',r1,one,zero,norm2,6);record('norm2',norm2,torch.nn.functional.layer_norm(r1,(4096,),eps=1e-6))
scaled2=torch.empty_like(x);call('scale',norm2,mod,selected,scaled2,6,2);record('scale2',scaled2,norm2*(1+selected_param(2)))
ffn=[]
for label,wn in [('gate','img_mlp.gate_layer'),('up','img_mlp.proj')]:
 a=torch.empty(2,3,12288,device='cuda',dtype=torch.bfloat16);call('linear',scaled2,weights[wn],a,6,12288,4096);record(label,a,torch.nn.functional.linear(scaled2,weights[wn]));ffn.append(a)
gate,up=ffn;activated=torch.empty_like(gate);call('activate',gate,up,activated,gate.numel());record('activation',activated,torch.nn.functional.silu(gate)*up)
down=torch.empty_like(x);call('linear',activated,weights['img_mlp.out'],down,6,4096,12288);record('down',down,torch.nn.functional.linear(activated,weights['img_mlp.out']))
final=torch.empty_like(x);call('residual',r1,down,mod,selected,final,6,3);record('residual2',final,r1+selected_param(3).tanh()*down)
with torch.no_grad(): expected=ref(x,mod,rotary_emb=ref_cis,target_token_mask=mask,segments=[(0,1,True),(1,2,False),(2,3,True)])
record('complete_block',final,expected)
# Full finite BF16 gate encoding coverage, with an independent up operand.
all_gates=torch.arange(65536,dtype=torch.int32,device='cuda').to(torch.uint16).view(torch.bfloat16);all_gates=all_gates[torch.isfinite(all_gates)]
all_up=torch.randn_like(all_gates);all_out=torch.empty_like(all_gates);call('activate',all_gates,all_up,all_out,all_gates.numel());record('activation_all_finite',all_out,torch.nn.functional.silu(all_gates)*all_up)
wrong=(torch.nn.functional.silu(gate.float())*up.float()).bfloat16();control=metric(wrong,torch.nn.functional.silu(gate)*up)['bf16_bit_mismatches']
receipt=dict(seed=seed,checkpoint_revision='d26bb61231c349cf6b7896fa83353113880e1ba3',reference_source_sha256=refsha,torch=torch.__version__,cuda=torch.version.cuda,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},weight_sha256=weight_hashes,compile_command=command,results=results,known_bad_controls=dict(missing_activation_rounding=control),qualified=False,scope='complete block-0 diagnostic with exact stage characterization; no transformer/image/performance qualification')
(out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps(receipt,indent=2,allow_nan=False));assert control>0
assert all(r['bf16_bit_mismatches']==0 for r in results),'complete-block exact gate remains failed'
