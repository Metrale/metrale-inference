#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Dense Qwen3-VL native block, actual embedding/weights, pinned source.
REPO CHECKPOINT NEW_OUT SEED. Text-only unpadded prefix, no image encoder claim.
"""
import ctypes,hashlib,inspect,json,pathlib,subprocess,sys
import torch
from safetensors import safe_open
from transformers import AutoConfig
import transformers.models.qwen3_vl.modeling_qwen3_vl as reference
repo,model,out=map(pathlib.Path,sys.argv[1:4]);seed=int(sys.argv[4]);out.mkdir(parents=True,exist_ok=False)
refsha=hashlib.sha256(pathlib.Path(inspect.getfile(reference)).read_bytes()).hexdigest();assert refsha=='1a02b852d3b113c1664ff5e7ba6f7600e28da2ddf9d479e873de723ff1dc2e60'
sources=[]
for name in ['rms_norm_vanilla.cu','image_modulation.cu','dense_gemm_bf16.cu','attn_prefill_h128.cu','residual_add.cu']:
 p=out/name;p.write_bytes((repo/'kernels/gb10/common'/name).read_bytes());sources.append(p)
wrapper=out/'wrapper.cu';wrapper.write_text(''.join('#include "'+str(p.resolve())+'"\n' for p in sources)+r'''
extern "C" int linear(void*x,void*w,void*y,unsigned m,unsigned n,unsigned k){dense_gemm_bf16_pipelined<<<dim3((n+127)/128,(m+127)/128),256>>>((__nv_bfloat16*)x,(__nv_bfloat16*)w,(__nv_bfloat16*)y,m,n,k);return cudaDeviceSynchronize();}
extern "C" int staged_norm(void*x,void*one,void*w,void*tmp,void*y,unsigned rows,unsigned cols){rms_norm_vanilla<<<rows,min(cols,1024u)>>>((__nv_bfloat16*)x,(__nv_bfloat16*)one,(__nv_bfloat16*)tmp,cols,1e-6f);image_head_weight_bf16<<<(rows*cols+255)/256,256>>>((__nv_bfloat16*)tmp,(__nv_bfloat16*)w,(__nv_bfloat16*)y,rows,cols);return cudaDeviceSynchronize();}
extern "C" int rotate(void*x,void*c,void*s,unsigned tokens,unsigned heads){image_text_rope_bf16<<<(tokens*heads*64+255)/256,256>>>((__nv_bfloat16*)x,(__nv_bfloat16*)c,(__nv_bfloat16*)s,tokens,heads);return cudaDeviceSynchronize();}
extern "C" int attend(void*q,void*k,void*v,void*y,unsigned tokens){attn_prefill_h128<<<dim3(32,(tokens+31)/32),128>>>((__nv_bfloat16*)q,(__nv_bfloat16*)k,(__nv_bfloat16*)v,(__nv_bfloat16*)y,tokens,32,8,128,1.0f/sqrtf(128.f),1,0);return cudaDeviceSynchronize();}
extern "C" int activate(void*g,void*u,void*y,unsigned n){image_silu_staged_mul_bf16<<<(n+255)/256,256>>>((__nv_bfloat16*)g,(__nv_bfloat16*)u,(__nv_bfloat16*)y,n);return cudaDeviceSynchronize();}
extern "C" int add(void*x,void*y,unsigned n){bf16_residual_add<<<(n+255)/256,256>>>((__nv_bfloat16*)x,(__nv_bfloat16*)y,n);return cudaDeviceSynchronize();}
''')
command=['nvcc','-shared','-Xcompiler','-fPIC','-O3','--fmad=false','-arch=sm_121',str(wrapper),'-o',str(out/'encoder.so')]
build=subprocess.run(command,capture_output=True,text=True);(out/'build.log').write_text(build.stdout+build.stderr);build.check_returncode();lib=ctypes.CDLL(str(out/'encoder.so'))
for name,pointers,ints in [('linear',3,3),('staged_norm',5,2),('rotate',3,2),('attend',4,1),('activate',3,1),('add',2,1)]:
 fn=getattr(lib,name);fn.argtypes=[ctypes.c_void_p]*pointers+[ctypes.c_uint]*ints;fn.restype=ctypes.c_int
torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(seed);component=model/'text_encoder';idx=json.loads((component/'model.safetensors.index.json').read_text())['weight_map'];weights={};hashes={}
for name in ['input_layernorm','self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','self_attn.q_norm','self_attn.k_norm','post_attention_layernorm','mlp.gate_proj','mlp.up_proj','mlp.down_proj']:
 key='model.language_model.layers.0.'+name+'.weight'
 with safe_open(component/idx[key],framework='pt',device='cpu') as f:w=f.get_tensor(key)
 assert w.dtype==torch.bfloat16;hashes[key]=hashlib.sha256(w.view(torch.uint16).numpy().tobytes()).hexdigest();weights[name]=w.cuda()
ids=torch.randint(0,151936,(33,),dtype=torch.long);key='model.language_model.embed_tokens.weight'
with safe_open(component/idx[key],framework='pt',device='cpu') as f:x=f.get_tensor(key)[ids].unsqueeze(0).cuda()
config=AutoConfig.from_pretrained(component,local_files_only=True).text_config;config._attn_implementation='sdpa'
with torch.device('meta'): block=reference.Qwen3VLTextDecoderLayer(config,0)
block.load_state_dict({k+'.weight':v for k,v in weights.items()},assign=True);block.eval();rope=reference.Qwen3VLTextRotaryEmbedding(config).cuda()
positions=torch.arange(33,device='cuda').reshape(1,-1);cos,sin=rope(x,positions);c=cos[0,:,:64].contiguous();s=sin[0,:,:64].contiguous()
one=torch.ones(4096,device='cuda',dtype=torch.bfloat16);results=[]
def call(name,*args):assert getattr(lib,name)(*(v.data_ptr() if isinstance(v,torch.Tensor) else v for v in args))==0
def record(name,a,b):
 finite=torch.isfinite(a)&torch.isfinite(b);af,bf=a[finite].double(),b[finite].double();err=af-bf
 results.append(dict(stage=name,elements=a.numel(),bit_mismatches=int((a.view(torch.int16)!=b.view(torch.int16)).sum()),nonfinite_pairs=int((~finite).sum()),relative_l2_finite=float(torch.linalg.vector_norm(err)/torch.linalg.vector_norm(bf).clamp_min(1e-30)),max_abs_error_finite=float(err.abs().max())))
 for label,t in [('native',a),('reference',b)]: (out/(name+'-'+label+'.bf16')).write_bytes(t.detach().cpu().contiguous().view(torch.uint16).numpy().tobytes())
def norm(name,a,key):
 y=torch.empty_like(a);tmp=torch.empty_like(a);cols=a.shape[-1];call('staged_norm',a,one,weights[key],tmp,y,a.numel()//cols,cols);expected=(a.float()*torch.rsqrt(a.float().square().mean(-1,keepdim=True)+1e-6)).bfloat16()*weights[key];record(name,y,expected);return y
def linear(name,a,key):
 w=weights[key];y=torch.empty((*a.shape[:-1],w.shape[0]),device='cuda',dtype=torch.bfloat16);call('linear',a,w,y,a.numel()//a.shape[-1],w.shape[0],w.shape[1]);record(name,y,torch.nn.functional.linear(a,w));return y
h=norm('input_norm',x,'input_layernorm');q=linear('q',h,'self_attn.q_proj').reshape(1,33,32,128);k=linear('k',h,'self_attn.k_proj').reshape(1,33,8,128);v=linear('v',h,'self_attn.v_proj').reshape(1,33,8,128);q=norm('q_norm',q,'self_attn.q_norm');k=norm('k_norm',k,'self_attn.k_norm')
eq,ek=reference.apply_rotary_pos_emb(q,k,cos,sin,unsqueeze_dim=2);fused=(q.float()*cos.unsqueeze(2).float()+reference.rotate_half(q).float()*sin.unsqueeze(2).float()).bfloat16();controls={'missing_product_rounding':int((fused.view(torch.int16)!=eq.view(torch.int16)).sum())};call('rotate',q,c,s,33,32);call('rotate',k,c,s,33,8);record('q_rope',q,eq);record('k_rope',k,ek)
attn=torch.empty_like(q);call('attend',q,k,v,attn,33);expected=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),is_causal=True,enable_gqa=True).transpose(1,2).contiguous();record('attention',attn,expected)
branch=linear('o',attn.reshape(1,33,4096),'self_attn.o_proj');residual=x.clone();expected=residual+branch;call('add',residual,branch,residual.numel());record('residual1',residual,expected)
h=norm('post_norm',residual,'post_attention_layernorm');gate=linear('gate',h,'mlp.gate_proj');up=linear('up',h,'mlp.up_proj');active=torch.empty_like(gate);call('activate',gate,up,active,gate.numel());record('activation',active,torch.nn.functional.silu(gate)*up);branch=linear('down',active,'mlp.down_proj');expected=residual+branch;call('add',residual,branch,residual.numel());record('residual2',residual,expected)
mask=torch.zeros(33,33,device='cuda',dtype=torch.bfloat16).masked_fill(torch.ones(33,33,device='cuda',dtype=torch.bool).triu(1),float('-inf')).reshape(1,1,33,33)
with torch.inference_mode(): expected=block(x,(cos,sin),attention_mask=mask)
record('complete_block',residual,expected)
for name,t in [('input',x),('cos',c),('sin',s)]: (out/(name+'.bf16')).write_bytes(t.cpu().contiguous().view(torch.uint16).numpy().tobytes())
(out/'input-ids.json').write_text(json.dumps(ids.tolist()))
receipt=dict(seed=seed,reference_source_sha256=refsha,weight_sha256=hashes,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},command=command,results=results,known_bad_controls=controls,scope='one native dense text block, actual embeddings/weights; supplied reference RoPE coefficients; no encoder/pipeline admission',criterion='exact BF16 bits retained separately from finite error characterization')
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

assert controls["missing_product_rounding"]>0
