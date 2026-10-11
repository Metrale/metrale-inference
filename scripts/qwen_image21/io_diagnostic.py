#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Actual-weight native time/text/latent projection diagnostics.
REPO CHECKPOINT NATIVE_TIME_FREQUENCIES NEW_OUT SEED. External encoder inputs;
no VLM, transformer stack, VAE or image-generation claim.
"""
import ctypes,hashlib,inspect,json,pathlib,subprocess,sys
import torch
from safetensors import safe_open
import diffusers.models.transformers.transformer_qwenimage21 as reference
repo,model,table,out=map(pathlib.Path,sys.argv[1:5]);seed=int(sys.argv[5]);out.mkdir(parents=True,exist_ok=False)
refsha=hashlib.sha256(pathlib.Path(inspect.getfile(reference)).read_bytes()).hexdigest();assert refsha=='0eb0555e21ca93195e1fe9389113cafbf0e8822f6454c3001cebb3d21521ebd7'
sources=[]
for relative in ['common/image_modulation.cu','common/rms_norm.cu','common/nllb_encoder.cu','common/dense_gemm_bf16.cu','gemma-4-26b-a4b/nvfp4/gelu.cu']:
 p=out/pathlib.Path(relative).name;p.write_bytes((repo/'kernels/gb10'/relative).read_bytes());sources.append(p)
wrapper=out/'wrapper.cu';wrapper.write_text(''.join('#include "'+str(p.resolve())+'"\n' for p in sources)+r'''
extern "C" int linear(void*x,void*w,void*y,unsigned m,unsigned n,unsigned k){dense_gemm_bf16_pipelined<<<dim3((n+127)/128,(m+127)/128),256>>>((__nv_bfloat16*)x,(__nv_bfloat16*)w,(__nv_bfloat16*)y,m,n,k);return cudaDeviceSynchronize();}
extern "C" int temporal(void*t,void*f,void*y,unsigned rows){image_timestep_bf16<<<rows,256>>>((__nv_bfloat16*)t,(float*)f,(__nv_bfloat16*)y,rows);return cudaDeviceSynchronize();}
extern "C" int activate(void*x,void*y,unsigned n){image_silu_staged_mul_bf16<<<(n+255)/256,256>>>((__nv_bfloat16*)x,nullptr,(__nv_bfloat16*)y,n);return cudaDeviceSynchronize();}
extern "C" int gelu_apply(void*x,void*y,unsigned n){gelu_tanh<<<(n+255)/256,256>>>((__nv_bfloat16*)x,(__nv_bfloat16*)y,n);return cudaDeviceSynchronize();}
extern "C" int textnorm(void*x,void*w,void*y,unsigned rows){rms_norm<<<rows,256>>>((__nv_bfloat16*)x,(__nv_bfloat16*)w,(__nv_bfloat16*)y,4096,1e-6f);return cudaDeviceSynchronize();}
extern "C" int layer_norm(void*x,void*one,void*zero,void*y,unsigned rows){nllb_layernorm_oop_bf16<<<rows,256,1024>>>((__nv_bfloat16*)x,(__nv_bfloat16*)y,(__nv_bfloat16*)one,(__nv_bfloat16*)zero,rows,4096,1e-6f);return cudaDeviceSynchronize();}
extern "C" int scale(void*x,void*m,void*sel,void*y,unsigned rows){image_modulation_scale_bf16<<<(rows*4096+255)/256,256>>>((__nv_bfloat16*)x,(__nv_bfloat16*)m,(uint32_t*)sel,(__nv_bfloat16*)y,rows,4096,4096,0);return cudaDeviceSynchronize();}
''')
command=['nvcc','-shared','-Xcompiler','-fPIC','-O3','--fmad=false','-arch=sm_121',str(wrapper),'-o',str(out/'io.so')]
r=subprocess.run(command,capture_output=True,text=True);(out/'build.log').write_text(r.stdout+r.stderr);r.check_returncode()
lib=ctypes.CDLL(str(out/'io.so'))
for name,pointers,ints in [('linear',3,3),('temporal',3,1),('activate',2,1),('gelu_apply',2,1),('textnorm',3,1),('layer_norm',4,1),('scale',4,1)]:
 fn=getattr(lib,name);fn.argtypes=[ctypes.c_void_p]*pointers+[ctypes.c_uint]*ints;fn.restype=ctypes.c_int
component=model/'transformer';index=json.loads((component/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(seed);weights={};hashes={}
for name in ['img_in','modulation.1','norm_out.linear','proj_out','time_text_embed.timestep_embedder.linear_1','time_text_embed.timestep_embedder.linear_2','txt_in.in_layer','txt_in.out_layer','txt_in.text_norm']:
 with safe_open(component/index[name+'.weight'],framework='pt',device='cpu') as f: w=f.get_tensor(name+'.weight')
 assert w.dtype==torch.bfloat16;hashes[name]=hashlib.sha256(w.view(torch.uint16).numpy().tobytes()).hexdigest();weights[name]=w.cuda()
results=[]
def call(name,*args): assert getattr(lib,name)(*(a.data_ptr() if isinstance(a,torch.Tensor) else a for a in args))==0
def record(name,a,b):
 finite=torch.isfinite(a)&torch.isfinite(b);af,bf=a[finite].double(),b[finite].double();err=af-bf
 results.append(dict(stage=name,elements=a.numel(),bit_mismatches=int((a.view(torch.int16)!=b.view(torch.int16)).sum()),nonfinite_pairs=int((~finite).sum()),nonfinite_bit_mismatches=int(((a.view(torch.int16)!=b.view(torch.int16))&~finite).sum()),max_abs_error_finite=float(err.abs().max()) if err.numel() else 0.,relative_l2_finite=float(torch.linalg.vector_norm(err)/torch.linalg.vector_norm(bf).clamp_min(1e-30))))
 for label,tensor in [('native',a),('reference',b)]: (out/(name+'-'+label+'.bf16')).write_bytes(tensor.detach().cpu().contiguous().view(torch.uint16).numpy().tobytes())
def linear(name,x,key):
 w=weights[key];y=torch.empty((*x.shape[:-1],w.shape[0]),device='cuda',dtype=torch.bfloat16);call('linear',x,w,y,x.numel()//x.shape[-1],w.shape[0],w.shape[1]);record(name,y,torch.nn.functional.linear(x,w));return y
freq=torch.tensor(json.loads(table.read_text()),dtype=torch.int32).view(torch.float32).cuda()
time=torch.tensor([.731,.019,0],device='cuda',dtype=torch.bfloat16);emb=torch.empty(3,256,device='cuda',dtype=torch.bfloat16);call('temporal',time,freq,emb,3);record('temporal',emb,reference.QwenImage21TemporalTimesteps(256).cuda()(time).bfloat16())
h=linear('time_in',emb,'time_text_embed.timestep_embedder.linear_1');act=torch.empty_like(h);call('activate',h,act,h.numel());record('time_silu',act,torch.nn.functional.silu(h));temb=linear('time_out',act,'time_text_embed.timestep_embedder.linear_2');call('activate',temb,act,temb.numel());record('temb_silu',act,torch.nn.functional.silu(temb));mod=linear('modulation',act,'modulation.1');scale=linear('final_scale',act,'norm_out.linear')
text=torch.randn(2,3,4096,device='cuda',dtype=torch.bfloat16);norm=torch.empty_like(text);call('textnorm',text,weights['txt_in.text_norm'],norm,6)
rnorm=(text.float()*torch.rsqrt(text.float().square().mean(-1,keepdim=True)+1e-6)*(weights['txt_in.text_norm'].float()+1)).bfloat16();record('text_norm',norm,rnorm)
h=linear('text_in',norm,'txt_in.in_layer');act=torch.empty_like(h);call('gelu_apply',h,act,h.numel());record('text_gelu',act,torch.nn.functional.gelu(h,approximate='tanh'));txt=linear('text_out',act,'txt_in.out_layer')
image=torch.randn(2,8,64,device='cuda',dtype=torch.bfloat16);img=linear('image_in',image,'img_in')
for name,tensor in [('image-input',image),('text-input',text)]: (out/(name+'.bf16')).write_bytes(tensor.cpu().view(torch.uint16).numpy().tobytes())
# Final projection tests a distinct known input boundary; no block stack substitution.
hidden=torch.randn(2,10,4096,device='cuda',dtype=torch.bfloat16);normalized=torch.empty_like(hidden);one=torch.ones(4096,device='cuda',dtype=torch.bfloat16);zero=torch.zeros_like(one);call('layer_norm',hidden,one,zero,normalized,20);record('output_norm',normalized,torch.nn.functional.layer_norm(hidden,(4096,),eps=1e-6))
selection=torch.tensor([2]*6+[0]*4+[2]*6+[1]*4,device='cuda',dtype=torch.uint32);scaled=torch.empty_like(hidden);call('scale',normalized,scale,selection,scaled,20);record('output_scale',scaled,normalized*(1+scale[selection.long()].reshape_as(hidden)));output=linear('image_out',scaled,'proj_out')
wrong=(text.float()*torch.rsqrt(text.float().square().mean(-1,keepdim=True)+1e-6)*weights['txt_in.text_norm'].float()).bfloat16()
controls={'missing_zero_center':int((wrong.view(torch.int16)!=rnorm.view(torch.int16)).sum()),'gelu_exact_instead_tanh':int((torch.nn.functional.gelu(h).view(torch.int16)!=torch.nn.functional.gelu(h,approximate='tanh').view(torch.int16)).sum())}
receipt=dict(seed=seed,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},reference_source_sha256=refsha,weight_sha256=hashes,command=command,results=results,known_bad_controls=controls,scope='same-input actual-weight conditioning and IO only; no native image claim',criterion='exact BF16 bits; all discrepancies retained')
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2));assert all(controls.values())
