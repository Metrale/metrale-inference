#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Native FP32 residual and first-frame layout; no full VAE claim.
REPO CHECKPOINT NEW_OUT SEED. Original weights; exact failures retained.
"""
import ctypes,hashlib,inspect,json,pathlib,subprocess,sys
import torch
from safetensors import safe_open
import diffusers.models.autoencoders.autoencoder_kl_qwenimage21 as ref
repo,model,out=map(pathlib.Path,sys.argv[1:4]);seed=int(sys.argv[4]);out.mkdir(parents=True,exist_ok=False)
refsha=hashlib.sha256(pathlib.Path(inspect.getfile(ref)).read_bytes()).hexdigest();assert refsha=='1e29f252dbddd044b84c8b794589001ce94bb125a8f13eb5b226ef30c1e2ea11'
sources={}
for name in ['image_vae','nllb_encoder']:
 p=out/(name+'.cu');p.write_bytes((repo/'kernels/gb10/common'/p.name).read_bytes());sources[name]=hashlib.sha256(p.read_bytes()).hexdigest()
wrapper=out/'wrapper.cu';wrapper.write_text('#include "image_vae.cu"\n#include "nllb_encoder.cu"\n'+r'''
extern "C" int conv(void*x,void*w,void*b,void*y,unsigned in,unsigned out,unsigned h,unsigned width,unsigned k){image_vae_conv2d_f32<<<(out*h*width+127)/128,128>>>((float*)x,(float*)w,(float*)b,(float*)y,in,out,h,width,k);return cudaDeviceSynchronize();}
extern "C" int vae_norm(void*x,void*w,void*y,unsigned c,unsigned p){image_vae_norm_f32<<<p,256>>>((float*)x,(float*)w,(float*)y,c,p);return cudaDeviceSynchronize();}
extern "C" int activate(void*x,void*y,unsigned n){image_vae_silu_f32<<<(n+255)/256,256>>>((float*)x,(float*)y,n);return cudaDeviceSynchronize();}
extern "C" int add(void*x,void*y,unsigned n){nllb_add_inplace<<<(n+255)/256,256>>>((float*)x,(float*)y,n);return cudaDeviceSynchronize();}
extern "C" int up(void*x,void*y,unsigned ic,unsigned oc,unsigned h,unsigned w,unsigned ft){image_vae_upsample_f32<<<(oc*h*w*4+255)/256,256>>>((float*)x,(float*)y,ic,oc,h,w,ft);return cudaDeviceSynchronize();}
''')
command=['nvcc','-shared','-Xcompiler','-fPIC','-O3','--fmad=false','-arch=sm_121',str(wrapper),'-o',str(out/'components.so')];b=subprocess.run(command,capture_output=True,text=True);(out/'build.log').write_text(b.stdout+b.stderr);b.check_returncode();lib=ctypes.CDLL(str(out/'components.so'))
for name,pointers,ints in [('conv',4,5),('vae_norm',3,2),('activate',2,1),('add',2,1),('up',2,5)]:
 fn=getattr(lib,name);fn.argtypes=[ctypes.c_void_p]*pointers+[ctypes.c_uint]*ints;fn.restype=ctypes.c_int
def call(name,*args):assert getattr(lib,name)(*(a.data_ptr() if isinstance(a,torch.Tensor) else a for a in args))==0
torch.cuda.set_per_process_memory_fraction(.85);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(seed)
results=[];hashes={}
def record(name,a,b):
 finite=torch.isfinite(a)&torch.isfinite(b);av,bv=a[finite].double(),b[finite].double();d=av-bv
 results.append(dict(stage=name,elements=a.numel(),bit_mismatches=int((a.view(torch.int32)!=b.view(torch.int32)).sum()),nonfinite_pairs=int((~finite).sum()),max_abs=float(d.abs().max()),relative_l2=float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(bv).clamp_min(1e-30))))
 for label,t in [('native',a),('reference',b)]: (out/(name+'-'+label+'.f32')).write_bytes(t.detach().cpu().contiguous().numpy().tobytes())
prefix='decoder.mid_block.resnets.0.';weights={}
with safe_open(model/'vae/diffusion_pytorch_model.safetensors',framework='pt',device='cpu') as f:
 for name in ['norm1.gamma','conv1.weight','conv1.bias','norm2.gamma','conv2.weight','conv2.bias']:
  x=f.get_tensor(prefix+name);assert x.dtype==torch.float32;hashes[prefix+name]=hashlib.sha256(x.numpy().tobytes()).hexdigest();weights[name]=x.cuda()
x=torch.randn(1,1152,2,3,device='cuda');identity=x.clone();(out/'input.f32').write_bytes(x.cpu().numpy().tobytes())
for index in [1,2]:
 y=torch.empty_like(x);call('vae_norm',x,weights[f'norm{index}.gamma'],y,1152,6);expected=torch.nn.functional.normalize(x,dim=1)*(1152**.5)*weights[f'norm{index}.gamma'].squeeze(1);record(f'norm{index}',y,expected)
 z=torch.empty_like(y);call('activate',y,z,z.numel());record(f'silu{index}',z,torch.nn.functional.silu(y));w,bias=weights[f'conv{index}.weight'],weights[f'conv{index}.bias'];x=torch.empty_like(z);call('conv',z,w,bias,x,1152,1152,2,3,3);record(f'conv{index}',x,torch.nn.functional.conv2d(z,w,bias,padding=1))
expected=x+identity;call('add',x,identity,x.numel());record('residual',x,expected)
block=ref.QwenImage21ResidualBlock(1152,1152).cuda().eval();block.load_state_dict(weights);record('full_block',x,block(identity.unsqueeze(2)).squeeze(2))
controls={}
for ic,oc,ft in [(1152,1152,2),(1152,576,2),(576,288,1),(288,288,1)]:
 a=torch.arange(ic*3*5,device='cuda',dtype=torch.float32).reshape(1,ic,3,5);y=torch.empty(1,oc,6,10,device='cuda');call('up',a,y,ic,oc,3,5,ft);expected=ref.QwenImage21DupUp3D(ic,oc,ft,2)(a.unsqueeze(2),first_chunk=True).squeeze(2);record(f'up_{ic}_{oc}_{ft}',y,expected);assert torch.equal(y,expected)
 if ft==2 and ic!=oc:
  wrong=ref.QwenImage21DupUp3D(ic,oc,ft,2)(a.unsqueeze(2),first_chunk=False)[:,:,0];controls['wrong_first_temporal_subframe']=int((wrong!=expected).sum())
# Above first-subframe witness is independent of floating-point arithmetic.
assert controls['wrong_first_temporal_subframe']>0
receipt=dict(seed=seed,precision='original FP32; TF32 disabled',reference_source_sha256=refsha,kernel_source_sha256=sources,weight_sha256=hashes,command=command,results=results,known_bad_controls=controls,scope='residual and first-frame upsample only; numerical exact failures retained');(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
