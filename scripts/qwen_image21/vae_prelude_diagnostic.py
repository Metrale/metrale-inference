#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Original-FP32 VAE post-quant/input-conv/norm/SiLU diagnostic.
REPO CHECKPOINT NEW_OUT SEED. Reference F32/cuDNN TF32 disabled; previous
BF16 pipeline image results are a distinct policy, not this comparison.
"""
import ctypes,hashlib,inspect,json,pathlib,subprocess,sys
import torch
from safetensors import safe_open
import diffusers.models.autoencoders.autoencoder_kl_qwenimage21 as reference
repo,model,out=map(pathlib.Path,sys.argv[1:4]);seed=int(sys.argv[4]);out.mkdir(parents=True,exist_ok=False)
refsha=hashlib.sha256(pathlib.Path(inspect.getfile(reference)).read_bytes()).hexdigest();assert refsha=='1e29f252dbddd044b84c8b794589001ce94bb125a8f13eb5b226ef30c1e2ea11'
source=out/'image_vae.cu';source.write_bytes((repo/'kernels/gb10/common/image_vae.cu').read_bytes());wrapper=out/'wrapper.cu';wrapper.write_text('#include "'+str(source.resolve())+'"\n'+r'''
extern "C" int conv(void*x,void*w,void*b,void*y,unsigned in,unsigned out,unsigned h,unsigned width,unsigned k){image_vae_conv2d_f32<<<(out*h*width+127)/128,128>>>((float*)x,(float*)w,(float*)b,(float*)y,in,out,h,width,k);return cudaDeviceSynchronize();}
extern "C" int vae_norm(void*x,void*w,void*y,unsigned c,unsigned p){image_vae_norm_f32<<<p,256>>>((float*)x,(float*)w,(float*)y,c,p);return cudaDeviceSynchronize();}
extern "C" int activate(void*x,void*y,unsigned n){image_vae_silu_f32<<<(n+255)/256,256>>>((float*)x,(float*)y,n);return cudaDeviceSynchronize();}
''')
command=['nvcc','-shared','-Xcompiler','-fPIC','-O3','--fmad=false','-arch=sm_121',str(wrapper),'-o',str(out/'vae.so')];b=subprocess.run(command,capture_output=True,text=True);(out/'build.log').write_text(b.stdout+b.stderr);b.check_returncode();lib=ctypes.CDLL(str(out/'vae.so'))
for name,pointers,ints in [('conv',4,5),('vae_norm',3,2),('activate',2,1)]:
 fn=getattr(lib,name);fn.argtypes=[ctypes.c_void_p]*pointers+[ctypes.c_uint]*ints;fn.restype=ctypes.c_int
torch.cuda.set_per_process_memory_fraction(.85);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(seed)
weights={};hashes={}
with safe_open(model/'vae/diffusion_pytorch_model.safetensors',framework='pt',device='cpu') as f:
 for name in ['post_quant_conv.weight','post_quant_conv.bias','decoder.conv_in.weight','decoder.conv_in.bias','decoder.mid_block.resnets.0.norm1.gamma']:
  w=f.get_tensor(name);assert w.dtype==torch.float32;hashes[name]=hashlib.sha256(w.numpy().tobytes()).hexdigest();weights[name]=w.cuda()
results=[]
def call(name,*args):assert getattr(lib,name)(*(a.data_ptr() if isinstance(a,torch.Tensor) else a for a in args))==0
def record(name,a,b):
 finite=torch.isfinite(a)&torch.isfinite(b);av,bv=a[finite].double(),b[finite].double();error=av-bv
 results.append(dict(stage=name,elements=a.numel(),f32_bit_mismatches=int((a.view(torch.int32)!=b.view(torch.int32)).sum()),nonfinite_pairs=int((~finite).sum()),max_abs_error_finite=float(error.abs().max()),relative_l2_finite=float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(bv).clamp_min(1e-30))))
 for label,t in [('native',a),('reference',b)]: (out/(name+'-'+label+'.f32')).write_bytes(t.detach().cpu().contiguous().numpy().tobytes())
x=torch.randn(1,64,5,7,device='cuda',dtype=torch.float32)
for stage,name in [('post_quant','post_quant_conv'),('conv_in','decoder.conv_in')]:
 w,bias=weights[name+'.weight'],weights[name+'.bias'];y=torch.empty(1,w.shape[0],5,7,device='cuda');call('conv',x,w,bias,y,w.shape[1],w.shape[0],5,7,w.shape[2]);record(stage,y,torch.nn.functional.conv2d(x,w,bias,padding=w.shape[2]//2));x=y
norm=reference.QwenImage21RMS_norm(1152,images=False).cuda();norm.gamma.data.copy_(weights['decoder.mid_block.resnets.0.norm1.gamma']);normal=torch.empty_like(x);call('vae_norm',x,norm.gamma,normal,1152,35);record('norm',normal,norm(x.unsqueeze(2)).squeeze(2));activated=torch.empty_like(normal);call('activate',normal,activated,normal.numel());record('silu',activated,torch.nn.functional.silu(normal))
# Independent integer-exact padding/axes corpus and tiny-input clamp witness.
a=torch.arange(2*3*5,device='cuda',dtype=torch.float32).reshape(1,2,3,5);w=torch.arange(3*2*3*3,device='cuda',dtype=torch.float32).reshape(3,2,3,3);bias=torch.arange(3,device='cuda',dtype=torch.float32);y=torch.empty(1,3,3,5,device='cuda');call('conv',a,w,bias,y,2,3,3,5,3);expected=torch.nn.functional.conv2d(a,w,bias,padding=1);record('integer_padding',y,expected);assert torch.equal(y,expected)
wrong=torch.nn.functional.conv2d(a,w.flip(-1),bias,padding=1);controls={'flipped_kernel':int((wrong!=expected).sum())}
tiny=torch.full((1,4,3,5),1e-14,device='cuda');gamma=torch.ones(4,device='cuda');n=torch.empty_like(tiny);call('vae_norm',tiny,gamma,n,4,15);expected=torch.nn.functional.normalize(tiny,dim=1)*2;record('tiny_clamp',n,expected);controls['rms_epsilon_substitution']=int((tiny*torch.rsqrt(tiny.square().mean(1,keepdim=True)+1e-6)!=expected).sum())
receipt=dict(seed=seed,checkpoint_precision='FP32 retained',reference_precision='FP32; cuDNN and matmul TF32 disabled',reference_source_sha256=refsha,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),weight_sha256=hashes,command=command,results=results,known_bad_controls=controls,criterion='exact F32 bits reported; numerical failures retained',scope='VAE decoder prelude only; no image-generation support');(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2));assert all(controls.values())
