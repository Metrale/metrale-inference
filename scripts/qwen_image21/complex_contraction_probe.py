#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Isolate complex contraction with identical saved BF16/cis operands.
Usage: complex_contraction_probe.py NEW_OUTPUT_DIR. This tests arithmetic only.
"""
import ctypes,json,pathlib,subprocess,sys,torch
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Rope,apply_rotary_emb_qwen
out=pathlib.Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
source=r'''
#include <cuda_bf16.h>
__global__ void calc(const __nv_bfloat16*x,const float*c,__nv_bfloat16*y,unsigned n,unsigned seq,unsigned mode){
unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n/2)return;
unsigned p=((i/(32*64))%seq)*64+i%64;float a=__bfloat162float(x[2*i]),b=__bfloat162float(x[2*i+1]),u=c[2*p],v=c[2*p+1];
float r=mode==4?__fsub_rn(__fmul_rn(a,u),__fmul_rn(b,v)):(mode&1?__fmaf_rn(-b,v,__fmul_rn(a,u)):__fmaf_rn(a,u,-__fmul_rn(b,v)));
float s=mode==4?__fadd_rn(__fmul_rn(a,v),__fmul_rn(b,u)):(mode&2?__fmaf_rn(b,u,__fmul_rn(a,v)):__fmaf_rn(a,v,__fmul_rn(b,u)));
y[2*i]=__float2bfloat16_rn(r);y[2*i+1]=__float2bfloat16_rn(s);
}
extern "C" int run(void*x,void*c,void*y,unsigned n,unsigned seq,unsigned mode){calc<<<(n/2+255)/256,256>>>((__nv_bfloat16*)x,(float*)c,(__nv_bfloat16*)y,n,seq,mode);return cudaDeviceSynchronize();}
'''
(out/'test.cu').write_text(source);subprocess.run(['nvcc','-shared','-Xcompiler','-fPIC','-O3','--fmad=false','-arch=sm_121',str(out/'test.cu'),'-o',str(out/'test.so')],check=True)
lib=ctypes.CDLL(str(out/'test.so'));lib.run.argtypes=[ctypes.c_void_p]*3+[ctypes.c_uint]*3
mask=[False]*256+[True]*9+[False];seq=len(mask)
freq=QwenImage21Rope(theta=10000,axes_dim=[16,56,56])([[1,3,3]],torch.tensor(mask),torch.device('cpu')).cuda();cis=torch.view_as_real(freq).contiguous()
(out/'cis.f32').write_bytes(cis.cpu().numpy().tobytes())
torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(2127)
results=[]
for scale in [1,1e-4,1e4]:
 x=torch.randn(2,seq,32,128,device='cuda',dtype=torch.bfloat16)*scale;expected=apply_rotary_emb_qwen(x,freq,use_real=False);actual=torch.empty_like(x)
 (out/f'{scale}-input.bf16').write_bytes(x.cpu().view(torch.uint16).numpy().tobytes());(out/f'{scale}-reference.bf16').write_bytes(expected.cpu().view(torch.uint16).numpy().tobytes())
 for mode in range(5):
  assert lib.run(x.data_ptr(),cis.data_ptr(),actual.data_ptr(),x.numel(),seq,mode)==0
  (out/f'{scale}-mode{mode}.bf16').write_bytes(actual.cpu().view(torch.uint16).numpy().tobytes())
  bad=(actual.view(torch.int16)!=expected.view(torch.int16)).reshape(-1,2).sum(0).tolist();results.append(dict(scale=scale,mode=mode,bad_real_imag=bad))
(out/'receipt.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))
