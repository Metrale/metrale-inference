#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-06: Constructed wide-head F32 attention; report exact failures unchanged.
REPO NEW_OUTPUT. No weights, no model qualification, no performance measurement.
"""
import ctypes, hashlib, json, os, pathlib, subprocess, sys
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
repo,out=map(pathlib.Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=False)
source=out/'image_vae_attention.cu';source.write_bytes((repo/'kernels/gb10/common/image_vae_attention.cu').read_bytes())
wrapper=out/'wrapper.cu';wrapper.write_text('#include "'+str(source.resolve())+'"\n'+r'''
extern "C" int attention(void*x,void*y,unsigned n){image_vae_attention_f32<<<n,128>>>((float*)x,(float*)y,n);return cudaDeviceSynchronize();}
''')
command=[os.environ.get('NVCC', 'nvcc'),'-shared','-Xcompiler','-fPIC','-O3','--fmad=false','-arch=sm_121',str(wrapper),'-o',str(out/'attention.so')]
build=subprocess.run(command,capture_output=True,text=True);(out/'build.log').write_text(build.stdout+build.stderr);build.check_returncode()
lib=ctypes.CDLL(str(out/'attention.so'));lib.attention.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint];lib.attention.restype=ctypes.c_int
torch.cuda.set_per_process_memory_fraction(.85);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(2107)
rows=[];controls={}
def run(name,x):
 n=x.shape[-1]; y=torch.full((1152,n),float('nan'),device='cuda')
 assert lib.attention(x.data_ptr(),y.data_ptr(),n)==0
 q,k,v=[z.T.contiguous()[None,None] for z in x]
 with sdpa_kernel(SDPBackend.MATH):
  expected=torch.nn.functional.scaled_dot_product_attention(q,k,v)[0,0].T.contiguous()
 wide=torch.softmax((q.double()@k.double().transpose(-1,-2))/(1152**.5),dim=-1)@v.double()
 wide=wide[0,0].T.contiguous()
 finite=torch.isfinite(y)&torch.isfinite(expected)
 error=y.double()-expected.double()
 rows.append({'name':name,'pixels':n,'elements':y.numel(),'bit_mismatches':int((y.view(torch.int32)!=expected.view(torch.int32)).sum()),'nonfinite_pairs':int((~finite).sum()),'max_abs_error':float(error.abs().max()),'relative_l2':float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(expected.double()).clamp_min(1e-30)),'native_fp64_max_abs':float((y.double()-wide).abs().max()),'reference_fp64_max_abs':float((expected.double()-wide).abs().max())})
 for label,t in [('input',x),('native',y),('reference',expected)]:
  (out/(name+'-'+label+'.f32')).write_bytes(t.cpu().contiguous().numpy().tobytes())
 return y,expected,(q,k,v)
for n in (1,3,17,65):
 x=torch.randn(3,1152,n,device='cuda')*.25
 y,expected,operands=run('random-'+str(n),x)
 if n==17:
  with sdpa_kernel(SDPBackend.MATH): wrong=torch.nn.functional.scaled_dot_product_attention(*operands,is_causal=True)[0,0].T
  controls['wrong_causal']=int((wrong!=expected).sum())
x=torch.zeros(3,1152,4,device='cuda');x[2]=torch.tensor([0.,2.,4.,6.],device='cuda')
y,_,_=run('uniform-all-pixels',x)
assert torch.equal(y,torch.full_like(y,3.))
controls['first-value-instead-of-all-pixels']=int((x[2]!=y).sum())
x=torch.randn(3,1152,3,device='cuda');x[:2]*=1000
run('large-logits',x)
assert all(controls.values()) and all(r['nonfinite_pairs']==0 for r in rows)
receipt={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256((out/'attention.so').read_bytes()).hexdigest(),'command':command,'torch':torch.__version__,'reference':'SDPA MATH FP32 with TF32 disabled; separate FP64 explicit softmax','rows':rows,'known_bad_controls':controls,'exact_reference_gate_passed':all(r['bit_mismatches']==0 for r in rows),'scope':'constructed attention only; no checkpoint/model/image/performance claim'}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
