#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Evaluate unchanged NLLB LayerNorm reuse at Qwen's width/epsilon.
Candidate evidence only. REPO_ROOT NEW_OUTPUT_DIR. Exact BF16 criterion; no weights.
"""
import ctypes
import hashlib
import json
import pathlib
import subprocess
import sys
import torch

repo, out = map(pathlib.Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=False)
source = repo / "kernels/gb10/common/nllb_encoder.cu"
wrapper = out / "wrapper.cu"
wrapper.write_text('#include "'+str(source.resolve())+'"\n'+r'''
extern "C" int image_norm(void* x,void* out,void* one,void* zero,unsigned rows,unsigned width,float eps) {
 nllb_layernorm_oop_bf16<<<rows,256,256*sizeof(float)>>>((__nv_bfloat16*)x,(__nv_bfloat16*)out,(__nv_bfloat16*)one,(__nv_bfloat16*)zero,rows,width,eps);
 return cudaDeviceSynchronize();
}
''')
command=["nvcc","-shared","-Xcompiler","-fPIC","-O3","--fmad=false","-arch=sm_121",str(wrapper),"-o",str(out/"norm.so")]
compiled=subprocess.run(command,capture_output=True,text=True)
(out/"build.log").write_text(compiled.stdout+compiled.stderr)
compiled.check_returncode()
lib=ctypes.CDLL(str(out/"norm.so")); lib.image_norm.argtypes=[ctypes.c_void_p]*4+[ctypes.c_uint]*2+[ctypes.c_float];lib.image_norm.restype=ctypes.c_int
torch.cuda.set_per_process_memory_fraction(0.85); torch.manual_seed(2122)
width=4096
ones=torch.ones(width,device="cuda",dtype=torch.bfloat16);zeros=torch.zeros_like(ones)
random=torch.randn(65,width,device="cuda",dtype=torch.bfloat16)
cases={"random":random,"constant":torch.full_like(random,17),"tiny":random*1e-4,"large_offset":(1024+random.float()*8).bfloat16(),"small_variance":(1+random.float()*0.005).bfloat16(),"large_magnitude":random*1e10}
results=[];controls={"rmsnorm_substitution":0,"epsilon_1e5":0}
for name,x in cases.items():
    expected=torch.nn.functional.layer_norm(x,(width,),eps=1e-6)
    actual=torch.empty_like(x);torch.cuda.synchronize()
    assert lib.image_norm(x.data_ptr(),actual.data_ptr(),ones.data_ptr(),zeros.data_ptr(),x.shape[0],width,1e-6)==0
    neq=actual.view(torch.int16)!=expected.view(torch.int16)
    mismatch=int(neq.sum().item())
    result={"case":name,"rows":x.shape[0],"width":width,"elements":x.numel(),"bf16_bit_mismatches":mismatch,"max_abs_error":float((actual.float()-expected.float()).abs().max().item())}
    if mismatch:
        positions=neq.nonzero()[:8];result["first_differences"]=[{"row":int(i),"col":int(j),"native":float(actual[i,j]),"reference":float(expected[i,j])} for i,j in positions]
    results.append(result)
    rms=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).bfloat16()
    wrong_eps=torch.nn.functional.layer_norm(x,(width,),eps=1e-5)
    controls["rmsnorm_substitution"]+=int((rms.view(torch.int16)!=expected.view(torch.int16)).sum())
    controls["epsilon_1e5"]+=int((wrong_eps.view(torch.int16)!=expected.view(torch.int16)).sum())
receipt={"kind":"candidate_unchanged_layernorm_reuse","source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"compile_command":command,"torch":torch.__version__,"cuda":torch.version.cuda,"criterion":"exact BF16 bits","results":results,"known_bad_controls":controls,"scope":"native norm candidate only; no modulation or model qualification"}
(out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n");print(json.dumps(receipt,indent=2))
assert all(x["bf16_bit_mismatches"]==0 for x in results),"candidate norm is not exact; preserve failures"
assert all(controls.values())
