#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Bounded native modulation parity, never model qualification.
Run in the pinned Qwen reference environment: script REPO_ROOT NEW_OUTPUT_DIR.
Requires nvcc/CUDA; allocates only small diagnostic tensors, no checkpoint.
"""
import ctypes
import hashlib
import inspect
import json
import pathlib
import subprocess
import sys
import torch
import diffusers.models.transformers.transformer_qwenimage21 as reference

repo, out = map(pathlib.Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=False)
source = pathlib.Path(inspect.getfile(reference))
expected_sha = "0eb0555e21ca93195e1fe9389113cafbf0e8822f6454c3001cebb3d21521ebd7"
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_sha, "reference source pin differs"
kernel = repo / "kernels/gb10/common/image_modulation.cu"
wrapper = out / "wrapper.cu"
wrapper.write_text('#include "' + str(kernel.resolve()) + '"\n' + r'''
extern "C" int scale(void* x,void* p,void* sel,void* out,unsigned rows,unsigned width,unsigned stride,unsigned off) {
 image_modulation_scale_bf16<<<(uint64_t(rows)*width+255)/256,256>>>((__nv_bfloat16*)x,(__nv_bfloat16*)p,(uint32_t*)sel,(__nv_bfloat16*)out,rows,width,stride,off);
 return cudaDeviceSynchronize();
}
extern "C" int residual(void* x,void* branch,void* p,void* sel,void* out,unsigned rows,unsigned width,unsigned stride,unsigned off) {
 image_modulation_residual_bf16<<<(uint64_t(rows)*width+255)/256,256>>>((__nv_bfloat16*)x,(__nv_bfloat16*)branch,(__nv_bfloat16*)p,(uint32_t*)sel,(__nv_bfloat16*)out,rows,width,stride,off);
 return cudaDeviceSynchronize();
}
''')
command = ["nvcc", "-shared", "-Xcompiler", "-fPIC", "-O3", "--fmad=false", "-arch=sm_121", str(wrapper), "-o", str(out / "modulation.so")]
subprocess.run(command, check=True, capture_output=True)
lib = ctypes.CDLL(str(out / "modulation.so"))
lib.scale.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_uint]*4
lib.residual.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_uint]*4
lib.scale.restype = lib.residual.restype = ctypes.c_int
assert torch.cuda.is_available()
torch.cuda.set_per_process_memory_fraction(0.85)
torch.manual_seed(2121)
results = []
controls = {"missing_scale_rounding": 0, "missing_product_rounding": 0, "wrong_prefix_row": 0, "sigmoid_gate": 0}
for samples, tokens, width, causal in [(2,3,4096,True),(1,1,257,False),(1,1,65280,False)]:
    hidden = torch.randn(samples,tokens,width,device="cuda",dtype=torch.bfloat16)
    branch = torch.randn_like(hidden)
    params = torch.randn(samples+int(causal),width*4,device="cuda",dtype=torch.bfloat16)
    if width == 65280:
        # All finite BF16 encodings as gate values, including subnormals and signed zero.
        bits = torch.arange(65536,dtype=torch.int32)
        finite = bits[(bits & 0x7f80) != 0x7f80].to(torch.int16).view(torch.bfloat16).cuda()
        params[0,width:2*width] = finite
    mask = torch.tensor([False,True,False],device="cuda") if causal else None
    selected = torch.tensor([samples if causal and not bool(mask[t]) else s for s in range(samples) for t in range(tokens)],dtype=torch.uint32,device="cuda")
    for half in [0,1]:
        normalized = torch.nn.functional.layer_norm(hidden,(width,),eps=1e-6)
        mod, gate = reference.QwenImage21TransformerBlock._modulate(None,normalized,params[:,half*2*width:(half+1)*2*width],mask)
        expected = hidden + gate.tanh()*branch
        actual_scale, actual_residual = torch.empty_like(hidden), torch.empty_like(hidden)
        torch.cuda.synchronize()
        assert lib.scale(normalized.data_ptr(),params.data_ptr(),selected.data_ptr(),actual_scale.data_ptr(),samples*tokens,width,width*4,half*2*width)==0
        assert lib.residual(hidden.data_ptr(),branch.data_ptr(),params.data_ptr(),selected.data_ptr(),actual_residual.data_ptr(),samples*tokens,width,width*4,(half*2+1)*width)==0
        def mismatch(a,b):
            # Compare exact BF16 bits; NaNs are not used by the gate corpus.
            return int((a.view(torch.int16)!=b.view(torch.int16)).sum().item())
        result = dict(samples=samples,tokens=tokens,width=width,causal=causal,half=half,scale_mismatches=mismatch(actual_scale,mod),residual_mismatches=mismatch(actual_residual,expected),elements=hidden.numel())
        results.append(result)
        selected_scale = reference._select_modulation_rows(params[:,half*2*width:(half*2+1)*width],mask)
        controls["missing_scale_rounding"] += mismatch((normalized.float()*(1+selected_scale.float())).bfloat16(),mod)
        controls["missing_product_rounding"] += mismatch((hidden.float()+gate.tanh().float()*branch.float()).bfloat16(),expected)
        controls["sigmoid_gate"] += mismatch(hidden+gate.sigmoid()*branch,expected)
        if causal:
            wrong,_ = reference.QwenImage21TransformerBlock._modulate(None,normalized,params[:-1,half*2*width:(half+1)*2*width],None)
            controls["wrong_prefix_row"] += mismatch(wrong,mod)
receipt = dict(reference_revision="c6df88a511a98740646ee55577b590c9852650ce",reference_source_sha256=expected_sha,kernel_sha256=hashlib.sha256(kernel.read_bytes()).hexdigest(),compile_command=command,torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),criterion="exact BF16 bits",results=results,known_bad_controls=controls,scope="modulation and gated residual only; no norm/projection or model qualification")
(out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
assert all(r["scale_mismatches"]==r["residual_mismatches"]==0 for r in results)
assert all(controls.values()), "a known-bad precision/selection control escaped"
