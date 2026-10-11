#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Staged Q/K norm, actual saved projections plus adverse controls.
REPO_ROOT CHECKPOINT_DIR PRELUDE_OUTPUT_DIR NEW_OUTPUT_DIR. No full-block claim.
"""
import ctypes,hashlib,inspect,json,pathlib,subprocess,sys
import torch
from safetensors import safe_open
from diffusers.models.normalization import RMSNorm
repo,model,previous,out=map(pathlib.Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=False)
reference_sha=hashlib.sha256(pathlib.Path(inspect.getfile(RMSNorm)).read_bytes()).hexdigest()
manifest=json.loads((repo/"docs/model-manifests/qwen-image-2.1.json").read_text())
expected=next(x["sha256"] for x in manifest["reference_sources"]["diffusers"]["files"] if x["path"].endswith("/normalization.py"));assert reference_sha==expected
sources=[repo/"kernels/gb10/common"/p for p in ["rms_norm_vanilla.cu","image_modulation.cu"]]
wrapper=out/"wrapper.cu";wrapper.write_text(''.join('#include "'+str(p.resolve())+'"\n' for p in sources)+r'''
extern "C" int head_norm(void* x,void* one,void* w,void* tmp,void* out,unsigned rows) {
 rms_norm_vanilla<<<rows,128>>>((__nv_bfloat16*)x,(__nv_bfloat16*)one,(__nv_bfloat16*)tmp,128,1e-6f);
 image_head_weight_bf16<<<(uint64_t(rows)*128+255)/256,256>>>((__nv_bfloat16*)tmp,(__nv_bfloat16*)w,(__nv_bfloat16*)out,rows,128);
 return cudaDeviceSynchronize();
}
''')
command=["nvcc","-shared","-Xcompiler","-fPIC","-O3","--fmad=false","-arch=sm_121",str(wrapper),"-o",str(out/"qknorm.so")]
build=subprocess.run(command,capture_output=True,text=True);(out/"build.log").write_text(build.stdout+build.stderr);build.check_returncode()
lib=ctypes.CDLL(str(out/"qknorm.so"));lib.head_norm.argtypes=[ctypes.c_void_p]*5+[ctypes.c_uint];lib.head_norm.restype=ctypes.c_int
torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(2124)
component=model/"transformer";index=json.loads((component/"diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
one=torch.ones(128,device="cuda",dtype=torch.bfloat16)
results=[];controls={"fused_weight_before_rounding":0,"wrong_epsilon":0}
for name in ["q","k"]:
    wn=f"transformer_blocks.0.attn.norm_{name}.weight"
    with safe_open(component/index[wn],framework="pt",device="cpu") as f: w=f.get_tensor(wn).cuda()
    assert w.dtype==torch.bfloat16 and w.shape==(128,)
    layer=RMSNorm(128,eps=1e-6).cuda().bfloat16();layer.weight.data.copy_(w)
    saved=torch.frombuffer(bytearray((previous/f"{name}-native.bf16").read_bytes()),dtype=torch.bfloat16).reshape(-1,128).cuda()
    random=torch.randn(193,128,device="cuda",dtype=torch.bfloat16)
    cases={"saved_projection":saved,"random":random,"tiny":random*1e-4,"large":random*1e4,"constant":torch.full_like(random,3),"small_variance":(1+random.float()*.005).bfloat16()}
    for case,x in cases.items():
        with torch.no_grad(): expected=layer(x)
        tmp,actual=torch.empty_like(x),torch.empty_like(x);torch.cuda.synchronize()
        assert lib.head_norm(x.data_ptr(),one.data_ptr(),w.data_ptr(),tmp.data_ptr(),actual.data_ptr(),x.shape[0])==0
        neq=actual.view(torch.int16)!=expected.view(torch.int16)
        results.append(dict(head=name,case=case,elements=x.numel(),bf16_bit_mismatches=int(neq.sum()),max_abs_error=float((actual.float()-expected.float()).abs().max())))
        if case=="saved_projection":
            for label,t in [("native",actual),("reference",expected)]: (out/f"{name}-{label}.bf16").write_bytes(t.cpu().view(torch.uint16).numpy().tobytes())
        variance=x.float().square().mean(-1,keepdim=True)
        fused=(x.float()*torch.rsqrt(variance+1e-6)*w.float()).bfloat16()
        wrong=((x.float()*torch.rsqrt(variance+1e-5)).bfloat16()*w)
        controls["fused_weight_before_rounding"]+=int((fused.view(torch.int16)!=expected.view(torch.int16)).sum())
        controls["wrong_epsilon"]+=int((wrong.view(torch.int16)!=expected.view(torch.int16)).sum())
receipt=dict(reference_source_sha256=reference_sha,checkpoint_revision=manifest["revision"],source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},torch=torch.__version__,cuda=torch.version.cuda,compile_command=command,criterion="exact BF16 bits",results=results,known_bad_controls=controls,scope="staged QK normalization only; no RoPE/attention/block qualification")
(out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n");print(json.dumps(receipt,indent=2))
assert all(x["bf16_bit_mismatches"]==0 for x in results),"staged normalization still has a reduction discrepancy"
assert all(controls.values())
