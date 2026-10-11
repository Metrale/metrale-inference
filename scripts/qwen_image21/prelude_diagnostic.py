#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Real block-0 weights, diagnostic native norm/modulation/QKV.
REPO_ROOT CHECKPOINT_DIR NEW_OUTPUT_DIR. No attention or generation claim.
"""
import ctypes
import hashlib
import inspect
import json
import pathlib
import subprocess
import sys
import torch
from safetensors import safe_open
import diffusers.models.transformers.transformer_qwenimage21 as reference

repo, model, out = map(pathlib.Path, sys.argv[1:]); out.mkdir(parents=True,exist_ok=False)
source_sha=hashlib.sha256(pathlib.Path(inspect.getfile(reference)).read_bytes()).hexdigest()
assert source_sha=="0eb0555e21ca93195e1fe9389113cafbf0e8822f6454c3001cebb3d21521ebd7"
component=model/"transformer"
index=json.loads((component/"diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
config=json.loads((component/"config.json").read_text());assert config["num_attention_heads"]*config["attention_head_dim"]==4096 and config["eps"]==1e-6
sources=[repo/"kernels/gb10/common"/name for name in ["nllb_encoder.cu","image_modulation.cu","dense_gemm_bf16.cu"]]
wrapper=out/"wrapper.cu"
wrapper.write_text(''.join('#include "'+str(p.resolve())+'"\n' for p in sources)+r'''
extern "C" int prelude(void* x,void* mod,void* sel,void* one,void* zero,void* norm,void* scaled,void* weight,void* result,unsigned rows) {
 nllb_layernorm_oop_bf16<<<rows,256,1024>>>((__nv_bfloat16*)x,(__nv_bfloat16*)norm,(__nv_bfloat16*)one,(__nv_bfloat16*)zero,rows,4096,1e-6f);
 image_modulation_scale_bf16<<<(uint64_t(rows)*4096+255)/256,256>>>((__nv_bfloat16*)norm,(__nv_bfloat16*)mod,(uint32_t*)sel,(__nv_bfloat16*)scaled,rows,4096,16384,0);
 dense_gemm_bf16_pipelined<<<dim3(32,(rows+127)/128),256>>>((__nv_bfloat16*)scaled,(__nv_bfloat16*)weight,(__nv_bfloat16*)result,rows,4096,4096);
 return cudaDeviceSynchronize();
}
''')
command=["nvcc","-shared","-Xcompiler","-fPIC","-O3","--fmad=false","-arch=sm_121",str(wrapper),"-o",str(out/"prelude.so")]
built=subprocess.run(command,capture_output=True,text=True);(out/"build.log").write_text(built.stdout+built.stderr);built.check_returncode()
lib=ctypes.CDLL(str(out/"prelude.so"));lib.prelude.argtypes=[ctypes.c_void_p]*9+[ctypes.c_uint];lib.prelude.restype=ctypes.c_int
torch.cuda.set_per_process_memory_fraction(0.85);torch.manual_seed(2123)
x=torch.randn(2,3,4096,device="cuda",dtype=torch.bfloat16)
mod=torch.randn(3,16384,device="cuda",dtype=torch.bfloat16)
mask=torch.tensor([False,True,True],device="cuda");selected=torch.tensor([2,0,0,2,1,1],device="cuda",dtype=torch.uint32)
one=torch.ones(4096,device="cuda",dtype=torch.bfloat16);zero=torch.zeros_like(one)
norm,scaled,result=[torch.empty_like(x) for _ in range(3)]
expected_norm=torch.nn.functional.layer_norm(x,(4096,),eps=1e-6)
expected_mod,_=reference.QwenImage21TransformerBlock._modulate(None,expected_norm,mod[:,:8192],mask)
results=[]
def metrics(a,b):
    return {"bf16_bit_mismatches":int((a.view(torch.int16)!=b.view(torch.int16)).sum()),"max_abs_error":float((a.float()-b.float()).abs().max()),"relative_l2":float(torch.linalg.vector_norm(a.float()-b.float())/torch.linalg.vector_norm(b.float()))}
for projection in ["q","k","v"]:
    name=f"transformer_blocks.0.attn.to_{projection}.weight"
    with safe_open(component/index[name],framework="pt",device="cpu") as f: cpu=f.get_tensor(name)
    assert cpu.dtype==torch.bfloat16 and list(cpu.shape)==[4096,4096]
    weight_sha=hashlib.sha256(cpu.view(torch.uint16).numpy().tobytes()).hexdigest();w=cpu.cuda()
    expected=torch.nn.functional.linear(expected_mod,w);torch.cuda.synchronize()
    assert lib.prelude(x.data_ptr(),mod.data_ptr(),selected.data_ptr(),one.data_ptr(),zero.data_ptr(),norm.data_ptr(),scaled.data_ptr(),w.data_ptr(),result.data_ptr(),6)==0
    assert torch.isfinite(result).all()
    for label,tensor in [("native",result),("reference",expected)]: (out/f"{projection}-{label}.bf16").write_bytes(tensor.cpu().view(torch.uint16).numpy().tobytes())
    wrong=torch.nn.functional.linear(expected_mod,w.T.contiguous())
    control=metrics(wrong,expected);assert control["bf16_bit_mismatches"]>0
    results.append({"projection":projection,"weight":name,"shard":index[name],"weight_sha256":weight_sha,"shape":[2,3,4096],"norm":metrics(norm,expected_norm),"modulation":metrics(scaled,expected_mod),"projection_error":metrics(result,expected),"transposed_weight_control":control})
receipt={"kind":"native_diagnostic_block_prelude","checkpoint_revision":"d26bb61231c349cf6b7896fa83353113880e1ba3","reference_source_sha256":source_sha,"source_hashes":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},"compile_command":command,"torch":torch.__version__,"cuda":torch.version.cuda,"seed":2123,"results":results,"qualified":False,"limitations":["known non-exact LayerNorm policy","raw QKV only; no QK normalization, RoPE, attention or block residual","no model or image qualification","no speed acceptance criterion"]}
(out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n");print(json.dumps(receipt,indent=2))
