#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Native three-axis complex RoPE with production Rust tables.
REPO_ROOT RUST_TABLE_DIR QK_OUTPUT_DIR NEW_OUTPUT_DIR SEED. Rust tables must be
exported from the checked-in production layout with the pinned Rust compiler.
"""
import ctypes,hashlib,inspect,json,pathlib,subprocess,sys
import torch
import diffusers.models.transformers.transformer_qwenimage21 as reference
repo,tables,qk,out=map(pathlib.Path,sys.argv[1:5]);seed=int(sys.argv[5]);out.mkdir(parents=True,exist_ok=False)
refsha=hashlib.sha256(pathlib.Path(inspect.getfile(reference)).read_bytes()).hexdigest();assert refsha=="0eb0555e21ca93195e1fe9389113cafbf0e8822f6454c3001cebb3d21521ebd7"
source=repo/"kernels/gb10/common/image_modulation.cu";snapshot=out/"image_modulation.cu";snapshot.write_bytes(source.read_bytes());wrapper=out/"wrapper.cu"
wrapper.write_text('#include "'+str(snapshot.resolve())+'"\n'+r'''
extern "C" int rotate(void* x,void* cis,void* out,unsigned samples,unsigned sequence) {
 image_rope_complex_bf16<<<(uint64_t(samples)*sequence*32*64+255)/256,256>>>((__nv_bfloat16*)x,(float*)cis,(__nv_bfloat16*)out,samples,sequence,32);
 return cudaDeviceSynchronize();
}
''')
command=["nvcc","-shared","-Xcompiler","-fPIC","-O3","--fmad=false","-arch=sm_121",str(wrapper),"-o",str(out/"rope.so")]
build=subprocess.run(command,capture_output=True,text=True);(out/"build.log").write_text(build.stdout+build.stderr);build.check_returncode()
lib=ctypes.CDLL(str(out/"rope.so"));lib.rotate.argtypes=[ctypes.c_void_p]*3+[ctypes.c_uint]*2;lib.rotate.restype=ctypes.c_int
torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(seed)
results=[];controls={"conjugate_frequency":0,"split_half_pairing":0}
for name in ["saved","mixed","wide"]:
    tablefile=tables/f"{name}-pinned-table.json";data=json.loads(tablefile.read_text());seq=len(data["mask"])
    native_cpu=torch.tensor(data["cis_f32_bits"],dtype=torch.uint32).view(torch.float32).reshape(seq,64,2)
    ref_cpu=reference.QwenImage21Rope(theta=10000,axes_dim=[16,56,56])(data["shapes"],torch.tensor(data["mask"]),torch.device("cpu"))
    cis=native_cpu.cuda();expected_cis=ref_cpu.cuda()
    (out/f"{name}-table.json").write_bytes(tablefile.read_bytes())
    (out/f"{name}-reference-cis.f32").write_bytes(torch.view_as_real(ref_cpu).numpy().tobytes())
    if name=="saved":
        inputs={p:torch.frombuffer(bytearray((qk/f"{p}-native.bf16").read_bytes()),dtype=torch.bfloat16).reshape(2,seq,32,128).cuda() for p in ["q","k"]}
    else:
        random=torch.randn(2,seq,32,128,device="cuda",dtype=torch.bfloat16)
        inputs={"random":random,"tiny":random*1e-4,"large":random*1e4}
    for case,x in inputs.items():
        expected=reference.apply_rotary_emb_qwen(x,expected_cis,use_real=False);actual=torch.empty_like(x);torch.cuda.synchronize()
        assert lib.rotate(x.data_ptr(),cis.data_ptr(),actual.data_ptr(),2,seq)==0
        mismatches=int((actual.view(torch.int16)!=expected.view(torch.int16)).sum())
        result=dict(layout=name,case=case,elements=x.numel(),bf16_bit_mismatches=mismatches,max_abs_error=float((actual.float()-expected.float()).abs().max()),frequency_bit_mismatches=int((native_cpu.view(torch.int32)!=torch.view_as_real(ref_cpu).view(torch.int32)).sum()),frequency_max_abs_error=float((native_cpu-torch.view_as_real(ref_cpu)).abs().max()),table_sha256=hashlib.sha256(tablefile.read_bytes()).hexdigest())
        same_freq=torch.empty_like(x)
        ref_cis_real=torch.view_as_real(expected_cis).contiguous()
        assert lib.rotate(x.data_ptr(),ref_cis_real.data_ptr(),same_freq.data_ptr(),2,seq)==0
        result["same_frequency_bf16_mismatches"]=int((same_freq.view(torch.int16)!=expected.view(torch.int16)).sum())
        results.append(result)
        for label,t in [("input",x),("native",actual),("reference",expected),("same-frequency",same_freq)]: (out/f"{name}-{case}-{label}.bf16").write_bytes(t.cpu().view(torch.uint16).numpy().tobytes())
        wrong=reference.apply_rotary_emb_qwen(x,expected_cis.conj(),use_real=False)
        controls["conjugate_frequency"]+=int((wrong.view(torch.int16)!=expected.view(torch.int16)).sum())
        split=torch.complex(x[...,:64].float(),x[...,64:].float())*expected_cis.unsqueeze(1)
        wrong=torch.cat([split.real,split.imag],dim=-1).bfloat16()
        controls["split_half_pairing"]+=int((wrong.view(torch.int16)!=expected.view(torch.int16)).sum())
receipt=dict(seed=seed,reference_source_sha256=refsha,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),torch=torch.__version__,cuda=torch.version.cuda,compile_command=command,criterion="exact BF16 outputs",results=results,known_bad_controls=controls,scope="three-axis rotation only; no attention/block qualification")
(out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n");print(json.dumps(receipt,indent=2))
assert all(r["bf16_bit_mismatches"]==0 for r in results),"native rotary rounding differs"
assert all(controls.values())
