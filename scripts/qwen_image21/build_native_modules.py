#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Compile explicit diagnostic modules; no serving target registration.
REPO NEW_OUTPUT_DIR. CPU-only nvcc compilation; records source/PTX identities.
"""
import hashlib,json,pathlib,subprocess,sys
visual = "--visual" in sys.argv
encoder = "--encoder" in sys.argv
assert not (encoder and visual), "choose one module plan"
repo,out=map(pathlib.Path,[x for x in sys.argv[1:] if x not in ["--visual","--encoder"]]);out.mkdir(parents=True,exist_ok=False);modules=[]
names = ['nllb_encoder','image_modulation','dense_gemm_bf16','embed_from_argmax','rms_norm_vanilla'] + (['rms_norm','gelu'] if visual else [])
if encoder: names = ['image_modulation','dense_gemm_bf16','embed_from_argmax','rms_norm_vanilla','attn_prefill_h128','residual_add']
for name in names:
 source=out/(name+'.cu');source.write_bytes((repo/'kernels/gb10'/('gemma-4-26b-a4b/nvfp4/gelu.cu' if name=='gelu' else 'common/'+source.name)).read_bytes());ptx=out/(name+'.ptx')
 command=['nvcc','-ptx','-O3','--fmad=false','-arch=sm_121',str(source),'-o',str(ptx)]
 result=subprocess.run(command,capture_output=True,text=True);(out/(name+'.log')).write_text(result.stdout+result.stderr);result.check_returncode()
 modules.append(dict(name=name,ptx=ptx.name,ptx_sha256=hashlib.sha256(ptx.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),command=command))
(out/'modules.json').write_text(json.dumps(dict(modules=modules),indent=2)+'\n')
