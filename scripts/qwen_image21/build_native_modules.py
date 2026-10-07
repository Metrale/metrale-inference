#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Compile explicit diagnostic modules; no serving target registration.
REPO NEW_OUTPUT_DIR. CPU-only nvcc compilation; records source/PTX identities.
"""
import hashlib,json,pathlib,subprocess,sys
repo,out=map(pathlib.Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=False);modules=[]
for name in ['nllb_encoder','image_modulation','dense_gemm_bf16','embed_from_argmax','rms_norm_vanilla']:
 source=out/(name+'.cu');source.write_bytes((repo/'kernels/gb10/common'/source.name).read_bytes());ptx=out/(name+'.ptx')
 command=['nvcc','-ptx','-O3','--fmad=false','-arch=sm_121',str(source),'-o',str(ptx)]
 result=subprocess.run(command,capture_output=True,text=True);(out/(name+'.log')).write_text(result.stdout+result.stderr);result.check_returncode()
 modules.append(dict(name=name,ptx=ptx.name,ptx_sha256=hashlib.sha256(ptx.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),command=command))
(out/'modules.json').write_text(json.dumps(dict(modules=modules),indent=2)+'\n')
