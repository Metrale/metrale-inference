#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Native temporal embedding versus pinned source and same-frequency replay.
REPO NATIVE_FREQUENCY_JSON NEW_OUTPUT_DIR. Exhaustive finite BF16 timesteps [0,1].
"""
import ctypes, hashlib, inspect, json, pathlib, subprocess, sys
import torch
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TemporalTimesteps
repo, table, out = map(pathlib.Path, sys.argv[1:]); out.mkdir(parents=True, exist_ok=False)
source=pathlib.Path(inspect.getfile(QwenImage21TemporalTimesteps))
assert hashlib.sha256(source.read_bytes()).hexdigest()=='0eb0555e21ca93195e1fe9389113cafbf0e8822f6454c3001cebb3d21521ebd7'
cu=repo/'kernels/gb10/common/image_modulation.cu'
wrapper=out/'wrapper.cu'; wrapper.write_text('#include "'+str(cu.resolve())+'"\n'+r'''
extern "C" int temporal(void*t,void*f,void*y,unsigned rows){image_timestep_bf16<<<rows,256>>>((__nv_bfloat16*)t,(float*)f,(__nv_bfloat16*)y,rows);return cudaDeviceSynchronize();}
extern "C" int activate(void*g,void*u,void*y,unsigned n){image_silu_staged_mul_bf16<<<(n+255)/256,256>>>((__nv_bfloat16*)g,(__nv_bfloat16*)u,(__nv_bfloat16*)y,n);return cudaDeviceSynchronize();}
''')
command=['nvcc','-shared','-Xcompiler','-fPIC','-O3','--fmad=false','-arch=sm_121',str(wrapper),'-o',str(out/'probe.so')]
p=subprocess.run(command,capture_output=True,text=True); (out/'build.log').write_text(p.stdout+p.stderr);p.check_returncode()
lib=ctypes.CDLL(str(out/'probe.so'))
for fn in [lib.temporal,lib.activate]: fn.argtypes=[ctypes.c_void_p]*3+[ctypes.c_uint];fn.restype=ctypes.c_int
torch.cuda.set_per_process_memory_fraction(.85)
reference=QwenImage21TemporalTimesteps(256)
native=torch.tensor(json.loads(table.read_text()),dtype=torch.int32).view(torch.float32)
frequency_delta=int((native.view(torch.int32)!=reference.freqs.view(torch.int32)).sum())
reference=reference.cuda();native=native.cuda()
t=torch.arange(0x3f81,dtype=torch.int16).view(torch.bfloat16).cuda()
with torch.no_grad(): expected=reference(t).bfloat16()
results={}
for name,freq in [('native_frequency',native),('reference_frequency',reference.freqs)]:
 y=torch.empty_like(expected); assert lib.temporal(t.data_ptr(),freq.data_ptr(),y.data_ptr(),len(t))==0
 results[name]={'elements':y.numel(),'bit_mismatches':int((y.view(torch.int16)!=expected.view(torch.int16)).sum()),'max_abs_error':float((y.float()-expected.float()).abs().max())}
 (out/(name+'.bf16')).write_bytes(y.cpu().view(torch.uint16).numpy().tobytes())
# Both unary time SiLU and existing non-null block SiLU product must retain staging.
g=torch.arange(-32768,32768,dtype=torch.int32).short().view(torch.bfloat16);g=g[torch.isfinite(g)].cuda();u=torch.full_like(g,.75)
for name,up in [('unary',None),('product',u)]:
 expected_activation=torch.nn.functional.silu(g)
 if up is not None: expected_activation=expected_activation*up
 y=torch.empty_like(g);assert lib.activate(g.data_ptr(),0 if up is None else up.data_ptr(),y.data_ptr(),len(g))==0
 results[name]={'elements':len(g),'bit_mismatches':int((y.view(torch.int16)!=expected_activation.view(torch.int16)).sum())}
wrong_angle=t.float()[:,None]*reference.freqs[None,:]
wrong=torch.cat((wrong_angle.cos(),wrong_angle.sin()),-1).bfloat16()
controls={'missing_time_factor':int((wrong.view(torch.int16)!=expected.view(torch.int16)).sum()),'swapped_halves':int((expected.roll(128,-1).view(torch.int16)!=expected.view(torch.int16)).sum())}
receipt={'source_sha256':hashlib.sha256(cu.read_bytes()).hexdigest(),'reference_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'native_frequency_sha256':hashlib.sha256(table.read_bytes()).hexdigest(),'torch':torch.__version__,'command':command,'frequency_bit_mismatches':frequency_delta,'criterion':'exact BF16 bits; failures retained separately from controls','results':results,'known_bad_controls':controls}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2));assert all(controls.values())
