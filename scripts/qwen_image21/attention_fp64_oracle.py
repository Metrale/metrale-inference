#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Independent same-input CPU FP64 attention characterization.
ATTENTION_RESULTS NEW_OUTPUT_DIR. Does not change the failed exact SDPA gate.
"""
import hashlib,json,pathlib,sys,torch
previous,out=map(pathlib.Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
receipt=json.loads((previous/'receipt.json').read_text());results=[]
for item in receipt['results']:
 name,case=item['layout'],item['case'];plan=json.loads((previous/f'{name}-plan.json').read_text())
 ids=torch.tensor(plan['image_ids']);valid=torch.tensor(plan['key_valid']);batch,seq=valid.shape
 def read(label): return torch.frombuffer(bytearray((previous/f'{name}-{case}-{label}.bf16').read_bytes()),dtype=torch.bfloat16).reshape(batch,seq,32,128)
 q,k,v=[read(label).transpose(1,2).double() for label in ['q','k','v']]
 positions=torch.arange(seq);mask=(((positions[:,None]>=positions[None,:])|((ids[:,None]==ids[None,:])&(ids[:,None]>=0)))[None,None]&valid[:,None,None,:])
 scores=(q@k.transpose(-1,-2))*(128.0**-.5);prob=torch.nan_to_num(torch.softmax(scores.masked_fill(~mask,float('-inf')),-1),nan=0)
 oracle=(prob@v).transpose(1,2).contiguous();rounded=oracle.bfloat16()
 def metric(x):
  err=(x.double()-oracle).abs();return dict(bf16_differences_from_rounded_fp64=int((x.view(torch.int16)!=rounded.view(torch.int16)).sum()),max_abs_error_from_fp64=float(err.max()),relative_l2_from_fp64=float(torch.linalg.vector_norm(err)/torch.linalg.vector_norm(oracle).clamp_min(1e-30)))
 results.append(dict(layout=name,case=case,elements=rounded.numel(),native=metric(read('native')),sdpa_reference=metric(read('reference'))))
 (out/f'{name}-{case}-oracle.bf16').write_bytes(rounded.view(torch.uint16).numpy().tobytes())
result=dict(source_receipt_sha256=hashlib.sha256((previous/'receipt.json').read_bytes()).hexdigest(),torch=torch.__version__,device='CPU only',precision='FP64 dot/scale/softmax/value sum then BF16 rounding',results=results,scope='independent error characterization; original exact SDPA gate remains failed')
(out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
