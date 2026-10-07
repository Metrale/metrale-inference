#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# 2026-10-07: Evaluate Torch CUDA top-k on exactly the stored native BF16 logits, one original batch-shape row at a time.
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--staged',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.cuda.set_per_process_memory_fraction(.85)
    report=dict(scope='same captured BF16 router logits; actual Torch CUDA topk, no tie policy changes',torch=torch.__version__,device=torch.cuda.get_device_name(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=[])
    for tag,root in [('baseline',a.baseline),('staged',a.staged)]:
        for position in [49,215,248]:
            for layer in range(24):
                prefix=f'p{position}-l{layer}'
                lp=root/(prefix+'-router_logits.bin');ip=root/(prefix+'-router_ids.bin')
                bits=np.fromfile(lp,dtype='<i2').copy()
                native=np.fromfile(ip,dtype='<u4').tolist()
                logits=torch.from_numpy(bits).view(torch.bfloat16).reshape(1,32).cuda()
                values,indices=torch.topk(logits,4,dim=-1)
                actual=indices[0].cpu().tolist()
                ranked=torch.sort(logits,descending=True).values[0]
                cutoff=ranked[3];gap=float(cutoff.float()-ranked[4].float())
                ties=torch.where(logits[0]==cutoff)[0].cpu().tolist()
                def bf16_file(name,shape):
                    return torch.from_numpy(np.fromfile(root/(prefix+'-'+name+'.bin'),dtype='<i2').copy()).view(torch.bfloat16).reshape(shape).cuda()
                selected=bf16_file('selected_experts',(4,2880))
                scores=bf16_file('router_scores',(32,))
                moe=bf16_file('moe',(2880,))
                dense=torch.zeros((32,2880),device='cuda',dtype=torch.bfloat16)
                dense[torch.tensor(native,device='cuda')]=selected
                reduced=(dense*scores[:,None]).sum(dim=0)
                reduction_differences=int((reduced.view(torch.int16)!=moe.view(torch.int16)).sum())
                report['rows'].append(dict(reduction_bit_mismatches=reduction_differences,reduction_max_abs=float((reduced.float()-moe.float()).abs().max()),policy=tag,position=position,layer=layer,native_ids=native,torch_cuda_ids=actual,
                                           same_set=sorted(native)==sorted(actual),same_order=native==actual,cutoff_gap=gap,cutoff_tie_ids=ties,
                                           logits_bf16_bits=bits.tolist(),logits_sha256=hashlib.sha256(lp.read_bytes()).hexdigest(),ids_sha256=hashlib.sha256(ip.read_bytes()).hexdigest()))
    report['set_mismatches']=[{k:r[k] for k in ['policy','position','layer','native_ids','torch_cuda_ids','cutoff_gap','cutoff_tie_ids']} for r in report['rows'] if not r['same_set']]
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['set_mismatches'],indent=2))
