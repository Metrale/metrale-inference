#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# 2026-10-07: Build an explicitly named staged-attention A/B package; baseline PTX is never modified.
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--primitive-receipt',type=Path,required=True,help='Reviewed successful primitive receipt, retained verbatim for provenance')
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument('--nvcc',default='/usr/local/cuda-13.0/bin/nvcc')
    a=p.parse_args()
    source=a.root/'kernels/gb10/common/gpt_oss_staged_attention.cu'
    baseline=json.loads(a.baseline.read_text())
    if (baseline['hardware'],baseline['weight_format'],baseline['kv_format'])!=('gb10','mxfp4','bf16'):
        raise RuntimeError('A/B package requires actual GB10 MXFP4/BF16KV baseline')
    if not a.primitive_receipt.is_file():raise FileNotFoundError(a.primitive_receipt)
    gate=json.loads(a.primitive_receipt.read_text())
    if gate.get('passed') is not True or not gate.get('cases'):
        raise RuntimeError('primitive gate did not pass')
    if gate.get('kernel_source_sha256') != sha(source):
        raise RuntimeError('primitive receipt does not match current kernel source')
    a.output.mkdir(parents=True,exist_ok=False)
    count=0
    for module in baseline['modules']:
        old=a.baseline.parent/module['ptx']
        if sha(old)!=module['ptx_sha256']:raise RuntimeError(f'baseline PTX changed: {old}')
        target=a.output/module['ptx']
        if module['name']=='paged_decode':
            if module['required_symbols']!=['paged_decode_attn_sink']:
                raise RuntimeError('unexpected shared ABI requirements')
            command=[a.nvcc,'-ptx','-arch=compute_121','--fmad=false',
                     '-Dgpt_oss_staged_attention_bf16=paged_decode_attn_sink',str(source),'-o',str(target)]
            subprocess.run(command,check=True)
            module.update(baseline_ptx_sha256=module['ptx_sha256'],source=str(source),source_sha256=sha(source),
                          ptx_sha256=sha(target),command=command,flags=['--fmad=false','-Dgpt_oss_staged_attention_bf16=paged_decode_attn_sink'])
            count+=1
        else:
            shutil.copyfile(old,target)
    if count!=1:raise RuntimeError('expected one attention module override')
    baseline['policy_override']=dict(scope='diagnostic A/B only; no runtime/factory default switch',
                                    attention='staged BF16 eager boundaries; finite QKV; maximum4096 tokens',max_context_tokens=4096,
                                    baseline_manifest_sha256=sha(a.baseline),primitive_receipt_sha256=sha(a.primitive_receipt),
                                    script_sha256=sha(Path(__file__)),entry_alias='gpt_oss_staged_attention_bf16 -> paged_decode_attn_sink')
    shutil.copyfile(a.primitive_receipt,a.output/'primitive-receipt.json')
    (a.output/'modules.json').write_text(json.dumps(baseline,indent=2)+'\n')
