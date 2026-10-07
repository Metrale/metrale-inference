#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# 2026-10-07: Explicit diagnostic module additions; every baseline PTX remains immutable.
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ['baseline', 'output', 'primitive_receipt', 'policy_identity']:
        p.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument('--nvcc', default='/usr/local/cuda/bin/nvcc')
    a = p.parse_args()
    baseline = json.loads(a.baseline.read_text())
    gate = json.loads(a.primitive_receipt.read_text())
    identity = json.loads(a.policy_identity.read_text())
    source = a.root / 'kernels/gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu'
    if (baseline['hardware'], baseline['weight_format'], baseline['kv_format']) != ('gb10', 'mxfp4', 'bf16'):
        raise ValueError('Requires GB10 MXFP4/BF16KV baseline')
    if gate.get('passed') is not True or not gate.get('cases'):
        raise ValueError('Constructed primitive gate did not pass')
    if not identity.get('all_legacy_equal') or not identity.get('tested_gpt_entry_identical') or identity.get('source_sha256') != sha(source):
        raise ValueError('Reviewed tested-policy/legacy identity does not match source')
    tested = gate.get('source_receipt', {}).get('sources', {}).get('new/kernels/gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu')
    if not tested or identity.get('tested_source_sha256') != tested or identity.get('primitive_receipt_sha256') != sha(a.primitive_receipt):
        raise ValueError('Identity is not bound to this exact primitive receipt/tested source')
    helper = a.root / 'kernels/gb10/common/mx_block_scale.cuh'
    expected_helper = gate.get('source_receipt', {}).get('sources', {}).get('new/kernels/gb10/common/mx_block_scale.cuh')
    if not expected_helper or sha(helper) != expected_helper:
        raise ValueError('Scale helper changed since the primitive gate')
    a.output.mkdir(parents=True, exist_ok=False)
    for module in baseline['modules']:
        old = a.baseline.parent / module['ptx']
        if sha(old) != module['ptx_sha256']:
            raise ValueError('Baseline module changed: ' + str(old))
        if not module['name'].replace('_', '').replace('-', '').isalnum():
            raise ValueError('Unsafe module name')
        target = a.output / (module['name'] + '.ptx')
        module['ptx'] = target.name
        shutil.copyfile(old, target)
    for name, filename, entry in [
        ('gpt_oss_mxfp4_mma', 'moe_w4a16_grouped_gemm.cu', 'moe_w4a16_grouped_gemm_ptrtable_e8m0_gpt'),
        ('moe_v41', 'moe_v41.cu', 'moe_v41_gather_rows'),
    ]:
        if any(m['name'] == name for m in baseline['modules']):
            raise ValueError('Diagnostic must add a distinct module: ' + name)
        src = source.parent / filename
        target = a.output / (name + '.ptx')
        argv = [a.nvcc, '-O3', '--fmad=false', '-arch=sm_121', '-ptx', str(src), '-o', str(target)]
        (a.output / (name + '-command.json')).write_text(json.dumps(argv))
        subprocess.run(argv, check=True)
        if '.entry ' + entry + '(' not in target.read_text():
            raise ValueError('Compiled entry missing: ' + entry)
        baseline['modules'].append(dict(name=name, ptx=target.name, ptx_sha256=sha(target),
            source=str(src), source_sha256=sha(src), command=argv, required_symbols=[entry]))
    baseline['packed_tc_diagnostic'] = dict(scope='Explicit full128 chunk diagnostic; different MMA reduction policy; no serving admission',
        baseline_manifest_sha256=sha(a.baseline), primitive_receipt_sha256=sha(a.primitive_receipt),
        policy_identity_sha256=sha(a.policy_identity), script_sha256=sha(Path(__file__)))
    (a.output / 'modules.json').write_text(json.dumps(baseline, indent=2) + '\n')


if __name__ == '__main__':
    main()
