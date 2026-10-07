#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# 2026-10-07: Isolated midpoint diagnostic package; never changes the baseline PTX.
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--primitive-receipt', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--nvcc', default='/usr/local/cuda-13.0/bin/nvcc')
    args = parser.parse_args()
    source = args.root / 'crates/model-layers/tests/cuda/gpt_oss_mxfp4_midpoint.cu'
    incumbent = args.root / 'kernels/gb10/common/gpt_oss_mxfp4_gemv.cu'
    baseline = json.loads(args.baseline.read_text())
    if tuple(baseline[k] for k in ('hardware', 'weight_format', 'kv_format')) != ('gb10', 'mxfp4', 'bf16'):
        raise RuntimeError('requires original GB10 MXFP4/BF16 baseline')
    if baseline.get('policy_override'):
        raise RuntimeError('cannot stack diagnostic policies')
    gate = json.loads(args.primitive_receipt.read_text())
    if not (gate.get('bounded_gate_passed') and gate.get('wide_exponent_limit_observed') and gate.get('experiment_completed_as_predicted')):
        raise RuntimeError('bounded primitive gate or preserved limitation missing')
    if gate.get('kernel_sha256') != sha(source) or gate.get('incumbent_sha256') != sha(incumbent):
        raise RuntimeError('primitive source identity mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    count = 0
    for module in baseline['modules']:
        old = args.baseline.parent / module['ptx']
        if sha(old) != module['ptx_sha256']:
            raise RuntimeError(f'baseline PTX changed: {old}')
        target = args.output / module['ptx']
        target.parent.mkdir(parents=True, exist_ok=True)
        if module['name'] == 'gpt_oss_mxfp4_gemv':
            if module['required_symbols'] != ['gpt_oss_mxfp4_gemv_bf16']:
                raise RuntimeError('unexpected incumbent ABI')
            command = [args.nvcc, '-ptx', '-arch=compute_121', '--fmad=false', '-DGPT_MIDPOINT_OVERRIDE', str(source), '-o', str(target)]
            subprocess.run(command, check=True)
            module.update(baseline_ptx_sha256=module['ptx_sha256'], ptx_sha256=sha(target), source=str(source), source_sha256=sha(source), incumbent_source_sha256=sha(incumbent), command=command)
            module['required_symbols'].append('gpt_oss_midpoint_get_counts')
            ptx = target.read_text()
            for symbol in module['required_symbols']:
                if not re.search(r'\.entry\s+' + re.escape(symbol) + r'\s*\(', ptx):
                    raise RuntimeError(f'compiled diagnostic lacks entry: {symbol}')
            count += 1
        else:
            shutil.copyfile(old, target)
    if count != 1:
        raise RuntimeError('expected exactly one expert module')
    baseline['policy_override'] = dict(scope='diagnostic FP64 exact-FP32-midpoint retries only; wide-exponent failure retained; no production promotion', midpoint_counts=True, max_context_tokens=512, baseline_manifest_sha256=sha(args.baseline), primitive_receipt_sha256=sha(args.primitive_receipt), script_sha256=sha(Path(__file__)))
    shutil.copyfile(args.primitive_receipt, args.output / 'primitive-receipt.json')
    (args.output / 'modules.json').write_text(json.dumps(baseline, indent=2) + '\n')


if __name__ == '__main__':
    main()
