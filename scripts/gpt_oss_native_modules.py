#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# 2026-10-07: Explicit MXFP4/BF16KV validation package, independent of default NVFP4 target fallback.
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root, output, nvcc):
    manifest = root / 'kernels/circuits/checkpoints/gpt_oss_native_modules.json'
    spec = json.loads(manifest.read_text())
    assert (spec['schema'], spec['hardware'], spec['weight_format'], spec['kv_format']) == (1, 'gb10', 'mxfp4', 'bf16')
    output.mkdir(parents=True, exist_ok=True)
    receipt = dict(spec, manifest_sha256=sha(manifest), compiler=subprocess.check_output([nvcc, '--version'], text=True))
    modules = []
    for item in spec['modules']:
        source = root / item['source']
        target = output / (item['name'] + '.ptx')
        command = [nvcc, '-ptx', '-arch=compute_121', '--fmad=false', *item['flags'], str(source), '-o', str(target)]
        subprocess.run(command, check=True)
        symbols = set(re.findall(r'\.entry\s+([A-Za-z0-9_]+)\s*\(', target.read_text()))
        missing = set(item['required_symbols']) - symbols
        if missing:
            raise RuntimeError(f"{item['name']} missing PTX entries: {sorted(missing)}")
        modules.append(dict(item, ptx=target.name, source_sha256=sha(source), ptx_sha256=sha(target), command=command))
    receipt['modules'] = modules
    (output / 'modules.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def resolve(output):
    # 2026-10-07: Load every module on the real CUDA driver and resolve every required entry; no missing exemptions.
    receipt = json.loads((output / 'modules.json').read_text())
    driver = ctypes.CDLL('libcuda.so.1')
    def checked(name, *args):
        code = getattr(driver, name)(*args)
        if code:
            raise RuntimeError(f'{name}: CUDA error {code}')
    checked('cuInit', 0)
    device = ctypes.c_int()
    checked('cuDeviceGet', ctypes.byref(device), 0)
    context = ctypes.c_void_p()
    checked('cuDevicePrimaryCtxRetain', ctypes.byref(context), device)
    checked('cuCtxSetCurrent', context)
    resolved = []
    try:
        for item in receipt['modules']:
            path = output / item['ptx']
            if sha(path) != item['ptx_sha256']:
                raise RuntimeError(f'PTX digest mismatch: {path}')
            module = ctypes.c_void_p()
            checked('cuModuleLoad', ctypes.byref(module), ctypes.c_char_p(str(path).encode()))
            try:
                for symbol in item['required_symbols']:
                    fn = ctypes.c_void_p()
                    checked('cuModuleGetFunction', ctypes.byref(fn), module, ctypes.c_char_p(symbol.encode()))
                    resolved.append([item['name'], symbol])
            finally:
                checked('cuModuleUnload', module)
    finally:
        checked('cuDevicePrimaryCtxRelease', device)
    (output / 'resolution.json').write_text(json.dumps(dict(resolved=resolved), indent=2) + '\n')
    print(f'Resolved {len(resolved)} required entries in {len(receipt["modules"])} modules')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--nvcc', default='/usr/local/cuda-13.0/bin/nvcc')
    parser.add_argument('--resolve', action='store_true', help='CUDA driver resolution after compilation; requires coordinated free GPU')
    args = parser.parse_args()
    build(args.root.resolve(), args.output.resolve(), args.nvcc)
    if args.resolve:
        resolve(args.output.resolve())
