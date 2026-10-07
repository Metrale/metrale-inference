#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Exact rational oracle for a packed-expert gate or raw down-projection row.

Run beside private checkpoints. Output includes one learned row and must remain
private; it is not a redistributable constructed fixture or a CUDA bmm oracle.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np
from safetensors import safe_open


def main():
    p = argparse.ArgumentParser()
    for field in ['checkpoint', 'snapshots', 'replay', 'output']:
        p.add_argument('--'+field, type=Path, required=True)
    for field in ['position', 'layer', 'expert', 'row']:
        p.add_argument('--'+field, type=int, required=True)
    p.add_argument('--projection', choices=['gate', 'down'], default='gate')
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    index = json.loads((a.checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    projection = 'gate_up_proj' if a.projection == 'gate' else 'down_proj'
    prefix = f'model.layers.{a.layer}.mlp.experts.{projection}_'
    values = {}
    for suffix in ['blocks', 'scales']:
        name = prefix + suffix
        with safe_open(a.checkpoint/index[name], framework='pt', device='cpu') as f:
            values[suffix] = f.get_slice(name)[a.expert][a.row].numpy()
        (a.output/(suffix+'.bin')).write_bytes(values[suffix].tobytes())
    packed = values['blocks'].reshape(-1)
    codes = np.empty(packed.size * 2, dtype=np.uint8)
    codes[::2], codes[1::2] = packed & 15, packed >> 4
    lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
    weights = np.ldexp(lut[codes], np.repeat(values['scales'].astype(np.int32)-127, 32))
    # Actual checkpoint exponent range is finite and exactly representable in BF16.
    as_fp32 = weights.astype(np.float32)
    assert np.isfinite(as_fp32).all()
    assert np.all((as_fp32.view(np.uint32) & 0xffff) == 0), "unpack requires BF16 rounding; unsupported oracle operands"
    assert np.array_equal(as_fp32.astype(float), weights)
    source = (a.snapshots/f'p{a.position}-l{a.layer}-post_attention_norm.bin') if a.projection == 'gate' else (a.replay/f'p{a.position}-l{a.layer}-e{a.expert}-activation-isolated-native.bin')
    input_bytes = source.read_bytes()
    (a.output/'input.bf16').write_bytes(input_bytes)
    bits = np.frombuffer(input_bytes, dtype='<u2')
    x = (bits.astype(np.uint32) << 16).view(np.float32).astype(float)
    assert x.size == weights.size and np.isfinite(x).all()
    exact = sum((Fraction(float(w))*Fraction(float(v)) for w, v in zip(weights, x)), Fraction())
    def decode(n):
        return float(np.array([int(n) << 16], dtype=np.uint32).view(np.float32)[0])
    stage = 'gate-bmm32' if a.projection == 'gate' else 'down-raw-bmm32'
    name = f'p{a.position}-l{a.layer}-e{a.expert}-{stage}'
    native = int(np.fromfile(a.replay/(name+'-native.bin'), dtype='<u2')[a.row])
    reference = int(np.fromfile(a.replay/(name+'-reference.bin'), dtype='<u2')[a.row])
    assert abs(native-reference) <= 1, "oracle neighborhood requires adjacent observed BF16 outputs"
    # Candidate neighborhood includes both observed outputs and their neighbors;
    # check it straddles the exact sum before selecting nearest-even rationally.
    candidates = {n+d for n in [native, reference] for d in [-1, 0, 1]}
    assert min(map(decode, candidates)) <= exact <= max(map(decode, candidates))
    nearest = min(candidates, key=lambda n: (abs(exact-Fraction(decode(n))), n & 1))
    lanes = np.zeros(32, dtype=np.float32)
    for lane in range(32):
        for i in range(lane, x.size, 32):
            # BF16 products are exact in FP32; double intermediate avoids an
            # extra multiplication rounding when emulating the native FMA.
            lanes[lane] = np.float32(float(lanes[lane]) + weights[i]*x[i])
    for shift in [16, 8, 4, 2, 1]:
        old = lanes.copy()
        for lane in range(32-shift):
            lanes[lane] = np.float32(float(old[lane])+float(old[lane+shift]))
    report = dict(projection=a.projection, position=a.position, layer=a.layer, expert=a.expert, row=a.row,
                  exact_numerator=str(exact.numerator), exact_denominator=str(exact.denominator),
                  exact_float=float(exact), native_bf16_bits=native, reference_bf16_bits=reference,
                  correctly_rounded_bf16_bits=nearest, native_warp_fp32_bits=int(lanes[:1].view(np.uint32)[0]),
                  native_warp_value=float(lanes[0]), native_warp_error=float(Fraction(float(lanes[0]))-exact),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  operands_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output.iterdir()},
                  limitation='CPU emulation does not expose internal CUDA bmm accumulation')
    (a.output/'exact-dot.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
