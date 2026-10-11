#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Known-structure check of the Laguna-XS-2.1-INT4 packed weight layout.

Fetches a few tensors by HTTP range request (safetensors header, then the tensor bytes:
about 6 MB in total) from the pinned INT4 checkpoint and from the BF16 release, decodes the
packed experts under four conventions (field order x sign convention), and correlates each
with the BF16 weight after the checkpoint's declared weight-side transform: a block-diagonal
Sylvester Hadamard of size 128 (transform_config R1) and the post-attention RMSNorm rescale
(BF16 norm weight folded in, the INT4 checkpoint's own norm weight divided out).

Expected (2026-10-07): only least-significant-first, offset-binary decoding correlates
(INT8 layer 35 gate_proj 0.99997, INT4 layer 1 gate_proj 0.994); the others are near 0 or
negative. Requires numpy. Usage: python3 scripts/laguna/int4_layout_probe.py
"""

import json
import struct
import urllib.request

import numpy as np

INT4 = ("poolside/Laguna-XS-2.1-INT4", "4b7e28abdc0a8b121def816b89d631750bc53c92")
BF16 = ("poolside/Laguna-XS-2.1", "c5f36269bbdbd3f27fddc9a9f9dbae0cf2cf57db")


def _range(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def _index(repo):
    url = f"https://huggingface.co/{repo[0]}/resolve/{repo[1]}/model.safetensors.index.json"
    with urllib.request.urlopen(url) as r:
        return json.load(r)["weight_map"]


def tensor(repo, index, name):
    url = f"https://huggingface.co/{repo[0]}/resolve/{repo[1]}/{index[name]}"
    n = struct.unpack("<Q", _range(url, 0, 7))[0]
    meta = json.loads(_range(url, 8, 8 + n - 1))[name]
    a, b = meta["data_offsets"]
    raw = _range(url, 8 + n + a, 8 + n + b - 1)
    if meta["dtype"] == "BF16":
        u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return u.view(np.float32).reshape(meta["shape"])
    assert meta["dtype"] == "I32", meta
    return np.frombuffer(raw, dtype="<u4").reshape(meta["shape"])


def unpack(packed, bits, msb_first, twos):
    per, mask = 32 // bits, (1 << bits) - 1
    u = np.zeros((packed.shape[0], packed.shape[1] * per), dtype=np.int64)
    for i in range(per):
        shift = bits * (per - 1 - i) if msb_first else bits * i
        u[:, i::per] = (packed >> shift) & mask
    half = 1 << (bits - 1)
    return np.where(u >= half, u - 2 * half, u) if twos else u - half


def hadamard_blocks(dim, block=128):
    h = np.array([[1.0]])
    while h.shape[0] < block:
        h = np.block([[h, h], [h, -h]])
    h /= np.sqrt(block)
    out = np.zeros((dim, dim))
    for i in range(0, dim, block):
        out[i : i + block, i : i + block] = h
    return out


def corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def main():
    i_idx, b_idx = _index(INT4), _index(BF16)
    rot = hadamard_blocks(2048)
    for layer, bits in ((1, 4), (35, 8)):
        p = f"model.layers.{layer}"
        e = f"{p}.mlp.experts.0.gate_proj"
        packed = tensor(INT4, i_idx, f"{e}.weight_packed")
        scale = tensor(INT4, i_idx, f"{e}.weight_scale")
        w = tensor(BF16, b_idx, f"{e}.weight")
        g_bf16 = tensor(BF16, b_idx, f"{p}.post_attention_layernorm.weight")
        g_int = tensor(INT4, i_idx, f"{p}.post_attention_layernorm.weight")
        want = ((w * g_bf16) @ rot) / g_int
        print(f"layer {layer} gate_proj INT{bits}: packed {packed.shape}, scale {scale.shape}")
        for msb in (False, True):
            for twos in (False, True):
                q = unpack(packed, bits, msb, twos)
                deq = q * np.repeat(scale, 128, axis=1)
                order = "msb-first" if msb else "lsb-first"
                sign = "twos-complement" if twos else "offset-binary"
                print(f"  {order:9} {sign:15} corr {corr(deq, want):+.5f}")


if __name__ == "__main__":
    main()
