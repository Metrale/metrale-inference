# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Torch CUDA staged BF16 reference corpus, all constructed operands.

Pinned expression source: Transformers v4.55.0 modeling_gpt_oss.py lines110-119.
Generate on the authorized GPU, run the standalone native CUDA harness, compare.
Acceptance fixed before execution: BF16 bit equality, ignoring only NaN payloads.
This is primitive parity, not a model or performance qualification.
"""
import argparse
import json
import pathlib
import struct
import torch


def raw(tensor):
    return tensor.contiguous().cpu().view(torch.uint8).numpy().tobytes()


def write(root, name, header, *tensors):
    (root / name).write_bytes(struct.pack("<" + "I" * len(header), *header) + b"".join(raw(t) for t in tensors))


def generate(root):
    device = "cuda"
    rows, cols = 3, 259
    bias = torch.linspace(-2, 2, cols, device=device).bfloat16()
    values = torch.linspace(-16, 16, rows * cols, device=device).reshape(rows, cols).bfloat16()
    write(root, "bias-input.bin", [rows, cols], values, bias)
    write(root, "bias-expected.bin", [], values + bias)
    write(root, "bias-known-bad.bin", [], values)

    # 2026-10-06: Every possible BF16 gate encoding plus tail rows. Up includes clamp,
    # offset and rounding boundaries. NaNs/infinities are deliberate controls.
    bits = torch.arange(65536, dtype=torch.int32).to(torch.uint16)
    gate = torch.cat([bits.view(torch.bfloat16), torch.linspace(-8, 8, 259).bfloat16()]).to(device)
    pattern = torch.tensor([-8, -7.03125, -7, -1.00390625, -1, -.99609375, 0, .00390625, 1, 6.96875, 7, 7.03125, 8], device=device).bfloat16()
    up = pattern[torch.arange(gate.numel(), device=device) % len(pattern)]
    pairs = torch.stack([gate, up], dim=-1)
    g, u = gate.clamp(max=7), up.clamp(min=-7, max=7)
    glu = g * torch.sigmoid(g * 1.702)
    expected = (u + 1) * glu
    bad_fused = ((u.float() + 1) * g.float() * torch.sigmoid(g.float() * 1.702)).bfloat16()
    bad_symmetric = (u + 1) * (g.clamp(min=-7) * torch.sigmoid(g.clamp(min=-7) * 1.702))
    write(root, "activation-input.bin", [gate.numel()], pairs)
    write(root, "activation-expected.bin", [], expected)
    write(root, "activation-known-bad.bin", [], bad_fused)
    write(root, "activation-symmetric-bad.bin", [], bad_symmetric)

    tokens, hidden = 3, 259
    ids = torch.tensor([[31, 2, 17, 0], [7, 30, 1, 11], [3, 19, 8, 4]], device=device, dtype=torch.int64)
    scores = torch.zeros(tokens, 32, device=device, dtype=torch.bfloat16)
    scores.scatter_(1, ids, torch.tensor([.11, .19, .31, .39], device=device).bfloat16().expand(tokens, -1))
    data = torch.arange(4 * tokens * hidden, device=device, dtype=torch.float32)
    selected = ((data % 71 - 35) / 7).bfloat16().reshape(4, tokens, hidden)
    dense = torch.zeros(32, tokens, hidden, device=device, dtype=torch.bfloat16)
    for token in range(tokens):
        dense[ids[token], token] = selected[:, token]
    expected = (dense * scores.T[..., None]).sum(dim=0)
    bad = (dense.float() * scores.T[..., None].float()).sum(dim=0).bfloat16()
    write(root, "reduce-input.bin", [tokens, hidden], selected, scores, ids.to(torch.uint32))
    write(root, "reduce-expected.bin", [], expected)
    write(root, "reduce-known-bad.bin", [], bad)
    (root / "reference.json").write_text(json.dumps(dict(torch=torch.__version__, device=torch.cuda.get_device_name(), seed="deterministic constructed formulas", acceptance="BF16 exact except NaN payload", expression_source="Transformers v4.55.0 GptOssExperts inference", sparse_boundary="unselected finite outputs only"), indent=2) + "\n")


def bits(root, name):
    return torch.frombuffer(bytearray((root / name).read_bytes()), dtype=torch.uint16).to(torch.int32)


def same(a, b):
    nan_a = ((a & 0x7f80) == 0x7f80) & ((a & 0x7f) != 0)
    nan_b = ((b & 0x7f80) == 0x7f80) & ((b & 0x7f) != 0)
    return (a == b) | (nan_a & nan_b)


def compare(root):
    report = {}
    for op in ["bias", "activation", "reduce"]:
        expected, observed, bad = [bits(root, f"{op}-{kind}.bin") for kind in ["expected", "observed", "known-bad"]]
        mismatch = (~same(expected, observed)).nonzero().flatten().tolist()
        detections = int((~same(expected, bad)).sum())
        report[op] = dict(count=len(expected), mismatch_count=len(mismatch), first_mismatches=[dict(index=i, expected=int(expected[i]), observed=int(observed[i])) for i in mismatch[:20]], known_bad_detections=detections)
    report["activation"]["symmetric_clamp_detections"] = int((~same(bits(root, "activation-expected.bin"), bits(root, "activation-symmetric-bad.bin"))).sum())
    (root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    assert all(v["mismatch_count"] == 0 and v["known_bad_detections"] > 0 for v in report.values())
    assert report["activation"]["symmetric_clamp_detections"] > 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["generate", "compare"])
    p.add_argument("directory", type=pathlib.Path)
    args = p.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.action == "generate":
        generate(args.directory)
    else:
        compare(args.directory)
