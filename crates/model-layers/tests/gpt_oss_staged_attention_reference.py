#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Exact-BF16 gate, declared before execution; preserve every mismatch."""
import argparse
import ctypes
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import transformers
from safetensors import safe_open
from transformers.models.gpt_oss.modeling_gpt_oss import eager_attention_forward

# 2026-10-07: Repository root; the harness and kernel sources are hashed in place.
ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    a = parser.parse_args()
    a.output.mkdir(exist_ok=False)
    assert transformers.__version__ == "4.55.0", "reference version changed"
    reference_source = inspect.getsource(eager_attention_forward).encode()
    reference_sha = hashlib.sha256(reference_source).hexdigest()
    assert reference_sha == "390aa75a455a553c819f6c35a9682b62233f6440695e6bdaf288646110769722", "pinned reference implementation changed"
    torch.cuda.set_per_process_memory_fraction(.85)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(1441)
    lib = ctypes.CDLL(str(a.library.resolve()))
    lib.staged_run.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_uint] * 4 + [ctypes.c_float, ctypes.c_uint, ctypes.c_ulonglong]
    lib.staged_run.restype = ctypes.c_int
    report = {"gate": "exact BF16 bits, NaN payload excluded", "library_sha256": hashlib.sha256(a.library.read_bytes()).hexdigest(), "kernel_source_sha256": hashlib.sha256((ROOT / "kernels/gb10/common/gpt_oss_staged_attention.cu").read_bytes()).hexdigest(), "harness_source_sha256": hashlib.sha256((ROOT / "crates/model-layers/tests/cuda/gpt_oss_staged_attention_test.cu").read_bytes()).hexdigest(), "cases": []}

    report["provenance"] = {
        "transformers": transformers.__version__, "torch": torch.__version__,
        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "reference_function_sha256": reference_sha,
        "reference_module_sha256": hashlib.sha256(Path(inspect.getfile(eager_attention_forward)).read_bytes()).hexdigest(),
        "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tf32": False, "bf16_reduced_precision_reduction": False,
        "snapshot_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.snapshots.glob("*.bin"))},
    }

    def reference(q, k, v, sinks, window):
        length, kvh, _ = k.shape
        holder = SimpleNamespace(num_key_value_groups=q.shape[0] // kvh, sinks=sinks, training=False)
        mask = torch.zeros((1, 1, 1, length), device="cuda", dtype=torch.bfloat16)
        if window and length > window:
            mask[..., :length-window] = torch.finfo(torch.bfloat16).min
        result, _ = eager_attention_forward(holder, q[None, :, None, :], k.permute(1, 0, 2)[None], v.permute(1, 0, 2)[None], mask, .125)
        return result.reshape_as(q)

    def run(name, q, k, v, sinks, window=0, controls=False):
        length, kvh, _ = k.shape
        block = 16
        count = (length + block - 1) // block
        table = torch.arange(count-1, -1, -1, device="cuda", dtype=torch.int32)
        kp = torch.zeros((count, block, kvh, 64), device="cuda", dtype=torch.bfloat16)
        vp = torch.zeros_like(kp)
        for t in range(length):
            kp[count-1-t//block, t%block] = k[t]
            vp[count-1-t//block, t%block] = v[t]
        lengths = torch.tensor([length], dtype=torch.int32, device="cuda")
        output = torch.full_like(q, float("nan"))
        code = lib.staged_run(*(ctypes.c_void_p(x.data_ptr()) for x in [q, kp, vp, output, table, lengths, sinks]), count, q.shape[0], kvh, block, .125, window, torch.cuda.current_stream().cuda_stream)
        if code:
            raise RuntimeError(f"CUDA {code}")
        expected = reference(q, k, v, sinks, window)
        mismatch = (output.view(torch.int16) != expected.view(torch.int16)) & ~(torch.isnan(output) & torch.isnan(expected))
        entry = {"name": name, "shape": list(q.shape), "bit_mismatches": int(mismatch.sum()), "max_abs": float((output.float()-expected.float()).abs().nan_to_num().max())}
        for tag, x in [("q", q), ("k", k), ("v", v), ("sinks", sinks), ("native", output), ("reference", expected)]:
            (a.output / f"{name}-{tag}.bin").write_bytes(x.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        if controls:
            no_sink = reference(q, k, v, torch.full_like(sinks, -float("inf")), window)
            no_window = reference(q, k, v, sinks, 0)
            scores = (q.float()[:, None, :] * k.repeat_interleave(q.shape[0]//kvh, dim=1).permute(1, 0, 2).float()).sum(-1) * .125
            if window and length > window:
                scores[:, :length-window] = -float("inf")
            probs = torch.softmax(torch.cat([scores, sinks.float()[:, None]], dim=-1), dim=-1)[:, :-1]
            fp32 = (probs[:, :, None] * v.repeat_interleave(q.shape[0]//kvh, dim=1).permute(1, 0, 2).float()).sum(1).bfloat16()
            entry["known_bad"] = {"missing_sink": int((no_sink != expected).sum()), "ignored_window": int((no_window != expected).sum()), "fp32_policy": int((fp32 != expected).sum())}
            assert all(entry["known_bad"].values()), entry
        report["cases"].append(entry)

    # 2026-10-06: Nontrivial BF16 rounding, sink, window and reversed physical block mapping.
    q = torch.randn((8, 64), device="cuda").bfloat16()
    k = torch.randn((145, 2, 64), device="cuda").bfloat16()
    v = torch.randn_like(k)
    sinks = torch.linspace(-2, 3, 8, device="cuda").bfloat16()
    run("constructed-window", q, k, v, sinks, 128, True)
    run("constructed-no-sink", q, k, v, torch.full_like(sinks, -float("inf")))
    run("constructed-inf-sink", q, k, v, torch.full_like(sinks, float("inf")))
    run("constructed-nan-sink", q, k, v, torch.full_like(sinks, float("nan")))
    run("constructed-one-token", q, k[:1].contiguous(), v[:1].contiguous(), sinks)
    run("constructed-max-context", q[:2].contiguous(), torch.randn((4096, 1, 64), device="cuda").bfloat16(), torch.randn((4096, 1, 64), device="cuda").bfloat16(), sinks[:2].contiguous())
    # 2026-10-06: Empty and raw-device over-cap lengths must overwrite the output safely.
    for length in [0, 4097]:
        output = torch.ones_like(q)
        table = torch.zeros(1, dtype=torch.int32, device="cuda")
        lengths = torch.tensor([length], dtype=torch.int32, device="cuda")
        code = lib.staged_run(*(ctypes.c_void_p(x.data_ptr()) for x in [q, k, v, output, table, lengths, sinks]), 1, 8, 2, 16, .125, 0, torch.cuda.current_stream().cuda_stream)
        assert code == 0
        assert bool((output == 0).all()) if length == 0 else bool(torch.isnan(output).all())
        report["cases"].append({"name": f"length-{length}-fail-closed", "bit_mismatches": 0})
    index = json.loads((a.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    name = "model.layers.9.self_attn.sinks"
    with safe_open(a.checkpoint / index[name], framework="pt", device="cpu") as f:
        real_sinks = f.get_tensor(name).cuda()
    report["provenance"]["sinks_sha256"] = hashlib.sha256(real_sinks.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    for position in [49, 215]:
        def load(name, shape):
            path = a.snapshots / f"p{position}-l9-{name}.bin"
            return torch.from_numpy(np.fromfile(path, dtype="<i2").copy()).view(torch.bfloat16).reshape(shape).cuda()
        run(f"captured-{position}", load("q_post_rope", (64, 64)), load("cache_keys", (position+1, 8, 64)), load("cache_values", (position+1, 8, 64)), real_sinks)
    report["passed"] = all(x["bit_mismatches"] == 0 for x in report["cases"])
    (a.output / "comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
