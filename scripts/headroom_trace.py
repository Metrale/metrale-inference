#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Export an nsys SQLite trace as a `met circuit headroom --trace` capture (TOML on stdout).

    headroom_trace.py <trace.sqlite> --step-kernel <name> --source '<fingerprint>'
        [--attribution <map.toml>]                    launch shape -> plan site (scripts/headroom/)
        [--tokens-per-step X --tokens-basis '<text>'] tokens a step emits in the capture
        [--kv-tokens N --kv-basis '<text>']           KV length the window's decode attended
        [--device N]                                  the device to read (default: the only one)
        --execution '<mode n=rows>' legacy|plan '<why>'   (repeat per run) who issued the run's
                                                      launches in the captured serve
        [--executed-sites '<mode n=rows>' '<site>;...']  under `plan`, the sites it issued (`*`: all)

The window is whole steps: from the first launch of the step kernel to its last launch, so
`steps` = launches - 1. Every kernel that starts in the window counts. Per kernel:
  total_us     = sum of (end - start);
  exclusive_us = the part of each launch after every earlier-launched kernel ended, clipped to
                 the window, so the exclusive times of all kernels sum to the GPU busy time.
Per step (anchor to anchor) the trace also carries `step_us` and, per kernel, `step_total_us`
(durations of the launches that start in the step) and `step_exclusive_us` (exclusive time
split across the steps it covers): the series the headroom bootstraps its intervals from.
The launch floor is the median duration of the shortest kernel that ran at least once per
step and never overlapped an earlier one (its whole time exclusive): what one launch costs
here when it does almost nothing.

The attribution map (TOML) has `[plan]` (dims of the rank the trace profiled, `dims_basis`) and
`[[launch]]` entries: kernel, grid = [x, y, z], optional smem, optional pred (the same-stream
predecessor kernel), site (`block.local` of the plan), label, optional note. The first entry
that matches a launch names its site; a launch no entry matches is exported with no site.

Stdlib only (Python 3.11+ for tomllib). Self-test: scripts/headroom_trace_test.py.
"""
from __future__ import annotations

import argparse
import bisect
import sqlite3
import statistics
import sys
import tomllib
from pathlib import Path


def load_calls(path: Path, device: int | None):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    q = """select k.start, k.end, s.value, k.gridX, k.gridY, k.gridZ, k.dynamicSharedMemory,
                  k.streamId, k.deviceId
           from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.shortName
           order by k.start, k.end"""
    rows = [r for r in db.execute(q) if r[0] > 0 and 0 < r[1] - r[0] < 1_000_000_000]
    devices = sorted({r[8] for r in rows})
    if device is None:
        if len(devices) != 1:
            raise SystemExit(f"{path}: kernels on devices {devices}; pass --device")
        device = devices[0]
    return [r for r in rows if r[8] == device]


def match(rules, name, grid, smem, pred):
    for r in rules:
        if r["kernel"] != name or tuple(r["grid"]) != grid:
            continue
        if "smem" in r and r["smem"] != smem:
            continue
        if "pred" in r and r["pred"] != pred:
            continue
        return r
    return None


def window(calls, step_kernel):
    anchors = [c[0] for c in calls if c[2] == step_kernel]
    if len(anchors) < 2:
        raise SystemExit(f"step kernel `{step_kernel}` launched {len(anchors)} time(s); a window needs 2")
    return anchors


def split(a, b, anchors, out):
    """Add the interval [a, b) to `out[step]` for each step it covers (anchors delimit steps)."""
    i = max(0, bisect.bisect_right(anchors, a) - 1)
    while a < b and i < len(anchors) - 1:
        hi = min(b, anchors[i + 1])
        if hi > a:
            out[i] += hi - a
        a, i = hi, i + 1


def aggregate(calls, rules, anchors):
    """Per (kernel, site, label): calls, total ns, exclusive ns, raw durations, solo flags, and the
    per-step series (a launch's duration counts in the step it starts in; its exclusive part is
    split across the steps it covers, so each step's exclusive sum never exceeds the step)."""
    w0, w1 = anchors[0], anchors[-1]
    steps = len(anchors) - 1
    groups: dict = {}
    last_on_stream: dict = {}
    horizon = max((c[1] for c in calls if c[0] < w0), default=w0)
    for st, en, name, gx, gy, gz, smem, stream, _ in calls:
        pred = last_on_stream.get(stream)
        last_on_stream[stream] = name
        if st < w0 or st >= w1:
            continue
        rule = match(rules, name, (gx, gy, gz), smem, pred)
        key = (name, rule["site"], rule["label"], rule.get("note")) if rule else \
            (name, None, f"grid ({gx}, {gy}, {gz}) smem {smem}", None)
        lo = max(st, horizon, w0)
        excl = max(0, min(en, w1) - lo)
        g = groups.setdefault(key, {"calls": 0, "total": 0, "excl": 0, "raw": [], "solo": True,
                                    "s_total": [0] * steps, "s_excl": [0] * steps})
        g["calls"] += 1
        g["total"] += en - st
        g["excl"] += excl
        g["raw"].append(en - st)
        g["solo"] = g["solo"] and excl == en - st
        g["s_total"][bisect.bisect_right(anchors, st) - 1] += en - st
        split(lo, min(en, w1), anchors, g["s_excl"])
        horizon = max(horizon, en)
    return groups


def launch_floor(groups, steps):
    """(us, kernel) of the shortest never-overlapped kernel launched at least once per step."""
    best = None
    for (name, _site, label, _), g in groups.items():
        if g["solo"] and g["calls"] >= steps:
            med = statistics.median(g["raw"]) / 1e3
            if best is None or med < best[0]:
                best = (med, f"{name} ({label})")
    return best


def q(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def series(ns) -> str:
    return "[" + ", ".join(f"{x / 1e3:.3f}" for x in ns) + "]"


def render(a, groups, anchors, plan) -> str:
    w0, w1, steps = anchors[0], anchors[-1], len(anchors) - 1
    out = ["# Generated by scripts/headroom_trace.py; do not edit.",
           "schema = 1",
           f"source = {q(a.source)}",
           f"step_kernel = {q(a.step_kernel)}",
           f"steps = {steps}",
           f"window_us = {(w1 - w0) / 1e3:.3f}",
           f"step_us = {series(b - a for a, b in zip(anchors, anchors[1:]))}"]
    if a.tokens_per_step is not None:
        out.append(f"tokens_per_step = {{ value = {a.tokens_per_step}, basis = {q(a.tokens_basis)} }}")
    lf = launch_floor(groups, steps)
    if lf is not None:
        out.append(f"launch_floor_us = {{ value = {lf[0]:.3f}, basis = {q(f'median duration of the shortest kernel launched every step and never overlapped: {lf[1]}, this capture')} }}")
    out.append("")
    out.append("[plan]")
    dims = plan.get("dims", {})
    if dims:
        out.append("dims = { " + ", ".join(f"{k} = {v}" for k, v in sorted(dims.items())) + " }")
        out.append(f"dims_basis = {q(plan['dims_basis'])}")
    if a.kv_tokens is not None:
        out.append(f"kv_tokens = {a.kv_tokens}")
        out.append(f"kv_basis = {q(a.kv_basis)}")
    out.append("")
    out.append("[plan.execution]")
    sites = dict(a.executed_sites or [])
    for run, by, why in a.execution:
        parts = [f"by = {q(by)}"]
        if run in sites:
            parts.append("sites = [" + ", ".join(q(x) for x in sites[run].split(";")) + "]")
        parts.append(f"why = {q(why)}")
        out.append(f"{q(run)} = {{ {', '.join(parts)} }}")
    for (name, site, label, note), g in sorted(groups.items(), key=lambda kv: (-kv[1]["total"], kv[0][0], str(kv[0][2]))):
        out.append("")
        out.append("[[kernel]]")
        out.append(f"kernel = {q(name)}")
        if site is not None:
            out.append(f"site = {q(site)}")
        out.append(f"label = {q(label)}")
        if note is not None:
            out.append(f"note = {q(note)}")
        out.append(f"calls = {g['calls']}")
        out.append(f"total_us = {g['total'] / 1e3:.3f}")
        out.append(f"exclusive_us = {g['excl'] / 1e3:.3f}")
        out.append(f"step_total_us = {series(g['s_total'])}")
        out.append(f"step_exclusive_us = {series(g['s_excl'])}")
    return "\n".join(out) + "\n"


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite", type=Path)
    ap.add_argument("--step-kernel", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--attribution", type=Path)
    ap.add_argument("--tokens-per-step", type=float)
    ap.add_argument("--tokens-basis")
    ap.add_argument("--kv-tokens", type=int)
    ap.add_argument("--kv-basis")
    ap.add_argument("--device", type=int)
    # 2026-10-23: How the captured serve issued each run: required, a fact about the serve.
    ap.add_argument("--execution", nargs=3, action="append", required=True,
                    metavar=("RUN", "BY", "WHY"))
    ap.add_argument("--executed-sites", nargs=2, action="append", metavar=("RUN", "SITES"))
    a = ap.parse_args(argv)
    runs = [r for r, _, _ in a.execution]
    if len(set(runs)) != len(runs):
        ap.error(f"--execution states a run twice: {runs}")
    for run, by, why in a.execution:
        if by not in ("legacy", "plan") or not why.strip():
            ap.error(f"--execution {run!r}: BY is legacy or plan and WHY is stated")
    for run, _ in a.executed_sites or []:
        if (run, "plan") not in [(r, b) for r, b, _ in a.execution]:
            ap.error(f"--executed-sites {run!r}: only a run executed `plan` names sites")
    for run, by, _ in a.execution:
        if by == "plan" and run not in dict(a.executed_sites or []):
            ap.error(f"--execution {run!r} plan: state --executed-sites (`*` for all)")
    for v, b, what in ((a.tokens_per_step, a.tokens_basis, "--tokens-per-step"),
                       (a.kv_tokens, a.kv_basis, "--kv-tokens")):
        if (v is None) != (b is None):
            ap.error(f"{what} and its --*-basis go together")
    amap = tomllib.loads(a.attribution.read_text()) if a.attribution else {}
    rules = amap.get("launch", [])
    calls = load_calls(a.sqlite, a.device)
    anchors = window(calls, a.step_kernel)
    groups = aggregate(calls, rules, anchors)
    sys.stdout.write(render(a, groups, anchors, amap.get("plan", {})))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
