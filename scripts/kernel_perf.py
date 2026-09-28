#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate the kernel performance map, KERNEL-PERF.md, from the tree (SSOT).

    python3 scripts/kernel_perf.py            # rewrite the generated block of KERNEL-PERF.md,
                                              # docs/kernel-perf/TRADEOFFS.md and MEASUREMENTS.md
    python3 scripts/kernel_perf.py --check    # exit 1 if the committed block is stale or a
                                              # curated input no longer matches the tree
    python3 scripts/kernel_perf.py --json     # the joined inventory as JSON, for tooling

Inputs: kernels/** and crates/** (derived: entry points, targets, call sites), and three
curated files under docs/kernel-perf/: taxonomy.toml (families, components, rules),
tradeoffs.toml (known trade-offs and PRs per kernel), measurements.toml (per-kernel % of
the roofline floor, written by the measurement procedure in KERNEL-PERF.md). Only the
text between the BEGIN/END markers of KERNEL-PERF.md is generated; the prose around it
(intro, methodology) is hand-written.

Stdlib only (Python 3.11+ for tomllib). The self-test is scripts/kernel_perf_test.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
import kernel_perf_inventory as inventory  # noqa: E402
import kernel_perf_model as model  # noqa: E402
import kernel_perf_render as render  # noqa: E402

BEGIN = "<!-- kernel_perf.py: BEGIN GENERATED (edit the inputs, then run scripts/kernel_perf.py) -->"
END = "<!-- kernel_perf.py: END GENERATED -->"
DATA = Path("docs/kernel-perf")


def _load(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def build(root: Path) -> model.Model:
    inv = inventory.build(root)
    crates = {
        os.path.relpath(os.path.join(dp, f), root)
        for dp, _dn, fn in os.walk(root / "crates")
        for f in fn
    }
    return model.build(
        inv,
        _load(root / DATA / "taxonomy.toml"),
        _load(root / DATA / "tradeoffs.toml"),
        _load(root / DATA / "measurements.toml"),
        crates,
    )


def splice(doc: str, block: str) -> str:
    if doc.count(BEGIN) != 1 or doc.count(END) != 1 or doc.index(BEGIN) > doc.index(END):
        raise SystemExit("KERNEL-PERF.md must hold exactly one BEGIN marker followed by one END marker")
    head, rest = doc.split(BEGIN, 1)
    _old, tail = rest.split(END, 1)
    return f"{head}{BEGIN}\n\n{block}\n{END}{tail}"


def as_json(m: model.Model) -> list[dict]:
    return [
        {
            "kernel": f"{sorted(r.k.modules)[0]}::{r.k.name}",
            "file": r.k.file,
            "line": r.k.line,
            "modules": sorted(r.k.modules),
            "compiled_from": sorted(r.k.sources),
            "hardware": r.hardware,
            "targets": sorted("/".join(t) for t in r.k.targets),
            "component": r.component,
            "kind": r.kind,
            "used_by": sorted(r.used_by),
            "families": sorted(r.families),
            "engine_call_sites": [f"{p}:{n}" for p, n in r.engine_sites],
            "measurements": r.measures,
        }
        for r in m.rows
    ]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail if KERNEL-PERF.md is stale")
    mode.add_argument("--json", action="store_true", help="print the joined inventory as JSON")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    try:
        m = build(root)
    except model.ModelError as e:
        print(f"kernel_perf: {len(e.problems)} inconsistencies between the curated inputs and the tree:", file=sys.stderr)
        for p in e.problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    if args.json:
        json.dump(as_json(m), sys.stdout, indent=1)
        print()
        return 0
    r = render.Renderer(m)
    doc_path = root / "KERNEL-PERF.md"
    doc = doc_path.read_text()
    want = {
        doc_path: splice(doc, r.render()),
        root / render.REGISTER: r.render_register(),
        root / render.MEASUREMENTS: r.render_measurements(),
    }
    have = {p: (p.read_text() if p.is_file() else None) for p in want}
    stale = [p for p in want if want[p] != have[p]]
    if args.check:
        for p in stale:
            print(f"kernel_perf: {p.relative_to(root)} is stale; run `python3 scripts/kernel_perf.py` and commit it",
                  file=sys.stderr)
        if stale:
            return 1
        print(f"kernel_perf: KERNEL-PERF.md, {render.REGISTER} and {render.MEASUREMENTS} are current "
              f"({len(m.rows)} entry points)")
        return 0
    for p in stale:
        p.write_text(want[p])
        print(f"kernel_perf: wrote {p.relative_to(root)}")
    if not stale:
        print("kernel_perf: already current")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
