#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Self-test of scripts/headroom_trace.py on a hand-built nsys-shaped SQLite file.

    python3 scripts/headroom_trace_test.py

Each expected number is computed by hand from the launches below (times in ns).
"""
from __future__ import annotations

import sqlite3
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (start, end, name, grid, smem, stream)
LAUNCHES = [
    (500, 900, "warm", (1, 1, 1), 0, 7),        # before the window, overlaps its start
    (1000, 1100, "step", (1, 1, 1), 0, 7),      # anchor 1: the window starts at 1000
    (1100, 1600, "gemv", (8, 1, 1), 0, 7),      # 500 ns, all exclusive
    (1300, 1800, "side", (2, 1, 1), 0, 9),      # 500 ns, 1600..1800 exclusive = 200
    (1900, 2000, "gemv", (4, 1, 1), 0, 7),      # another shape: attributed elsewhere
    (2000, 2100, "step", (1, 1, 1), 0, 7),      # anchor 2
    (2100, 2600, "gemv", (8, 1, 1), 0, 7),      # 500 exclusive
    (2900, 3100, "tail", (1, 1, 1), 0, 7),      # crosses the window end (3000): 100 clipped in
    (3000, 3100, "step", (1, 1, 1), 0, 7),      # anchor 3: the window ends at 3000
    (3200, 3300, "late", (1, 1, 1), 0, 7),      # after the window: not counted
]

MAP = """
[plan]
dims = { q_heads = 22 }
dims_basis = "test shard"

[[launch]]
kernel = "gemv"
grid = [8, 1, 1]
site = "blk.big"
label = "big"

[[launch]]
kernel = "gemv"
grid = [4, 1, 1]
pred = "side"
site = "blk.never"
label = "only after side on the same stream (never: side is on stream 9)"

[[launch]]
kernel = "gemv"
grid = [4, 1, 1]
site = "blk.small"
label = "small"
note = "folded"
"""


def build(db: Path) -> None:
    c = sqlite3.connect(db)
    c.execute("create table StringIds (id integer primary key, value text)")
    c.execute("""create table CUPTI_ACTIVITY_KIND_KERNEL (start int, end int, shortName int,
                 gridX int, gridY int, gridZ int, dynamicSharedMemory int, streamId int, deviceId int)""")
    names = sorted({x[2] for x in LAUNCHES})
    for i, n in enumerate(names):
        c.execute("insert into StringIds values (?, ?)", (i, n))
    for st, en, n, g, smem, stream in LAUNCHES:
        c.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?,?,?,0)",
                  (st, en, names.index(n), *g, smem, stream))
    c.commit()


def export(db: Path, amap: Path) -> dict:
    out = subprocess.run(
        [sys.executable, str(HERE / "headroom_trace.py"), str(db), "--step-kernel", "step",
         "--source", "test", "--attribution", str(amap), "--kv-tokens", "100", "--kv-basis", "t"],
        capture_output=True, text=True, check=True)
    return tomllib.loads(out.stdout)


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        db, amap = Path(d) / "t.sqlite", Path(d) / "map.toml"
        build(db)
        amap.write_text(MAP)
        t = export(db, amap)
    assert t["steps"] == 2 and t["window_us"] == 2.0, (t["steps"], t["window_us"])
    rows = {(k["kernel"], k.get("site")): k for k in t["kernel"]}
    big = rows[("gemv", "blk.big")]
    assert (big["calls"], big["total_us"], big["exclusive_us"]) == (2, 1.0, 1.0), big
    small = rows[("gemv", "blk.small")]
    assert small["note"] == "folded" and small["exclusive_us"] == 0.1, small
    assert ("gemv", "blk.never") not in rows, "a pred rule matched a launch on another stream"
    side = rows[("side", None)]
    assert (side["total_us"], side["exclusive_us"]) == (0.5, 0.2), side
    tail = rows[("tail", None)]
    assert (tail["total_us"], tail["exclusive_us"]) == (0.2, 0.1), tail
    assert ("late", None) not in rows and ("warm", None) not in rows
    # Busy = step 1 0.1 + gemv 0.5 + side 0.2 + small gemv 0.1 + step 2 0.1 + gemv 0.5 + tail 0.1.
    busy = sum(k["exclusive_us"] for k in t["kernel"])
    assert abs(busy - 1.6) < 1e-9 and busy <= t["window_us"], busy
    # The shortest never-overlapped kernel launched every step: `step` (100 ns).
    assert t["launch_floor_us"]["value"] == 0.1, t["launch_floor_us"]
    assert t["plan"]["dims"] == {"q_heads": 22} and t["plan"]["kv_tokens"] == 100
    # Per-step series: two 1 us steps; each kernel's series sums to its totals, and no step's
    # exclusive time exceeds the step.
    assert t["step_us"] == [1.0, 1.0], t["step_us"]
    assert big["step_total_us"] == [0.5, 0.5] and big["step_exclusive_us"] == [0.5, 0.5], big
    assert tail["step_exclusive_us"] == [0.0, 0.1], tail
    for k in t["kernel"]:
        assert abs(sum(k["step_total_us"]) - k["total_us"]) < 1e-9, k
        assert abs(sum(k["step_exclusive_us"]) - k["exclusive_us"]) < 1e-9, k
    for i, step in enumerate(t["step_us"]):
        assert sum(k["step_exclusive_us"][i] for k in t["kernel"]) <= step + 1e-9, i
    print("headroom_trace_test: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
