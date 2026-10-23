#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Self-test of scripts/critical_trace.py on a hand-built nsys-shaped SQLite file.

    python3 scripts/critical_trace_test.py

Each expected record is written out by hand from the rows below (times in ns).
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
NONE_EVENT = 4294967295

# (start, end, name, stream, corr, graph node, grid, block, regs, smem static, smem dyn, device)
KERNELS = [
    (-9_000_000, -8_999_000, "ancient", 7, 1, None, 1, 32, 16, 0, 0, 0),  # before 5 ms of context
    (-1_000, -500, "warm", 7, 2, None, 1, 32, 16, 0, 0, 0),               # context: kept
    (0, 100, "step", 7, 10, 5, 1, 32, 16, 0, 0, 0),                       # anchor 1 (t0)
    (150, 900, "gemv", 9, 11, 6, 8, 256, 64, 1024, 512, 0),
    (500, 600, "other", 7, 12, None, 1, 32, 16, 0, 0, 1),                 # device 1: dropped
    (1_000, 1_100, "step", 7, 20, 5, 1, 32, 16, 0, 0, 0),                 # anchor 2 (t1)
    (1_200, 1_300, "late", 7, 21, None, 1, 32, 16, 0, 0, 0),              # after t1: dropped
]
# (start, end, name, corr, thread)
API = [
    (-2_000, -1_990, "cuLaunchKernel", 2, 100),
    (-50, -40, "cuGraphLaunch", 10, 100),
    (-30, -20, "cuEventRecord", 30, 100),
    (-20, -10, "cuStreamWaitEvent", 31, 100),
    (950, 960, "cuGraphLaunch", 20, 100),
    (960, 990, "cuStreamSynchronize", 32, 100),
] + [(10 * i, 10 * i + 5, "cuMemAlloc", 40 + i, 200) for i in range(1, 9)]  # busier, launches nothing


def build(db: Path) -> None:
    c = sqlite3.connect(db)
    c.execute("create table StringIds (id integer primary key, value text)")
    c.execute("""create table CUPTI_ACTIVITY_KIND_KERNEL (start int, end int, shortName int,
                 streamId int, correlationId int, graphNodeId int, gridX int, gridY int, gridZ int,
                 blockX int, blockY int, blockZ int, registersPerThread int,
                 staticSharedMemory int, dynamicSharedMemory int, deviceId int)""")
    for t in ("MEMCPY", "MEMSET"):
        c.execute(f"""create table CUPTI_ACTIVITY_KIND_{t} (start int, end int, streamId int,
                      correlationId int, graphNodeId int, bytes int, deviceId int)""")
    c.execute("""create table CUPTI_ACTIVITY_KIND_RUNTIME (start int, end int, nameId int,
                 correlationId int, globalTid int)""")
    c.execute("create table CUPTI_ACTIVITY_KIND_CUDA_EVENT (correlationId int, streamId int, eventId int, deviceId int)")
    c.execute("""create table CUPTI_ACTIVITY_KIND_SYNCHRONIZATION (start int, end int,
                 correlationId int, streamId int, syncType int, eventId int)""")
    names = sorted({k[2] for k in KERNELS} | {a[2] for a in API})
    for i, n in enumerate(names):
        c.execute("insert into StringIds values (?, ?)", (i, n))
    # Offset every time: nsys starts are positive, the exporter drops start <= 0.
    o = 10_000_000
    for st, en, n, s, corr, g, grid, blk, regs, ss, ds, dev in KERNELS:
        c.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?,?,1,1,?,1,1,?,?,?,?)",
                  (st + o, en + o, names.index(n), s, corr, g, grid, blk, regs, ss, ds, dev))
    c.execute("insert into CUPTI_ACTIVITY_KIND_MEMCPY values (?,?,?,?,?,?,?)", (950 + o, 980 + o, 9, 33, None, 64, 0))
    c.execute("insert into CUPTI_ACTIVITY_KIND_MEMSET values (?,?,?,?,?,?,?)", (5_000 + o, 5_100 + o, 9, 34, None, 8, 0))
    for st, en, n, corr, tid in API:
        c.execute("insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?,?,?,?,?)", (st + o, en + o, names.index(n), corr, tid))
    c.execute("insert into CUPTI_ACTIVITY_KIND_CUDA_EVENT values (30, 7, 8, 0)")
    c.execute("insert into CUPTI_ACTIVITY_KIND_CUDA_EVENT values (99, 7, 9, 0)")  # nobody's call
    c.execute("insert into CUPTI_ACTIVITY_KIND_SYNCHRONIZATION values (?,?,31,9,2,8)", (-20 + o, -10 + o))
    c.execute("insert into CUPTI_ACTIVITY_KIND_SYNCHRONIZATION values (?,?,32,7,3,?)", (960 + o, 990 + o, NONE_EVENT))
    c.execute("insert into CUPTI_ACTIVITY_KIND_SYNCHRONIZATION values (?,?,41,7,3,?)", (10 + o, 15 + o, NONE_EVENT))
    c.commit()


def run(db: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(HERE / "critical_trace.py"), str(db), "--step-kernel", "step",
                           "--source", "test", *extra], capture_output=True, text=True)


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "t.sqlite"
        build(db)
        r = run(db)
        assert r.returncode != 0 and "--device" in r.stderr, (r.returncode, r.stderr)
        r = run(db, "--device", "0")
        assert r.returncode == 0, r.stderr
        t = json.loads(r.stdout)
        bad = run(db, "--device", "0", "--step-kernel", "gemv")
        assert bad.returncode != 0 and "a window needs 2" in bad.stderr, bad.stderr
    assert t["schema"] == "critical-trace/v1" and t["anchors"] == [0, 1000], t["anchors"]
    g = t["groups"]
    ops = [[o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8], o[9], o[10], o[11]] for o in t["ops"]]
    assert ops == [
        ["k", "warm", 7, -1000, -500, 2, False, 1, 32, 16, 0, 0],
        ["k", "step", 7, 0, 100, 10, True, 1, 32, 16, 0, 0],
        ["k", "gemv", 9, 150, 900, 11, True, 8, 256, 64, 1536, 0],
        ["m", "memcpy", 9, 950, 980, 33, False, 0, 0, 0, 0, 64],
        ["k", "step", 7, 1000, 1100, 20, True, 1, 32, 16, 0, 0],
    ], ops
    assert all(g[o[12]][0] == o[1] for o in t["ops"]), "each op's group names its kernel"
    assert [a[2] for a in t["api"]] != [] and {a[3] for a in t["api"]} == {2, 10, 30, 31, 20, 32}, t["api"]
    assert t["record"] == [[30, 7, 8]], t["record"]
    assert t["sync"] == [[31, "stream_wait", 9, 8, -20, -10], [32, "stream_sync", 7, None, 960, 990]], t["sync"]
    print("critical_trace_test: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
