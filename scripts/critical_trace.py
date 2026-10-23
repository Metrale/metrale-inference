#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Export an nsys SQLite trace as a `met circuit whatif` event capture (JSON on stdout).

    critical_trace.py <trace.sqlite> --step-kernel <name> --source '<fingerprint>'
        [--attribution <map.toml>]   launch shape -> plan site (scripts/headroom/), as headroom_trace.py
        [--device N]                 the device to read (default: the only one)
        [--context-us U]             ops before the first anchor kept as fixed context (default 5000)

Unlike headroom_trace.py (per-kernel sums), this keeps every operation: each kernel, memcpy and
memset with its stream, start, end, launch correlation id, whether it is a CUDA-graph node and
its launch configuration (CTAs, threads, registers, shared memory); every runtime/driver API
call of the thread that launched the window's work (start, end, name, correlation id); every
event record (correlation id, stream, event) and every synchronization (stream wait / stream
sync / event sync, with the waiting stream and event). Nothing here infers, models or sums:
the dependency graph, the latencies and the critical path are metrale-circuit's
(`hardware::critical`). Times are integer nanoseconds relative to the first anchor.

Records are arrays (an object per record would triple the file): see `FIELDS` in the output.
Stdlib only (Python 3.11+). Self-test: scripts/critical_trace_test.py.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from headroom_trace import match  # noqa: E402  (one attribution rule, SSOT)

FIELDS = {
    "op": ["kind", "name", "stream", "start", "end", "corr", "graph", "ctas", "threads", "regs",
           "smem", "bytes", "group"],
    "api": ["name", "start", "end", "corr"],
    "record": ["corr", "stream", "event"],
    "sync": ["corr", "kind", "stream", "event", "start", "end"],
    "group": ["kernel", "site", "label", "note"],
}

SYNC_KIND = {1: "event_sync", 2: "stream_wait", 3: "stream_sync", 4: "context_sync"}


def rows(db, q, *args):
    return list(db.execute(q, args))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite", type=Path)
    ap.add_argument("--step-kernel", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--attribution", type=Path)
    ap.add_argument("--device", type=int)
    ap.add_argument("--context-us", type=float, default=5000.0)
    a = ap.parse_args(argv)
    rules = tomllib.loads(a.attribution.read_text()).get("launch", []) if a.attribution else []
    db = sqlite3.connect(f"file:{a.sqlite}?mode=ro", uri=True)
    ks = rows(db, """select k.start, k.end, s.value, k.streamId, k.correlationId,
                            k.graphNodeId is not null, k.gridX*k.gridY*k.gridZ,
                            k.blockX*k.blockY*k.blockZ, k.registersPerThread,
                            k.staticSharedMemory + k.dynamicSharedMemory, k.deviceId,
                            k.gridX, k.gridY, k.gridZ, k.dynamicSharedMemory
                     from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.shortName
                     order by k.start, k.end""")
    devices = sorted({k[10] for k in ks})
    device = a.device if a.device is not None else (devices[0] if len(devices) == 1 else None)
    if device is None:
        raise SystemExit(f"{a.sqlite}: kernels on devices {devices}; pass --device")
    ks = [k for k in ks if k[10] == device and k[0] > 0 and 0 < k[1] - k[0] < 1_000_000_000]
    anchors = [k[0] for k in ks if k[2] == a.step_kernel]
    if len(anchors) < 2:
        raise SystemExit(f"step kernel `{a.step_kernel}` launched {len(anchors)} time(s); a window needs 2")
    t0, t1 = anchors[0], anchors[-1]
    lo = t0 - int(a.context_us * 1e3)
    groups: dict = {}
    ops = []
    last_on_stream: dict = {}
    for st, en, name, stream, corr, graph, ctas, thr, regs, smem, _, gx, gy, gz, dsm in ks:
        pred = last_on_stream.get(stream)
        last_on_stream[stream] = name
        if st < lo or st > t1:
            continue
        r = match(rules, name, (gx, gy, gz), dsm, pred)
        key = (name, r["site"], r["label"], r.get("note")) if r else (name, None, None, None)
        g = groups.setdefault(key, len(groups))
        ops.append(["k", name, stream, st - t0, en - t0, corr, bool(graph), ctas, thr, regs, smem, 0, g])
    for table, kind, extra in (("CUPTI_ACTIVITY_KIND_MEMCPY", "m", "bytes"),
                               ("CUPTI_ACTIVITY_KIND_MEMSET", "s", "bytes")):
        for st, en, stream, corr, graph, nbytes in rows(
                db, f"""select start, end, streamId, correlationId, graphNodeId is not null, {extra}
                        from {table} where deviceId = ? and start >= ? and start <= ?""", device, lo, t1):
            name = "memcpy" if kind == "m" else "memset"
            g = groups.setdefault((name, None, None, None), len(groups))
            ops.append([kind, name, stream, st - t0, en - t0, corr, bool(graph), 0, 0, 0, 0, nbytes, g])
    ops.sort(key=lambda o: (o[3], o[4], o[5]))
    corrs = {o[5] for o in ops}
    # The launching thread: the one whose API calls own the window's launches.
    tids = rows(db, """select globalTid, count(*) from CUPTI_ACTIVITY_KIND_RUNTIME
                       where start >= ? and start < ? group by globalTid order by 2 desc""", lo, t1)
    owner = None
    for tid, _ in tids:
        n = rows(db, """select count(*) from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid = ?
                        and start >= ? and start < ? and correlationId in (select correlationId
                        from CUPTI_ACTIVITY_KIND_KERNEL where start >= ? and start < ?)""", tid, lo, t1, lo, t1)[0][0]
        if n:
            owner = tid
            break
    if owner is None:
        raise SystemExit("no API thread launched the window's kernels")
    api = [[name, st - t0, en - t0, corr] for st, en, name, corr in rows(
        db, """select r.start, r.end, s.value, r.correlationId from CUPTI_ACTIVITY_KIND_RUNTIME r
               join StringIds s on s.id = r.nameId where r.globalTid = ? and r.start >= ? and r.start < ?
               order by r.start""", owner, lo, t1)]
    api_corrs = {x[3] for x in api}
    record = [[corr, stream, ev] for corr, stream, ev in rows(
        db, """select correlationId, streamId, eventId from CUPTI_ACTIVITY_KIND_CUDA_EVENT
               where deviceId = ?""", device) if corr in api_corrs or corr in corrs]
    sync = [[corr, SYNC_KIND.get(t, "unknown"), stream, (None if ev == 4294967295 else ev), st - t0, en - t0]
            for st, en, corr, stream, t, ev in rows(
                db, """select start, end, correlationId, streamId, syncType, eventId
                       from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where start >= ? and start < ?""", lo, t1)
            if corr in api_corrs]
    out = {
        "schema": "critical-trace/v1",
        "source": a.source,
        "step_kernel": a.step_kernel,
        "anchors": [x - t0 for x in anchors],
        "fields": FIELDS,
        "groups": [list(k) for k, _ in sorted(groups.items(), key=lambda kv: kv[1])],
        "ops": ops,
        "api": api,
        "record": record,
        "sync": sync,
    }
    json.dump(out, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
