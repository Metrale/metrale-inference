// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Hand-built captures for the critical-path tests: every time is chosen so the
//! expected step, chain and slack can be computed by hand (stated beside each fixture).

use crate::prelude::*;

use serde_json::{Value, json};

use super::capture::{Capture, parse_capture};

/// 2026-10-23: `(kind, name, stream, start_ns, end_ns, corr, graph, ctas, threads, regs, smem)`.
pub type OpRow = (
    &'static str,
    &'static str,
    u64,
    i64,
    i64,
    u64,
    bool,
    u32,
    u32,
    u32,
    u32,
);

/// 2026-10-23: A capture of `ops`, `api` (`name, start, end, corr`), `record` (`corr, stream,
/// event`) and `sync` (`corr, kind, stream, event, start, end`), anchored on kernel `step`.
pub fn capture(
    mut ops: Vec<OpRow>,
    api: Vec<(&str, i64, i64, u64)>,
    record: Vec<(u64, u64, u64)>,
    sync: Vec<(u64, &str, u64, Option<u64>, i64, i64)>,
) -> Capture {
    ops.sort_by_key(|o| (o.3, o.4, o.5));
    let mut names: Vec<&str> = ops.iter().map(|o| o.1).collect();
    names.sort_unstable();
    names.dedup();
    let groups: Vec<Value> = names
        .iter()
        .map(|n| {
            let site = format!("blk.{n}");
            json!([n, site, Value::Null, Value::Null])
        })
        .collect();
    let rows: Vec<Value> = ops
        .iter()
        .map(|o| {
            let g = names.iter().position(|n| *n == o.1).unwrap_or(0);
            json!([o.0, o.1, o.2, o.3, o.4, o.5, o.6, o.7, o.8, o.9, o.10, 0, g])
        })
        .collect();
    let anchors: Vec<i64> = ops
        .iter()
        .filter(|o| o.1 == "step" && o.3 >= 0)
        .map(|o| o.3)
        .collect();
    let fields: Value = super::capture::FIELDS
        .iter()
        .map(|(k, v)| ((*k).to_string(), json!(v)))
        .collect::<serde_json::Map<String, Value>>()
        .into();
    let v = json!({
        "schema": "critical-trace/v1",
        "source": "fixture",
        "step_kernel": "step",
        "anchors": anchors,
        "fields": fields,
        "groups": groups,
        "ops": rows,
        "api": api.iter().map(|a| json!([a.0, a.1, a.2, a.3])).collect::<Vec<_>>(),
        "record": record.iter().map(|r| json!([r.0, r.1, r.2])).collect::<Vec<_>>(),
        "sync": sync.iter().map(|s| json!([s.0, s.1, s.2, s.3, s.4, s.5])).collect::<Vec<_>>(),
    });
    parse_capture(&serde_json::to_string(&v).unwrap_or_default())
        .unwrap_or_else(|e| panic!("fixture: {e}"))
}

/// 2026-10-23: `steps` 23 µs graph steps (one launch each, corr 101 + k; streams 10k+1..10k+3):
///
/// ```text
/// s1: step [0,1] (0.5) a1 [1.5,11.5]  a2 [11,14.5] (PDL on a1: work 3)
/// s2:        (1) b1 [2,20]                        fork: first on s2, after `step`
/// s3:                                (1) a3 [21,22]  join: first on s3, after b1 (latest end)
/// next step [23,24] on the next launch's s1: first there, after a3
/// ```
///
/// Joins 1 µs and the one graph-stream edge 0.5 µs, so the model latencies are exact. Critical
/// chain per step: step 1 -> 1 -> b1 18 -> 1 -> a3 1 -> 1 = 23 µs; a1 and a2 feed nothing.
/// `a2_at` moves a2's start (11.0 overlaps a1: PDL; 13.5: a 2 µs mid-stream gap).
pub fn diamond(steps: i64, a2_at: i64) -> Capture {
    let mut ops: Vec<OpRow> = Vec::new();
    for k in -1..steps {
        let b = k * 23_000;
        let c = u64::try_from(101 + k).unwrap();
        let s = 10 * u64::try_from(k + 1).unwrap();
        ops.push(("k", "step", s + 1, b, b + 1_000, c, true, 1, 32, 16, 0));
        ops.push((
            "k",
            "a1",
            s + 1,
            b + 1_500,
            b + 11_500,
            c,
            true,
            192,
            256,
            64,
            0,
        ));
        ops.push((
            "k",
            "a2",
            s + 1,
            b + a2_at,
            b + 14_500,
            c,
            true,
            4,
            256,
            32,
            0,
        ));
        ops.push((
            "k",
            "b1",
            s + 2,
            b + 2_000,
            b + 20_000,
            c,
            true,
            4,
            256,
            32,
            0,
        ));
        ops.push((
            "k",
            "a3",
            s + 3,
            b + 21_000,
            b + 22_000,
            c,
            true,
            4,
            256,
            32,
            0,
        ));
    }
    let s = 10 * u64::try_from(steps + 1).unwrap();
    ops.push((
        "k",
        "step",
        s + 1,
        steps * 23_000,
        steps * 23_000 + 1_000,
        u64::try_from(101 + steps).unwrap(),
        true,
        1,
        32,
        16,
        0,
    ));
    capture(ops, Vec::new(), Vec::new(), Vec::new())
}

/// 2026-10-23: Three 26 µs steps whose only link from the long side op `l` to `m` is an event
/// record/wait (every API call is issued before the window, so the host never binds):
///
/// ```text
/// s1: step [0,0.5] -rec ev8-                      m [24,25]  (1)  step [26]
/// s2:            (3) l [3.5,21] -rec ev7-  (3) -> m
/// s3:  x1 [2,3] (1) x2 [4,5] (1) x3 [6,7]          (stream latency 1 µs, the median)
/// ```
///
/// With the waits the max-plus step is 26 µs exactly; without the wait on ev7, `m` follows
/// `step` by the stream latency and the step collapses.
pub fn event_wait(steps: i64, with_wait: bool) -> Capture {
    let mut ops: Vec<OpRow> = Vec::new();
    let mut api = Vec::new();
    let mut record = Vec::new();
    let mut sync = Vec::new();
    let pre = -100_000 - 1_000 * steps;
    for k in -1..steps {
        let b = k * 26_000;
        let c = 20 * u64::try_from(k + 2).unwrap();
        ops.push(("k", "step", 1, b, b + 500, c, false, 1, 32, 16, 0));
        ops.push((
            "k",
            "l",
            2,
            b + 3_500,
            b + 21_000,
            c + 3,
            false,
            48,
            256,
            64,
            0,
        ));
        ops.push((
            "k",
            "m",
            1,
            b + 24_000,
            b + 25_000,
            c + 6,
            false,
            4,
            256,
            32,
            0,
        ));
        ops.push((
            "k",
            "x1",
            3,
            b + 2_000,
            b + 3_000,
            c + 7,
            false,
            1,
            32,
            16,
            0,
        ));
        ops.push((
            "k",
            "x2",
            3,
            b + 4_000,
            b + 5_000,
            c + 8,
            false,
            1,
            32,
            16,
            0,
        ));
        ops.push((
            "k",
            "x3",
            3,
            b + 6_000,
            b + 7_000,
            c + 9,
            false,
            1,
            32,
            16,
            0,
        ));
        let calls = [
            (c, "cuLaunchKernel"),
            (c + 1, "cuEventRecord"),
            (c + 2, "cuStreamWaitEvent"),
            (c + 3, "cuLaunchKernel"),
            (c + 4, "cuEventRecord"),
            (c + 5, "cuStreamWaitEvent"),
            (c + 6, "cuLaunchKernel"),
            (c + 7, "cuLaunchKernel"),
            (c + 8, "cuLaunchKernel"),
            (c + 9, "cuLaunchKernel"),
        ];
        for (j, (corr, name)) in calls.iter().enumerate() {
            let t = pre + 1_000 * (k + 1) + 50 * i64::try_from(j).unwrap();
            api.push((*name, t, t + 10, *corr));
        }
        let t = pre + 1_000 * (k + 1);
        record.push((c + 1, 1, 8));
        sync.push((c + 2, "stream_wait", 2, Some(8), t + 100, t + 110));
        if with_wait {
            record.push((c + 4, 2, 7));
            sync.push((c + 5, "stream_wait", 1, Some(7), t + 250, t + 260));
        }
    }
    let c = 20 * u64::try_from(steps + 2).unwrap();
    ops.push((
        "k",
        "step",
        1,
        steps * 26_000,
        steps * 26_000 + 500,
        c,
        false,
        1,
        32,
        16,
        0,
    ));
    api.push((
        "cuLaunchKernel",
        pre + 1_000 * (steps + 1),
        pre + 1_000 * (steps + 1) + 10,
        c,
    ));
    capture(ops, api, record, sync)
}
