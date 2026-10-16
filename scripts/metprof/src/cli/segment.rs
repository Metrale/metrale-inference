// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `segment requests | decode | spec`: per-request and per-step
//! tables.
//!
//! A table is always divided by its unit count, and the count is printed.
//! `requests` never prints a per-step column; `decode` refuses a window that is
//! not dominated by the pinned decode graph (a prefill window has no step).

use serde_json::{Map, Value, json};

use crate::cli::args::{Args, Kind, Opt, Spec};
use crate::cli::busy::{CAPTURE, DEVICE, FAMILIES};
use crate::cli::{Command, Io, Run, load};
use crate::core::capture::{device_copies, device_kernels, graph_node_traced, ordered_calls};
use crate::core::error::Result;
use crate::core::model::{ms, ms_to_ns};
use crate::core::segment::{
    Unit, decode_steps, inter_gaps, per_unit_rows, split_requests, step_stats,
};
use crate::core::spec::spec_steps;
use crate::core::tables::into_map;
use crate::core::union::union_ns;
use crate::fail;

const TOP: Opt = Opt {
    name: "top",
    kind: Kind::Int,
    required: true,
    metavar: "N",
    help: "rows per table",
};

pub const COMMANDS: &[&Command] = &[
    &Command {
        spec: Spec {
            name: "segment requests",
            about: "split the window into requests at idle gaps; per-request tables",
            positionals: &[CAPTURE],
            opts: &[
                DEVICE,
                FAMILIES,
                Opt {
                    name: "gap-ms",
                    kind: Kind::Float,
                    required: true,
                    metavar: "MS",
                    help: "a request boundary is an idle gap at least this long",
                },
                Opt {
                    name: "expect",
                    kind: Kind::Int,
                    required: true,
                    metavar: "N",
                    help: "requests in the window; any other count is an error",
                },
                TOP,
            ],
        },
        run: Run::Report(requests),
    },
    &Command {
        spec: Spec {
            name: "segment decode",
            about: "pinned decode steps (one launch of the largest graph)",
            positionals: &[CAPTURE],
            opts: &[
                DEVICE,
                FAMILIES,
                Opt {
                    name: "min-graph-share",
                    kind: Kind::Float,
                    required: true,
                    metavar: "SHARE",
                    help: "refuse unless the pinned graph covers at least this share of busy time",
                },
                TOP,
            ],
        },
        run: Run::Report(decode),
    },
    &Command {
        spec: Spec {
            name: "segment spec",
            about: "speculative step split: verify / post / host sync (node trace required)",
            positionals: &[CAPTURE],
            opts: &[DEVICE, FAMILIES],
        },
        run: Run::Report(spec),
    },
];

fn top(a: &Args) -> Result<usize> {
    let n = a.int("top")?;
    match usize::try_from(n) {
        Ok(n) => Ok(n),
        Err(_) => fail!("argument --top: must be at least 0, got {n}"),
    }
}

fn requests(a: &Args, io: &Io) -> Result<Map<String, Value>> {
    let table = load::load_families(io, a.str("families")?)?;
    let (kernels, labels) = load::classified_window(io, a.pos(0)?, a.int("device")?, &table)?;
    let gap_ms = a.float("gap-ms")?;
    let segs = split_requests(&kernels, ms_to_ns(gap_ms), a.int("expect")?)?;
    let t0 = segs.first().map(|s| s.start).unwrap_or(0);
    let gaps = inter_gaps(&segs);
    let units: Vec<Vec<usize>> = segs.iter().map(|s| s.kernels.clone()).collect();
    let mut rows = Vec::with_capacity(segs.len());
    for (n, s) in segs.iter().enumerate() {
        let ks = s.kernels.iter().map(|&i| &kernels[i]);
        rows.push(json!({
            "request": n,
            "start_ms": ms((s.start - t0) as f64),
            "span_ms": ms((s.end - s.start) as f64),
            "kernels": s.kernels.len(),
            "u_all_ms": ms(union_ns(ks.clone().map(|k| (k.start, k.end)))? as f64),
            "sum_ms": ms(ks.map(|k| k.dur()).sum::<i64>() as f64),
            "max_intra_gap_ms": ms(s.max_intra_gap as f64),
        }));
    }
    let n = top(a)?;
    Ok(into_map(json!({
        "requests": segs.len(),
        "gap_threshold_ms": gap_ms,
        "max_intra_request_gap_ms": ms(segs.iter().map(|s| s.max_intra_gap).max().unwrap_or(0) as f64),
        "min_inter_request_gap_ms": gaps.iter().min().map(|g| ms(*g as f64)),
        "segments": rows,
        "families_per_request": per_unit_rows(&kernels, &labels, &units, Unit::Request, false, n)?,
        "kernels_per_request": per_unit_rows(&kernels, &labels, &units, Unit::Request, true, n)?,
    })))
}

fn decode(a: &Args, io: &Io) -> Result<Map<String, Value>> {
    let table = load::load_families(io, a.str("families")?)?;
    let (kernels, labels) = load::classified_window(io, a.pos(0)?, a.int("device")?, &table)?;
    let (ps, share) = decode_steps(&kernels, a.float("min-graph-share")?)?;
    let st = step_stats(&kernels, &ps)?;
    let n = top(a)?;
    Ok(into_map(json!({
        "steps": ps.steps.len(),
        "nodes_per_step": ps.nodes,
        "graph_launches": ps.graph_launches,
        "graph_share": share,
        "span_median_ms": st.span_median_ms,
        "sum_per_step_ms": st.sum_per_step_ms,
        "idle_median_ms": st.idle_median_ms,
        "families_per_step": per_unit_rows(&kernels, &labels, &ps.steps, Unit::Step, false, n)?,
        "kernels_per_step": per_unit_rows(&kernels, &labels, &ps.steps, Unit::Step, true, n)?,
    })))
}

fn spec(a: &Args, io: &Io) -> Result<Map<String, Value>> {
    let table = load::load_families(io, a.str("families")?)?;
    let path = a.pos(0)?;
    let device = a.int("device")?;
    let trace = io.traces.open(path)?;
    let rows = trace.kernel_rows()?;
    if !graph_node_traced(&rows) {
        fail!(
            "the capture has no graph-node kernels: a spec step decomposition needs --cuda-graph-trace=node"
        );
    }
    let kernels = device_kernels(path, rows, device)?;
    let copies = device_copies(trace.copy_rows()?, device);
    let calls = ordered_calls(trace.call_rows()?);
    let s = spec_steps(&kernels, &copies, &calls, &table.host_sync)?;
    let medians: Map<String, Value> = s
        .columns
        .iter()
        .map(|(c, m, _)| (c.to_string(), json!(m)))
        .collect();
    let means: Map<String, Value> = s
        .columns
        .iter()
        .map(|(c, _, m)| (c.to_string(), json!(m)))
        .collect();
    Ok(into_map(json!({
        "graph_launches": s.graph_launches,
        "verify_nodes": s.verify_nodes,
        "steps": s.steps,
        "medians": medians,
        "means": means,
    })))
}
