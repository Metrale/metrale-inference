// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `summarize`: wire the capture and the family table to the pure
//! union core.

use indexmap::IndexMap;
use serde_json::{Map, Value, json};

use crate::cli::args::{Args, Kind, Opt, Spec};
use crate::cli::{Command, Io, Run, load};
use crate::core::error::Result;
use crate::core::families::{classify, coverage_gate};
use crate::core::model::{KernelKey, ms, ms_to_ns};
use crate::core::tables::into_map;
use crate::core::union::{attribute, busy_summary, union_ns};

pub const CAPTURE: (&str, &str) = ("capture", "nsys sqlite export");
pub const DEVICE: Opt = Opt {
    name: "device",
    kind: Kind::Int,
    required: true,
    metavar: "N",
    help: "CUDA device id to analyse",
};
pub const FAMILIES: Opt = Opt {
    name: "families",
    kind: Kind::File,
    required: true,
    metavar: "TOML",
    help: "family table (config/families.toml)",
};

pub const COMMANDS: &[&Command] = &[&Command {
    spec: Spec {
        name: "summarize",
        about: "span, work, busy union, exposed comm, idle, host gaps",
        positionals: &[CAPTURE],
        opts: &[
            DEVICE,
            FAMILIES,
            Opt {
                name: "gap-ms",
                kind: Kind::Float,
                required: true,
                metavar: "MS",
                help: "report idle gaps at least this long",
            },
            Opt {
                name: "top",
                kind: Kind::Int,
                required: true,
                metavar: "N",
                help: "kernels to list by union-attributed time",
            },
        ],
    },
    run: Run::Report(summarize),
}];

fn summarize(a: &Args, io: &Io) -> Result<Map<String, Value>> {
    let table = load::load_families(io, a.str("families")?)?;
    let device = a.int("device")?;
    let gap_ms = a.float("gap-ms")?;
    let kernels = load::kernels(io, a.pos(0)?, device)?;
    let fams = classify(&kernels, &table);
    let mut classes = Vec::with_capacity(fams.len());
    for f in &fams {
        classes.push(match f {
            Some(f) => Some(table.class(f)?),
            None => None,
        });
    }
    let intervals: Vec<(i64, i64)> = kernels.iter().map(|k| (k.start, k.end)).collect();
    let s = busy_summary(&intervals, &classes, ms_to_ns(gap_ms))?;
    let classified = union_ns(
        intervals
            .iter()
            .zip(&classes)
            .filter(|(_, c)| c.is_some())
            .map(|(iv, _)| *iv),
    )?;
    let coverage = coverage_gate(classified as f64, s.u_all_ns as f64, &table)?;
    let att = attribute(kernels.iter().map(|k| (k.start, k.end, k.key())))?;
    let mut top: Vec<(&KernelKey, &f64)> = att.iter().collect();
    top.sort_by(|x, y| y.1.total_cmp(x.1));
    let n_top = usize::try_from(a.int("top")?).unwrap_or(0);
    let mut work: IndexMap<KernelKey, (u64, i64)> = IndexMap::new();
    for k in &kernels {
        let w = work.entry(k.key()).or_insert((0, 0));
        w.0 += 1;
        w.1 += k.dur();
    }
    let rows: Vec<Value> = top
        .into_iter()
        .take(n_top)
        .map(|(key, v)| {
            let (launches, sum) = work.get(key).copied().unwrap_or((0, 0));
            let (x, y, z) = key.1;
            json!({"union_ms": ms(*v), "sum_ms": ms(sum as f64), "launches": launches, "grid": [x, y, z], "kernel": key.0})
        })
        .collect();
    Ok(into_map(json!({
        "device": device,
        "kernels": s.intervals,
        "span_ms": ms(s.span_ns as f64),
        "sum_ms": ms(s.sum_ns as f64),
        "u_all_ms": ms(s.u_all_ns as f64),
        "u_compute_ms": ms(s.u_compute_ns as f64),
        "u_comm_ms": ms(s.u_comm_ns as f64),
        "exposed_comm_ms": ms(s.exposed_comm_ns as f64),
        "idle_ms": ms(s.idle_ns as f64),
        "gap_threshold_ms": gap_ms,
        "gaps": s.gap_count,
        "gaps_ms": ms(s.gap_ns as f64),
        "coverage": coverage,
        "top_kernels": rows,
    })))
}
