// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: speculative-decode step decomposition (pure).
//!
//! Needs node-level graph trace: graph kernels share their launch's
//! correlation id, so each graph launch is one group. The verify graph is the
//! pinned largest graph (most nodes). Step i runs from the host launch call of
//! verify launch i to that of launch i + 1 (the kernel start when the call is
//! missing). Per step:
//!
//! - verify_gpu: first kernel start to last kernel end of the verify launch
//! - post: verify end to the next verify start
//! - post_gpu_idle: post minus the union of everything else that starts inside
//!   it (eager kernels, other graphs' kernels, copies), clipped
//! - host_sync: summed host time of blocking sync calls that start in the step
//! - counts: eager kernels, other-graph kernels, HtoD and DtoH copies, host
//!   sync calls
//!
//! Every value is reported as the median over steps (and the mean).

use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;
use regex::Regex;

use crate::core::error::Result;
use crate::core::model::{ApiCall, Copy, Kernel, ms};
use crate::core::stats::{mean, median_f64};
use crate::core::union::union_ns;
use crate::fail;

/// The per-step columns, in report order.
pub const COLUMNS: [&str; 10] = [
    "step_ms",
    "verify_gpu_ms",
    "post_ms",
    "post_gpu_idle_ms",
    "host_sync_ms",
    "n_host_sync",
    "n_eager",
    "n_other_graph",
    "n_htod",
    "n_dtoh",
];

#[derive(Debug, Clone, PartialEq)]
pub struct SpecSteps {
    pub graph_launches: usize,
    pub verify_nodes: usize,
    pub steps: usize,
    /// (column, median, mean) in `COLUMNS` order.
    pub columns: Vec<(&'static str, f64, f64)>,
}

fn lower_bound(xs: &[i64], t: i64) -> usize {
    xs.partition_point(|&x| x < t)
}

pub fn spec_steps(
    kernels: &[Kernel],
    copies: &[Copy],
    calls: &[ApiCall],
    host_sync: &Regex,
) -> Result<SpecSteps> {
    let mut groups: IndexMap<i64, (i64, i64, usize)> = IndexMap::new();
    for k in kernels.iter().filter(|k| k.in_graph()) {
        let g = groups.entry(k.correlation).or_insert((k.start, k.end, 0));
        g.0 = g.0.min(k.start);
        g.1 = g.1.max(k.end);
        g.2 += 1;
    }
    let Some(nodes) = groups.values().map(|g| g.2).max() else {
        fail!("no graph-node kernels: a spec step decomposition needs --cuda-graph-trace=node");
    };
    let mut verify: Vec<(i64, i64, i64)> = groups
        .iter()
        .filter(|(_, g)| g.2 == nodes)
        .map(|(c, g)| (g.0, g.1, *c))
        .collect();
    verify.sort_unstable();
    if verify.len() < 2 {
        fail!(
            "spec steps need at least 2 launches of the verify graph, found {}",
            verify.len()
        );
    }
    let vset: HashSet<i64> = verify.iter().map(|v| v.2).collect();
    let mut ks: Vec<&Kernel> = kernels.iter().collect();
    ks.sort_by_key(|k| (k.start, k.end));
    let mut cs: Vec<&ApiCall> = calls.iter().collect();
    cs.sort_by_key(|a| (a.start, a.end));
    let launch: HashMap<i64, i64> = cs.iter().map(|c| (c.correlation, c.start)).collect();
    let k_starts: Vec<i64> = ks.iter().map(|k| k.start).collect();
    let mut co: Vec<&Copy> = copies.iter().collect();
    co.sort_by_key(|c| c.start);
    let c_starts: Vec<i64> = co.iter().map(|c| c.start).collect();
    let a_starts: Vec<i64> = cs.iter().map(|a| a.start).collect();
    let mut cols: Vec<Vec<f64>> = vec![Vec::new(); COLUMNS.len()];
    for w in verify.windows(2) {
        let ((vs, ve, corr), (ns, _, ncorr)) = (w[0], w[1]);
        let t0 = launch.get(&corr).copied().unwrap_or(vs);
        let t1 = launch.get(&ncorr).copied().unwrap_or(ns);
        let (mut intervals, mut eager, mut other) = (Vec::new(), 0u32, 0u32);
        for k in &ks
            [lower_bound(&k_starts, ve)..lower_bound(&k_starts, ns).max(lower_bound(&k_starts, ve))]
        {
            if k.in_graph() && vset.contains(&k.correlation) {
                continue;
            }
            intervals.push((k.start.max(ve), k.end.min(ns)));
            if k.in_graph() {
                other += 1;
            } else {
                eager += 1;
            }
        }
        let (mut htod, mut dtoh) = (0u32, 0u32);
        for c in &co
            [lower_bound(&c_starts, ve)..lower_bound(&c_starts, ns).max(lower_bound(&c_starts, ve))]
        {
            intervals.push((c.start.max(ve), c.end.min(ns)));
            htod += u32::from(c.kind == "HtoD");
            dtoh += u32::from(c.kind == "DtoH");
        }
        let (lo, hi) = (lower_bound(&a_starts, t0), lower_bound(&a_starts, t1));
        let syncs: Vec<&&ApiCall> = cs[lo..hi.max(lo)]
            .iter()
            .filter(|a| host_sync.is_match(&a.name))
            .collect();
        let busy = union_ns(intervals.into_iter().filter(|iv| iv.1 > iv.0))?;
        let values = [
            ms((t1 - t0) as f64),
            ms((ve - vs) as f64),
            ms((ns - ve) as f64),
            ms((ns - ve - busy) as f64),
            ms(syncs.iter().map(|a| a.end - a.start).sum::<i64>() as f64),
            syncs.len() as f64,
            f64::from(eager),
            f64::from(other),
            f64::from(htod),
            f64::from(dtoh),
        ];
        for (col, v) in cols.iter_mut().zip(values) {
            col.push(v);
        }
    }
    let mut columns = Vec::with_capacity(COLUMNS.len());
    for (name, v) in COLUMNS.iter().zip(&cols) {
        let (Some(med), Some(avg)) = (median_f64(v), mean(v)) else {
            fail!("spec steps: no steps to tabulate");
        };
        columns.push((*name, med, avg));
    }
    Ok(SpecSteps {
        graph_launches: groups.len(),
        verify_nodes: nodes,
        steps: cols[0].len(),
        columns,
    })
}
