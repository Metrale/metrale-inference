// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Max-plus evaluation of the event graph: an event happens at the max over its
//! incoming edges of (source time + edge weight); the step is the longest path between two
//! anchors. The weights are the model: each operation's work and each host gap taken in
//! context, each launch latency one constant per edge kind ([`Latency`]).
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - Events before the first anchor (the context) and events no edge reaches keep their
//!   measured time; everything else is recomputed.
//! - `replay` weights (every edge its measured gap) reproduce every event exactly; the model
//!   weights differ only on latency edges, so the model's step error is the error of stating
//!   each latency as one constant.
//! - A site's work is the same number on its `Duration` and `PdlWait` edges, so a rewrite that
//!   changes it changes both.

use crate::prelude::*;

use alloc::collections::BTreeMap;

use super::graph::{CriticalGraph, EdgeKind, node_of};

/// 2026-10-23: One launch latency per edge kind: the median measured gap over the edges of
/// that kind that bound their target (the latest-arriving incoming edge); a kind that never
/// binds takes its smallest measured gap (`n` = 0), a lower bound, never the per-edge gap.
#[derive(Debug, Clone, PartialEq)]
pub struct Latency {
    /// 2026-10-23: Microseconds per kind.
    pub us: BTreeMap<EdgeKind, f64>,
    /// 2026-10-23: How many binding edges each constant is the median of.
    pub n: BTreeMap<EdgeKind, usize>,
}

/// 2026-10-23: The median of `v` (sorts it); 0 for an empty slice.
pub fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        0.5 * (v[m - 1] + v[m])
    }
}

/// 2026-10-23: Fit [`Latency`] on the capture's own window.
pub fn fit_latency(g: &CriticalGraph) -> Latency {
    let mut gaps: BTreeMap<EdgeKind, Vec<f64>> = BTreeMap::new();
    for v in 0..g.t.len() {
        let v32 = u32::try_from(v).unwrap_or(u32::MAX);
        if g.t[v] < 0.0 {
            continue;
        }
        let bind = g
            .incoming(v32)
            .iter()
            .map(|&k| k as usize)
            .min_by(|&a, &b| g.gap(a).total_cmp(&g.gap(b)));
        if let Some(k) = bind
            && g.edges[k].kind.is_latency()
        {
            gaps.entry(g.edges[k].kind).or_default().push(g.gap(k));
        }
    }
    let mut n: BTreeMap<EdgeKind, usize> = gaps.iter().map(|(k, v)| (*k, v.len())).collect();
    let mut us: BTreeMap<EdgeKind, f64> = gaps
        .into_iter()
        .map(|(k, mut v)| (k, median(&mut v)))
        .collect();
    for (k, e) in g.edges.iter().enumerate() {
        if e.kind.is_latency() && n.get(&e.kind).is_none_or(|&c| c == 0) {
            let x = us.entry(e.kind).or_insert(f64::INFINITY);
            *x = x.min(g.gap(k));
            n.insert(e.kind, 0);
        }
    }
    Latency { us, n }
}

/// 2026-10-23: Per node, its work in context: an operation's duration, or for a programmatic
/// successor the part after the last predecessor it waits on finished; an API call's host time,
/// or 0 for a call that blocks on the GPU (its wait is the `Sync` edge's).
pub fn work_in_context(g: &CriticalGraph) -> Vec<f64> {
    let mut wait = vec![f64::NEG_INFINITY; g.nodes.len()];
    let mut blocking = vec![false; g.nodes.len()];
    for e in &g.edges {
        match e.kind {
            EdgeKind::PdlWait => {
                let i = node_of(e.dst);
                wait[i] = wait[i].max(g.t[e.src as usize]);
            }
            EdgeKind::Sync => blocking[node_of(e.dst)] = true,
            _ => {}
        }
    }
    (0..g.nodes.len())
        .map(|i| {
            if blocking[i] {
                0.0
            } else {
                g.t[2 * i + 1] - g.t[2 * i].max(wait[i])
            }
        })
        .collect()
}

/// 2026-10-23: Edge weights from per-node work, the capture's host gaps and programmatic start
/// offsets, and `lat` (`None`: every latency edge its measured gap, the replay).
pub fn weights(g: &CriticalGraph, work: &[f64], lat: Option<&Latency>) -> Vec<f64> {
    g.edges
        .iter()
        .enumerate()
        .map(|(k, e)| match e.kind {
            EdgeKind::Duration | EdgeKind::ApiDuration | EdgeKind::PdlWait => work[node_of(e.dst)],
            kind if kind.is_latency() => match lat {
                Some(l) => l.us.get(&kind).copied().unwrap_or(0.0),
                None => g.gap(k),
            },
            _ => g.gap(k),
        })
        .collect()
}

/// 2026-10-23: Earliest time of every event under weights `w`.
pub fn longest(g: &CriticalGraph, w: &[f64]) -> Vec<f64> {
    let mut at = vec![f64::NEG_INFINITY; g.t.len()];
    for &v in &g.topo {
        let inc = g.incoming(v);
        let v = v as usize;
        if inc.is_empty() || g.t[v] < 0.0 {
            at[v] = g.t[v];
            continue;
        }
        let mut best = f64::NEG_INFINITY;
        for &k in inc {
            let e = g.edges[k as usize];
            best = best.max(at[e.src as usize] + w[k as usize]);
        }
        at[v] = best;
    }
    at
}

/// 2026-10-23: The mean step between the first and the last anchor, microseconds.
pub fn step_us(g: &CriticalGraph, at: &[f64]) -> f64 {
    let (a, b) = (
        g.anchors[0] as usize,
        g.anchors[g.anchors.len() - 1] as usize,
    );
    (at[b] - at[a]) / (g.anchors.len() - 1) as f64
}

/// 2026-10-23: The critical chain: the edges that bind, walked back from the last anchor to the
/// first, latest first.
pub fn critical_chain(g: &CriticalGraph, w: &[f64], at: &[f64]) -> Vec<u32> {
    let stop = at[g.anchors[0] as usize];
    let mut v = g.anchors[g.anchors.len() - 1];
    let mut chain = Vec::new();
    while at[v as usize] > stop {
        let Some(&k) = g.incoming(v).iter().max_by(|&&a, &&b| {
            let ka = at[g.edges[a as usize].src as usize] + w[a as usize];
            let kb = at[g.edges[b as usize].src as usize] + w[b as usize];
            ka.total_cmp(&kb)
        }) else {
            break;
        };
        chain.push(k);
        v = g.edges[k as usize].src;
    }
    chain
}

/// 2026-10-23: Latest time of every event that keeps the last anchor at `at`; `+inf` for an
/// event the last anchor does not depend on.
pub fn latest(g: &CriticalGraph, w: &[f64], at: &[f64]) -> Vec<f64> {
    let mut late = vec![f64::INFINITY; g.t.len()];
    let last = g.anchors[g.anchors.len() - 1] as usize;
    late[last] = at[last];
    for &v in g.topo.iter().rev() {
        let lv = late[v as usize];
        if lv == f64::INFINITY {
            continue;
        }
        for &k in g.incoming(v) {
            let s = g.edges[k as usize].src as usize;
            late[s] = late[s].min(lv - w[k as usize]);
        }
    }
    late
}
