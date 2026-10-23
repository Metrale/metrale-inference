// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: What-if rewrites of the event graph: shrink, fuse, persistent segment, PDL off,
//! an added side-stream kernel, and an added dependency (moving work off a side stream). A
//! [`Scenario`] is the graph plus everything the evaluators read; a rewrite changes it and the
//! max-plus and the simulation re-evaluate it.
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - A rewrite that matches nothing is an error ([`ErrorCode::WhatIfInvalid`]), never a no-op.
//! - A fused chain keeps its first member: it carries the members' work (times the stated ratio)
//!   and their bytes, the others carry nothing, and the latencies between members are zero.
//! - Every number a rewrite adds (a ratio, a handoff cost, a side kernel's work) is stated in
//!   the spec with its basis; the module adds none of its own except PDL-off's prologue, which
//!   it estimates from the capture and reports ([`PdlOffStats`]).

use crate::prelude::*;

use alloc::collections::BTreeMap;

use metrale_error::{ErrorCode, error};
use serde::Deserialize;

use super::graph::{EdgeKind, Graph, Node, NodeKind, ix, node_of};
use super::maxplus::{Latency, weights};
use super::sim::{Resources, cap_of};

/// 2026-10-23: Selects operations by attributed site (`block.local`) and/or kernel name.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sel {
    /// 2026-10-23: The site the attribution map gives the launch.
    pub site: Option<String>,
    /// 2026-10-23: The kernel function.
    pub kernel: Option<String>,
}

/// 2026-10-23: One rewrite.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Rewrite {
    /// 2026-10-23: Scale the selected operations' work (and solo work) by `factor`.
    Shrink {
        select: Sel,
        factor: f64,
        basis: String,
    },
    /// 2026-10-23: Fuse each run of `chain` on one stream into one kernel of `work_ratio` times
    /// the members' work.
    Fuse {
        chain: Vec<Sel>,
        work_ratio: f64,
        basis: String,
    },
    /// 2026-10-23: As `fuse` (ratio 1) plus `extra_us` of in-kernel handoff per run, holding
    /// `sms` SMs from its start to its end, through its phase waits.
    Persistent {
        chain: Vec<Sel>,
        sms: f64,
        extra_us: f64,
        basis: String,
    },
    /// 2026-10-23: Every programmatic successor starts after its predecessor finished, plus
    /// (`estimate`) the prologue it no longer overlaps.
    PdlOff { prologue: Prologue, basis: String },
    /// 2026-10-23: A kernel forked when each `fork_at` operation preceding a `target` on its
    /// stream starts; it reads `bytes` in `work_us` of solo work, and `target` saves `saves_us`.
    SideKernel {
        fork_at: Sel,
        target: Sel,
        bytes: f64,
        work_us: f64,
        ctas: u32,
        threads: u32,
        regs: u32,
        smem: u32,
        saves_us: f64,
        basis: String,
    },
    /// 2026-10-23: Each `select` operation also waits on the `wait_for` operation whose measured
    /// start is nearest to it within `window_us` (serializing it after that work).
    After {
        select: Sel,
        wait_for: Sel,
        window_us: f64,
        basis: String,
    },
}

/// 2026-10-23: What PDL-off adds to a successor's work beyond the ordinary launch latency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Prologue {
    /// 2026-10-23: Nothing: only the graph-stream latency is added.
    Zero,
    /// 2026-10-23: Per launch group and configuration seen both with and without a programmatic
    /// predecessor, the median full duration without minus the median work with (groups seen
    /// only one way add nothing, counted in [`PdlOffStats`]).
    Estimate,
}

/// 2026-10-23: The graph and every per-node input of the evaluators.
#[derive(Debug, Clone)]
pub struct Scenario {
    /// 2026-10-23: The graph.
    pub g: Graph,
    /// 2026-10-23: Per node, work in context (max-plus).
    pub work: Vec<f64>,
    /// 2026-10-23: Per node, the capture's work in context (before any rewrite): a programmatic
    /// successor's launch offset scales with its predecessor's duration change.
    pub work0: Vec<f64>,
    /// 2026-10-23: Per node, solo work (simulation).
    pub solo: Vec<f64>,
    /// 2026-10-23: Per node, DRAM bytes.
    pub bytes: Vec<f64>,
    /// 2026-10-23: Capacity overrides, SM units.
    pub cap: Vec<(usize, f64)>,
    /// 2026-10-23: Edge weight overrides.
    pub over: BTreeMap<u32, f64>,
    /// 2026-10-23: Per node, the attributed site (`block.local`) of its launch group.
    pub site: Vec<Option<String>>,
    /// 2026-10-23: Per node, its kernel name.
    pub name: Vec<String>,
}

/// 2026-10-23: What PDL-off estimated.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PdlOffStats {
    /// 2026-10-23: Programmatic edges turned into ordinary ones.
    pub edges: usize,
    /// 2026-10-23: Kernel classes whose prologue was estimated / left at 0 (never seen without
    /// a programmatic predecessor).
    pub classes_estimated: usize,
    /// 2026-10-23: Classes left at 0.
    pub classes_unestimated: usize,
}

pub(super) fn bad(what: &str, why: String) -> metrale_error::Error {
    error!(ErrorCode::WhatIfInvalid, what.to_string(), why)
}

impl Scenario {
    /// 2026-10-23: The max-plus weights under `lat`, overrides applied.
    pub fn weights(&self, lat: &Latency) -> Vec<f64> {
        let mut w = weights(&self.g, &self.work, Some(lat));
        for (k, e) in self.g.edges.iter().enumerate() {
            let p = node_of(e.src);
            if e.kind == EdgeKind::PdlStart && self.work[p] != self.work0[p] {
                let full = self.g.t[2 * p + 1] - self.g.t[2 * p];
                if full > 0.0 {
                    w[k] *= ((full - (self.work0[p] - self.work[p])) / full).max(0.0);
                }
            }
        }
        for (&k, &v) in &self.over {
            w[k as usize] = v;
        }
        w
    }

    /// 2026-10-23: The simulation's weights: as [`Scenario::weights`], with every residency wait
    /// back to the graph-stream latency (the simulation regenerates the wait).
    pub fn sim_weights(&self, lat: &Latency) -> Vec<f64> {
        let mut w = self.weights(lat);
        let c = lat.us.get(&EdgeKind::GraphStream).copied().unwrap_or(0.0);
        for (k, e) in self.g.edges.iter().enumerate() {
            if e.kind == EdgeKind::Residency && !self.over.contains_key(&ix(k)) {
                w[k] = c;
            }
        }
        w
    }

    pub(super) fn matches(&self, i: usize, s: &Sel) -> bool {
        self.g.nodes[i].kind != NodeKind::Api
            && self.g.t[2 * i] >= 0.0
            && s.site
                .as_ref()
                .is_none_or(|x| self.site[i].as_ref() == Some(x))
            && s.kernel.as_ref().is_none_or(|x| &self.name[i] == x)
    }

    /// 2026-10-23: Per operation, the next one on its stream (same-stream order edges).
    pub(super) fn next_on_stream(&self) -> Vec<Option<usize>> {
        let mut next = vec![None; self.g.nodes.len()];
        for e in &self.g.edges {
            if matches!(
                e.kind,
                EdgeKind::GraphStream | EdgeKind::PdlStart | EdgeKind::Stream | EdgeKind::Residency
            ) {
                let (a, b) = (node_of(e.src), node_of(e.dst));
                if self.g.nodes[a].stream == self.g.nodes[b].stream {
                    next[a] = Some(b);
                }
            }
        }
        next
    }

    /// 2026-10-23: Every run of `chain` on one stream.
    pub fn chains(&self, chain: &[Sel]) -> Vec<Vec<usize>> {
        let next = self.next_on_stream();
        let mut out = Vec::new();
        for i in 0..self.g.nodes.len() {
            if chain.is_empty() || !self.matches(i, &chain[0]) {
                continue;
            }
            let mut run = vec![i];
            for s in &chain[1..] {
                match next[*run.last().unwrap_or(&i)] {
                    Some(n) if self.matches(n, s) => run.push(n),
                    _ => break,
                }
            }
            if run.len() == chain.len() {
                out.push(run);
            }
        }
        out
    }

    pub(super) fn fuse_run(
        &mut self,
        run: &[usize],
        ratio: f64,
        extra: f64,
        cap: Option<f64>,
        r: &Resources,
    ) {
        let lead = run[0];
        let sum = |v: &[f64]| run.iter().map(|&m| v[m]).sum::<f64>();
        let (work, solo, bytes) = (sum(&self.work), sum(&self.solo), sum(&self.bytes));
        let widest = run
            .iter()
            .map(|&m| cap_of(&self.g, m, r))
            .fold(0.0, f64::max);
        for &m in run {
            self.work[m] = 0.0;
            self.solo[m] = 0.0;
            self.bytes[m] = 0.0;
            self.cap.push((m, 0.0));
        }
        self.work[lead] = ratio * work + extra;
        self.solo[lead] = ratio * solo + extra;
        self.bytes[lead] = bytes;
        self.cap.push((lead, cap.unwrap_or(widest)));
        for w in run.windows(2) {
            for (k, e) in self.g.edges.iter().enumerate() {
                if node_of(e.src) == w[0] && node_of(e.dst) == w[1] && e.kind != EdgeKind::PdlWait {
                    self.over.insert(ix(k), 0.0);
                }
            }
        }
    }

    /// 2026-10-23: Apply `rw`; returns how many places it changed.
    pub fn apply(
        &mut self,
        rw: &Rewrite,
        lat: &Latency,
        r: &Resources,
    ) -> metrale_error::Result<usize> {
        let sels: Vec<&Sel> = match rw {
            Rewrite::Shrink { select, .. } => vec![select],
            Rewrite::Fuse { chain, .. } | Rewrite::Persistent { chain, .. } => {
                chain.iter().collect()
            }
            Rewrite::PdlOff { .. } => Vec::new(),
            Rewrite::SideKernel {
                fork_at, target, ..
            } => vec![fork_at, target],
            Rewrite::After {
                select, wait_for, ..
            } => vec![select, wait_for],
        };
        if sels.iter().any(|s| s.site.is_none() && s.kernel.is_none()) {
            return Err(bad(
                &format!("{rw:?}"),
                "a selection names neither a site nor a kernel".into(),
            ));
        }
        let n = match rw {
            Rewrite::Shrink { select, factor, .. } => {
                if factor.is_nan() || *factor < 0.0 {
                    return Err(bad("shrink", format!("factor {factor} must be >= 0")));
                }
                let hit: Vec<usize> = (0..self.g.nodes.len())
                    .filter(|&i| self.matches(i, select))
                    .collect();
                for &i in &hit {
                    self.work[i] *= factor;
                    self.solo[i] *= factor;
                }
                hit.len()
            }
            Rewrite::Fuse {
                chain, work_ratio, ..
            } => {
                let runs = self.chains(chain);
                for run in &runs {
                    self.fuse_run(run, *work_ratio, 0.0, None, r);
                }
                runs.len()
            }
            Rewrite::Persistent {
                chain,
                sms,
                extra_us,
                ..
            } => {
                let runs = self.chains(chain);
                for run in &runs {
                    self.fuse_run(run, 1.0, *extra_us, Some(*sms), r);
                }
                runs.len()
            }
            Rewrite::PdlOff { prologue, .. } => self.pdl_off(lat, *prologue)?.edges,
            Rewrite::SideKernel {
                fork_at,
                target,
                bytes,
                work_us,
                ctas,
                threads,
                regs,
                smem,
                saves_us,
                ..
            } => {
                let node = Node {
                    kind: NodeKind::Kernel,
                    group: None,
                    ctas: *ctas,
                    threads: *threads,
                    regs: *regs,
                    smem: *smem,
                    stream: u64::MAX,
                };
                self.side_kernel(fork_at, target, node, *bytes, *work_us, *saves_us, lat)?
            }
            Rewrite::After {
                select,
                wait_for,
                window_us,
                ..
            } => self.after(select, wait_for, *window_us, lat)?,
        };
        if n == 0 {
            return Err(bad(
                &format!("{rw:?}"),
                "matches nothing in the capture".into(),
            ));
        }
        Ok(n)
    }
}
