// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The graph-changing rewrites of [`super::scenario::Scenario`]: added nodes, PDL
//! off, a forked side kernel and an added wait (split from `scenario.rs` for the 500-line cap).
//!
//! Owner: metrale-circuit (hardware).

use crate::prelude::*;

use alloc::collections::BTreeMap;

use super::graph::{Edge, EdgeKind, Node, NodeKind, f_of, ix, node_of, s_of};
use super::maxplus::{Latency, median};
use super::scenario::{PdlOffStats, Prologue, Scenario, Sel, bad};
use super::sim::{Resources, cap_of};

impl Scenario {
    pub(super) fn add(
        &mut self,
        edges: Vec<Edge>,
        extra: Vec<(Node, f64, f64, f64, f64, String)>,
    ) -> metrale_error::Result<()> {
        let mut e = self.g.edges.clone();
        e.extend(edges);
        let mut nodes = Vec::new();
        for (n, s, f, work, bytes, name) in extra {
            nodes.push((n, s, f));
            self.work.push(work);
            self.work0.push(work);
            self.solo.push(work);
            self.bytes.push(bytes);
            self.site.push(None);
            self.name.push(name);
        }
        self.g = self.g.rewired(e, nodes)?;
        Ok(())
    }

    pub(super) fn pdl_off(
        &mut self,
        lat: &Latency,
        mode: Prologue,
    ) -> metrale_error::Result<PdlOffStats> {
        if !lat.us.contains_key(&EdgeKind::GraphStream) {
            return Err(bad("pdl_off", "no graph-stream latency was fitted".into()));
        }
        type Class = (Option<u32>, u32, u32, u32, u32);
        let class = |s: &Scenario, i: usize| {
            let n = &s.g.nodes[i];
            (n.group, n.ctas, n.threads, n.regs, n.smem)
        };
        let mut pdl_succ = vec![false; self.g.nodes.len()];
        let mut pairs = Vec::new();
        for e in &self.g.edges {
            if e.kind == EdgeKind::PdlStart {
                pdl_succ[node_of(e.dst)] = true;
                pairs.push((node_of(e.src), node_of(e.dst)));
            }
        }
        let mut full: BTreeMap<Class, Vec<f64>> = BTreeMap::new();
        let mut tail: BTreeMap<Class, Vec<f64>> = BTreeMap::new();
        for i in 0..self.g.nodes.len() {
            if self.g.nodes[i].kind != NodeKind::Kernel || self.g.t[2 * i] < 0.0 {
                continue;
            }
            let d = self.g.t[2 * i + 1] - self.g.t[2 * i];
            if pdl_succ[i] {
                tail.entry(class(self, i)).or_default().push(self.work[i]);
            } else {
                full.entry(class(self, i)).or_default().push(d);
            }
        }
        let mut stats = PdlOffStats::default();
        let mut prologue: BTreeMap<Class, f64> = BTreeMap::new();
        for (k, mut v) in tail {
            match full.get_mut(&k) {
                Some(f) => {
                    prologue.insert(k, (median(f) - median(&mut v)).max(0.0));
                    stats.classes_estimated += 1;
                }
                None => {
                    prologue.insert(k, 0.0);
                    stats.classes_unestimated += 1;
                }
            }
        }
        let mut edges = Vec::new();
        let mut waited = vec![false; self.g.nodes.len()];
        for e in &self.g.edges {
            if e.kind == EdgeKind::PdlWait {
                waited[node_of(e.dst)] = true;
            }
        }
        for &(p, s) in &pairs {
            edges.push(Edge {
                src: f_of(p),
                dst: s_of(s),
                kind: EdgeKind::GraphStream,
            });
            let full_d = self.g.t[2 * s + 1] - self.g.t[2 * s];
            let add = match mode {
                Prologue::Estimate if waited[s] => {
                    prologue.get(&class(self, s)).copied().unwrap_or(0.0)
                }
                _ => 0.0,
            };
            let w = if waited[s] {
                self.work[s] + add
            } else {
                full_d
            };
            if self.work[s] > 0.0 {
                self.solo[s] *= w / self.work[s];
            }
            self.work[s] = w;
        }
        stats.edges = edges.len();
        for (k, e) in self.g.edges.iter().enumerate() {
            if e.kind == EdgeKind::PdlStart {
                self.over.insert(ix(k), f64::NEG_INFINITY);
            }
        }
        self.add(edges, Vec::new())?;
        Ok(stats)
    }

    pub(super) fn prev_on_stream(&self) -> Vec<Option<usize>> {
        let mut prev = vec![None; self.g.nodes.len()];
        for (a, b) in self.next_on_stream().iter().enumerate() {
            if let Some(b) = b {
                prev[*b] = Some(a);
            }
        }
        prev
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn side_kernel(
        &mut self,
        fork_at: &Sel,
        target: &Sel,
        node: Node,
        bytes: f64,
        work: f64,
        saves: f64,
        lat: &Latency,
    ) -> metrale_error::Result<usize> {
        let prev = self.prev_on_stream();
        let join = lat.us.get(&EdgeKind::Join).copied().unwrap_or(0.0);
        let mut edges = Vec::new();
        let mut extra = Vec::new();
        let base = self.g.nodes.len();
        for t in 0..self.g.nodes.len() {
            if !self.matches(t, target) || self.g.t[2 * t] < 0.0 {
                continue;
            }
            let mut p = prev[t];
            let mut hops = 0;
            while let Some(q) = p {
                if self.matches(q, fork_at) || hops >= 6 {
                    break;
                }
                p = prev[q];
                hops += 1;
            }
            let Some(f) = p.filter(|&q| self.matches(q, fork_at)) else {
                continue;
            };
            let x = base + extra.len();
            let at = self.g.t[2 * f];
            extra.push((
                node.clone(),
                at,
                at + work,
                work,
                bytes,
                "side_kernel".to_string(),
            ));
            edges.push(Edge {
                src: s_of(f),
                dst: s_of(x),
                kind: EdgeKind::Join,
            });
            edges.push(Edge {
                src: s_of(x),
                dst: f_of(x),
                kind: EdgeKind::Duration,
            });
            let w = self.work[t];
            let after = (w - saves).max(0.0);
            if w > 0.0 {
                self.solo[t] *= after / w;
            }
            self.work[t] = after;
        }
        let n = extra.len();
        let first_new = self.g.edges.len();
        self.add(edges, extra)?;
        for k in first_new..self.g.edges.len() {
            if self.g.edges[k].kind == EdgeKind::Join {
                self.over.insert(ix(k), join);
            }
        }
        Ok(n)
    }

    pub(super) fn after(
        &mut self,
        select: &Sel,
        wait_for: &Sel,
        window: f64,
        lat: &Latency,
    ) -> metrale_error::Result<usize> {
        let mut starts: Vec<(f64, usize)> = (0..self.g.nodes.len())
            .filter(|&i| self.matches(i, wait_for))
            .map(|i| (self.g.t[2 * i], i))
            .collect();
        starts.sort_by(|a, b| a.0.total_cmp(&b.0));
        let join = lat.us.get(&EdgeKind::Join).copied().unwrap_or(0.0);
        let mut edges = Vec::new();
        for i in 0..self.g.nodes.len() {
            if !self.matches(i, select) || self.g.t[2 * i] < 0.0 {
                continue;
            }
            let t = self.g.t[2 * i];
            let k = starts.partition_point(|x| x.0 < t);
            let cand = [k.checked_sub(1), Some(k)];
            let best = cand
                .iter()
                .flatten()
                .filter_map(|&k| starts.get(k))
                .filter(|x| (x.0 - t).abs() <= window && x.1 != i)
                .min_by(|a, b| (a.0 - t).abs().total_cmp(&(b.0 - t).abs()));
            if let Some(&(_, q)) = best {
                edges.push(Edge {
                    src: f_of(q),
                    dst: s_of(i),
                    kind: EdgeKind::Join,
                });
            }
        }
        let n = edges.len();
        let first_new = self.g.edges.len();
        self.add(edges, Vec::new())?;
        for k in first_new..self.g.edges.len() {
            self.over.insert(ix(k), join);
        }
        Ok(n)
    }

    /// 2026-10-23: [`super::scenario::Rewrite::Confine`]; returns the kernels it confined.
    pub(super) fn confine(
        &mut self,
        select: &[Sel],
        sms: f64,
        r: &Resources,
    ) -> metrale_error::Result<usize> {
        if !(sms.is_finite() && sms > 0.0 && sms <= r.sms) {
            return Err(bad(
                "confine",
                format!("{sms} SMs is not in (0, {}]", r.sms),
            ));
        }
        let hit: Vec<usize> = (0..self.g.nodes.len())
            .filter(|&i| self.g.nodes[i].kind == NodeKind::Kernel)
            .filter(|&i| select.iter().any(|s| self.matches(i, s)))
            .collect();
        for &i in &hit {
            let want = cap_of(&self.g, i, r);
            if want > sms {
                let stretch = want / sms;
                self.work[i] *= stretch;
                self.solo[i] *= stretch;
                self.cap.push((i, sms));
            }
        }
        Ok(hit.len())
    }
}
