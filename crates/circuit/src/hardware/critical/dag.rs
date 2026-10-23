// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The event graph of a capture. Every GPU operation and every API call is two
//! events, its start `S` and its finish `F`; an edge says one event cannot happen before another
//! (plus a latency). Edges come from what the capture states (stream order, event record/wait
//! pairs, host launches, host synchronizations) and, inside a CUDA graph, whose edges nsys does
//! not export, from the one inference this module makes: a node whose start its own stream does
//! not explain waited on the latest node to finish on another stream (a fork or a join).
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - Every edge runs forward in measured time (source time <= target time); an edge that would
//!   not is dropped and counted ([`Graph::dropped`]), never kept.
//! - A programmatic (PDL) successor, one that started before its stream predecessor finished,
//!   gets two edges: its start follows the predecessor's start, its finish the predecessor's
//!   finish. Its own work is only the part after the predecessor finished.
//! - The graph is acyclic and carries a topological order ([`Graph::topo`]).

use crate::prelude::*;

use alloc::collections::BTreeMap;

use super::capture::{Capture, OpKind};
use super::graph::{Edge, EdgeKind, Graph, Node, NodeKind, f_of, s_of};

#[derive(Clone, Copy)]
enum Entry {
    Op(usize),
    Graph(u64),
}

struct Builder<'a> {
    c: &'a Capture,
    t: Vec<f64>,
    edges: Vec<Edge>,
    dropped: BTreeMap<EdgeKind, usize>,
    glast: BTreeMap<u64, usize>,
    gfirst: BTreeMap<u64, usize>,
    seq: BTreeMap<u64, Vec<(u64, i64, Entry)>>,
}

impl Builder<'_> {
    fn edge(&mut self, src: u32, dst: u32, kind: EdgeKind) {
        if self.t[src as usize] <= self.t[dst as usize] + 1e-9 {
            self.edges.push(Edge { src, dst, kind });
        } else {
            *self.dropped.entry(kind).or_default() += 1;
        }
    }

    fn api_ev(&self, j: usize) -> u32 {
        s_of(self.c.ops.len() + j)
    }

    fn resolve(&self, e: Entry, last: bool) -> usize {
        match e {
            Entry::Op(i) => i,
            Entry::Graph(c) if last => self.glast[&c],
            Entry::Graph(c) => self.gfirst[&c],
        }
    }

    /// 2026-10-23: The last operation enqueued on `stream` before correlation id `corr`.
    fn last_before(&self, stream: u64, corr: u64) -> Option<usize> {
        let l = self.seq.get(&stream)?;
        let k = l.partition_point(|x| x.0 < corr);
        (k > 0).then(|| self.resolve(l[k - 1].2, true))
    }

    /// 2026-10-23: The first operation enqueued on `stream` after correlation id `corr`.
    fn first_after(&self, stream: u64, corr: u64) -> Option<usize> {
        let l = self.seq.get(&stream)?;
        let k = l.partition_point(|x| x.0 <= corr);
        l.get(k).map(|x| self.resolve(x.2, false))
    }
}

/// 2026-10-23: Latest-finishing entry of `ends` (sorted by end) at or before `at` on a stream
/// other than `stream`.
fn latest_other(
    ends: &[(i64, usize)],
    ops: &[super::capture::Op],
    at: i64,
    stream: u64,
) -> Option<usize> {
    let mut k = ends.partition_point(|x| x.0 <= at);
    while k > 0 {
        k -= 1;
        if ops[ends[k].1].stream() != stream {
            return Some(ends[k].1);
        }
    }
    None
}

/// 2026-10-23: Build the event graph of `c`. The first node of a stream inside a graph launch
/// waited on the latest node to finish on another stream (a fork or a join, `Join`). A node
/// that starts more than `tau_ns` after its stream predecessor finished waited for SMs
/// (`Residency`, its measured gap), unless it started within `prog_ns` of a node on another
/// stream that was still running and finished before it: then it is that node's programmatic
/// dependent (`PdlStart` + `PdlWait`). Only a mid-stream node is tested for that (a stream's
/// first node launched beside a running one is a fork, not a dependent).
pub fn build(c: &Capture, tau_ns: i64, prog_ns: i64) -> metrale_error::Result<Graph> {
    let (n, m) = (c.ops.len(), c.api.len());
    let mut t = vec![0.0; 2 * (n + m)];
    for (i, o) in c.ops.iter().enumerate() {
        t[2 * i] = o.start() as f64 / 1e3;
        t[2 * i + 1] = o.end() as f64 / 1e3;
    }
    for (j, a) in c.api.iter().enumerate() {
        t[2 * (n + j)] = a.1 as f64 / 1e3;
        t[2 * (n + j) + 1] = a.2 as f64 / 1e3;
    }
    let mut graphs: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for (i, o) in c.ops.iter().enumerate() {
        if o.graph() {
            graphs.entry(o.corr()).or_default().push(i);
        }
    }
    let mut b = Builder {
        c,
        t,
        edges: Vec::new(),
        dropped: BTreeMap::new(),
        glast: BTreeMap::new(),
        gfirst: BTreeMap::new(),
        seq: BTreeMap::new(),
    };
    for (&corr, v) in &graphs {
        let first = v
            .iter()
            .copied()
            .min_by_key(|&i| (c.ops[i].start(), i))
            .unwrap_or(v[0]);
        let last = v
            .iter()
            .copied()
            .max_by_key(|&i| (c.ops[i].end(), i))
            .unwrap_or(v[0]);
        b.gfirst.insert(corr, first);
        b.glast.insert(corr, last);
        let s = c.ops[first].stream();
        b.seq
            .entry(s)
            .or_default()
            .push((corr, c.ops[first].start(), Entry::Graph(corr)));
    }
    for (i, o) in c.ops.iter().enumerate() {
        if !o.graph() {
            b.seq
                .entry(o.stream())
                .or_default()
                .push((o.corr(), o.start(), Entry::Op(i)));
        }
    }
    for l in b.seq.values_mut() {
        l.sort_by_key(|x| (x.0, x.1));
    }
    let api_of: BTreeMap<u64, usize> = c.api.iter().enumerate().map(|(j, a)| (a.3, j)).collect();
    for i in 0..n {
        b.edge(s_of(i), f_of(i), EdgeKind::Duration);
        if let Some(&j) = api_of.get(&c.ops[i].corr()) {
            b.edge(b.api_ev(j), s_of(i), EdgeKind::Launch);
        }
    }
    for j in 0..m {
        let s = b.api_ev(j);
        b.edge(s, s | 1, EdgeKind::ApiDuration);
        if j > 0 {
            b.edge(b.api_ev(j - 1) | 1, s, EdgeKind::Host);
        }
    }
    let seq_pairs: Vec<(Entry, Entry)> = b
        .seq
        .values()
        .flat_map(|l| l.windows(2).map(|w| (w[0].2, w[1].2)))
        .collect();
    for (x, y) in seq_pairs {
        let (src, dst) = (b.resolve(x, true), b.resolve(y, false));
        b.edge(f_of(src), s_of(dst), EdgeKind::Stream);
    }
    let mut gends: Vec<(i64, usize)> = c
        .ops
        .iter()
        .enumerate()
        .map(|(i, o)| (o.end(), i))
        .collect();
    gends.sort_unstable();
    for v in graphs.values() {
        let mut v = v.clone();
        v.sort_by_key(|&i| (c.ops[i].start(), c.ops[i].end(), i));
        let mut last: BTreeMap<u64, usize> = BTreeMap::new();
        let mut ends: Vec<(i64, usize)> = Vec::with_capacity(v.len());
        for (pos, &i) in v.iter().enumerate() {
            let o = &c.ops[i];
            // 2026-10-23: A programmatic dependent on another stream: launched within `prog_ns`
            // of a running node's start, finishing after it (a combine launched by the sweep it
            // waits on).
            let prog = || {
                v[..pos]
                    .iter()
                    .rev()
                    .take_while(|&&q| c.ops[q].start() >= o.start() - prog_ns)
                    .find(|&&q| {
                        let p = &c.ops[q];
                        p.stream() != o.stream() && p.end() > o.start() && p.end() <= o.end()
                    })
                    .copied()
            };
            match last.get(&o.stream()).copied() {
                Some(p) if o.start() < c.ops[p].end() => {
                    b.edge(s_of(p), s_of(i), EdgeKind::PdlStart);
                    if o.end() >= c.ops[p].end() {
                        b.edge(f_of(p), f_of(i), EdgeKind::PdlWait);
                    }
                }
                Some(p) if o.start() - c.ops[p].end() <= tau_ns => {
                    b.edge(f_of(p), s_of(i), EdgeKind::GraphStream);
                }
                Some(p) => match prog() {
                    Some(q) => {
                        b.edge(f_of(p), s_of(i), EdgeKind::GraphStream);
                        b.edge(s_of(q), s_of(i), EdgeKind::PdlStart);
                        b.edge(f_of(q), f_of(i), EdgeKind::PdlWait);
                    }
                    // 2026-10-23: Mid-stream and unexplained: the node waited for SMs (a
                    // residency wait), not for another stream's data; a fork or join enters a
                    // graph stream at its first node.
                    None => b.edge(f_of(p), s_of(i), EdgeKind::Residency),
                },
                None => {
                    let q = latest_other(&ends, &c.ops, o.start(), o.stream())
                        .or_else(|| latest_other(&gends, &c.ops, o.start(), o.stream()));
                    if let Some(q) = q {
                        b.edge(f_of(q), s_of(i), EdgeKind::Join);
                    }
                }
            }
            last.insert(o.stream(), i);
            let k = ends.partition_point(|x| *x < (o.end(), i));
            ends.insert(k, (o.end(), i));
        }
    }
    let mut records: BTreeMap<u64, Vec<(u64, u64)>> = BTreeMap::new();
    for r in &c.record {
        records.entry(r.2).or_default().push((r.0, r.1));
    }
    for l in records.values_mut() {
        l.sort_unstable();
    }
    let rec_source = |b: &Builder<'_>, ev: Option<u64>, corr: u64| -> Option<usize> {
        let l = records.get(&ev?)?;
        let k = l.partition_point(|x| x.0 < corr);
        (k > 0)
            .then(|| b.last_before(l[k - 1].1, l[k - 1].0))
            .flatten()
    };
    for s in &c.sync {
        let Some(&j) = api_of.get(&s.0) else { continue };
        let name = c.api[j].0.as_str();
        let fin = b.api_ev(j) | 1;
        if name.contains("StreamWaitEvent") {
            if let (Some(src), Some(dst)) = (rec_source(&b, s.3, s.0), b.first_after(s.2, s.0)) {
                b.edge(f_of(src), s_of(dst), EdgeKind::Event);
            }
        } else if name.contains("StreamSynchronize") {
            if let Some(src) = b.last_before(s.2, s.0) {
                b.edge(f_of(src), fin, EdgeKind::Sync);
            }
        } else if name.contains("EventSynchronize")
            && let Some(src) = rec_source(&b, s.3, s.0)
        {
            b.edge(f_of(src), fin, EdgeKind::Sync);
        }
    }
    for (i, o) in c.ops.iter().enumerate() {
        if o.0 == OpKind::Memcpy
            && let Some(&j) = api_of.get(&o.corr())
            && c.api[j].0.starts_with("cuMemcpyDtoH")
        {
            b.edge(f_of(i), b.api_ev(j) | 1, EdgeKind::Sync);
        }
    }
    let anchors = c
        .ops
        .iter()
        .enumerate()
        .filter(|(_, o)| o.name() == c.step_kernel && o.start() >= 0)
        .map(|(i, _)| s_of(i))
        .collect();
    let mut nodes: Vec<Node> = c
        .ops
        .iter()
        .map(|o| {
            let (ctas, threads, regs, smem) = o.launch();
            Node {
                kind: if o.0 == OpKind::Kernel {
                    NodeKind::Kernel
                } else {
                    NodeKind::Copy
                },
                group: Some(o.group()),
                ctas,
                threads,
                regs,
                smem,
                stream: o.stream(),
            }
        })
        .collect();
    nodes.extend(c.api.iter().map(|_| Node::api()));
    Graph::assemble(&c.source, b.t, b.edges, nodes, anchors, b.dropped)
}
