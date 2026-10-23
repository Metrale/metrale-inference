// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The event graph's types: nodes (GPU operations, API calls, and operations a
//! rewrite adds), their start/finish events, typed edges, and the CSR/topological order the
//! evaluators walk.
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - Node `v` has events `2v` (start) and `2v + 1` (finish).
//! - [`Graph::assemble`] refuses a cycle ([`ErrorCode::CriticalCaptureInvalid`] for a capture,
//!   [`ErrorCode::WhatIfInvalid`] for a rewrite), so every graph carries a topological order.

use crate::prelude::*;

use alloc::collections::BTreeMap;

use metrale_error::{ErrorCode, error};

/// 2026-10-23: Why one event waits on another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EdgeKind {
    /// 2026-10-23: An operation's own work (`S -> F`).
    Duration,
    /// 2026-10-23: An API call's own host time (`S -> F`).
    ApiDuration,
    /// 2026-10-23: Host work between two API calls of the launching thread (`F -> S`).
    Host,
    /// 2026-10-23: An operation cannot start before its launch call started (`S api -> S op`).
    Launch,
    /// 2026-10-23: Stream order outside a graph (`F -> S`).
    Stream,
    /// 2026-10-23: Stream order inside a graph launch (`F -> S`).
    GraphStream,
    /// 2026-10-23: A programmatic successor starts after its predecessor started (`S -> S`).
    PdlStart,
    /// 2026-10-23: A programmatic successor finishes after its predecessor finished (`F -> F`).
    PdlWait,
    /// 2026-10-23: Inferred cross-stream fork or join (`F -> S`).
    Join,
    /// 2026-10-23: Stream order inside a graph launch with a wait for SMs in it (`F -> S`): the
    /// max-plus keeps the measured wait, the simulation regenerates it from residency.
    Residency,
    /// 2026-10-23: A stream wait on a recorded event (`F -> S`).
    Event,
    /// 2026-10-23: A host synchronization returns after the GPU work it waits on (`F -> F api`).
    Sync,
}

impl EdgeKind {
    /// 2026-10-23: The spelling in reports.
    pub fn name(self) -> &'static str {
        match self {
            EdgeKind::Duration => "duration",
            EdgeKind::ApiDuration => "api",
            EdgeKind::Host => "host",
            EdgeKind::Launch => "launch",
            EdgeKind::Stream => "stream",
            EdgeKind::GraphStream => "graph-stream",
            EdgeKind::PdlStart => "pdl-start",
            EdgeKind::PdlWait => "pdl-wait",
            EdgeKind::Join => "join",
            EdgeKind::Residency => "residency",
            EdgeKind::Event => "event",
            EdgeKind::Sync => "sync",
        }
    }

    /// 2026-10-23: A launch latency the model states as one constant per kind (the rest are
    /// work or host time, taken in context).
    pub fn is_latency(self) -> bool {
        matches!(
            self,
            EdgeKind::Launch
                | EdgeKind::Stream
                | EdgeKind::GraphStream
                | EdgeKind::Join
                | EdgeKind::Event
                | EdgeKind::Sync
        )
    }
}

/// 2026-10-23: One edge between two events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edge {
    /// 2026-10-23: Source event.
    pub src: u32,
    /// 2026-10-23: Target event.
    pub dst: u32,
    /// 2026-10-23: Why.
    pub kind: EdgeKind,
}

/// 2026-10-23: What a node is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    /// 2026-10-23: A kernel: holds SMs while resident, may read DRAM.
    Kernel,
    /// 2026-10-23: A copy or memset (copy engine: no SMs).
    Copy,
    /// 2026-10-23: An API call of the launching thread.
    Api,
}

/// 2026-10-23: One node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    /// 2026-10-23: Kind.
    pub kind: NodeKind,
    /// 2026-10-23: Launch group (index into the capture's groups) of an operation.
    pub group: Option<u32>,
    /// 2026-10-23: CTAs.
    pub ctas: u32,
    /// 2026-10-23: Threads per CTA.
    pub threads: u32,
    /// 2026-10-23: Registers per thread.
    pub regs: u32,
    /// 2026-10-23: Shared memory per CTA, bytes.
    pub smem: u32,
    /// 2026-10-23: Stream (0 for an API call).
    pub stream: u64,
}

impl Node {
    /// 2026-10-23: An API call.
    pub fn api() -> Node {
        Node {
            kind: NodeKind::Api,
            group: None,
            ctas: 0,
            threads: 0,
            regs: 0,
            smem: 0,
            stream: 0,
        }
    }
}

/// 2026-10-23: The event graph.
#[derive(Debug, Clone)]
pub struct Graph {
    /// 2026-10-23: The capture's fingerprint (named in errors).
    pub source: String,
    /// 2026-10-23: Nodes.
    pub nodes: Vec<Node>,
    /// 2026-10-23: Measured time of every event, microseconds from the first anchor (a node a
    /// rewrite added: the time of the event it follows).
    pub t: Vec<f64>,
    /// 2026-10-23: Edges.
    pub edges: Vec<Edge>,
    /// 2026-10-23: Per event, the indices of its incoming edges (`in_off[v]..in_off[v+1]`).
    pub in_off: Vec<u32>,
    /// 2026-10-23: Incoming edge indices, grouped by target.
    pub in_edges: Vec<u32>,
    /// 2026-10-23: Per event, the indices of its outgoing edges (`out_off[v]..out_off[v+1]`).
    pub out_off: Vec<u32>,
    /// 2026-10-23: Outgoing edge indices, grouped by source.
    pub out_edges: Vec<u32>,
    /// 2026-10-23: Events in a topological order.
    pub topo: Vec<u32>,
    /// 2026-10-23: The step kernel's start events in the window, in order.
    pub anchors: Vec<u32>,
    /// 2026-10-23: Edges dropped because they ran backwards in time, per kind.
    pub dropped: BTreeMap<EdgeKind, usize>,
}

/// 2026-10-23: Start event of node `i`.
pub fn s_of(i: usize) -> u32 {
    u32::try_from(2 * i).unwrap_or(u32::MAX)
}

/// 2026-10-23: Finish event of node `i`.
pub fn f_of(i: usize) -> u32 {
    s_of(i) | 1
}

/// 2026-10-23: An event or edge index as stored: [`Graph::assemble`] refuses a graph whose
/// events or edges do not fit `u32`, so the saturation is unreachable on an assembled graph.
pub fn ix(k: usize) -> u32 {
    u32::try_from(k).unwrap_or(u32::MAX)
}

/// 2026-10-23: The node of event `v`.
pub fn node_of(v: u32) -> usize {
    (v / 2) as usize
}

impl Graph {
    /// 2026-10-23: Index the edges and order the events; refuse a cycle.
    pub fn assemble(
        source: &str,
        t: Vec<f64>,
        edges: Vec<Edge>,
        nodes: Vec<Node>,
        anchors: Vec<u32>,
        dropped: BTreeMap<EdgeKind, usize>,
    ) -> metrale_error::Result<Graph> {
        let nv = t.len();
        if nv != 2 * nodes.len()
            || u32::try_from(edges.len()).is_err()
            || u32::try_from(nv).is_err()
        {
            return Err(error!(
                ErrorCode::CriticalCaptureInvalid,
                source.to_string(),
                format!(
                    "{nv} events for {} nodes, {} edges",
                    nodes.len(),
                    edges.len()
                )
            ));
        }
        let csr = |key: &dyn Fn(&Edge) -> usize| {
            let mut off = vec![0u32; nv + 1];
            for e in &edges {
                off[key(e) + 1] += 1;
            }
            for v in 0..nv {
                off[v + 1] += off[v];
            }
            let mut fill = off.clone();
            let mut idx = vec![0u32; edges.len()];
            for (k, e) in edges.iter().enumerate() {
                idx[fill[key(e)] as usize] = ix(k);
                fill[key(e)] += 1;
            }
            (off, idx)
        };
        let (in_off, in_edges) = csr(&|e: &Edge| e.dst as usize);
        let (out_off, out_edges) = csr(&|e: &Edge| e.src as usize);
        let mut deg: Vec<u32> = (0..nv).map(|v| in_off[v + 1] - in_off[v]).collect();
        let mut topo: Vec<u32> = (0..nv).filter(|&v| deg[v] == 0).map(ix).collect();
        let mut head = 0;
        while head < topo.len() {
            let v = topo[head] as usize;
            head += 1;
            for &k in &out_edges[out_off[v] as usize..out_off[v + 1] as usize] {
                let d = edges[k as usize].dst as usize;
                deg[d] -= 1;
                if deg[d] == 0 {
                    topo.push(ix(d));
                }
            }
        }
        if topo.len() != nv {
            return Err(error!(
                ErrorCode::WhatIfInvalid,
                source.to_string(),
                format!(
                    "the event graph has a cycle ({} of {nv} events ordered)",
                    topo.len()
                )
            ));
        }
        Ok(Graph {
            source: source.to_string(),
            nodes,
            t,
            edges,
            in_off,
            in_edges,
            out_off,
            out_edges,
            topo,
            anchors,
            dropped,
        })
    }

    /// 2026-10-23: The same graph with other edges and nodes added at the end (a rewrite).
    pub fn rewired(
        &self,
        edges: Vec<Edge>,
        extra: Vec<(Node, f64, f64)>,
    ) -> metrale_error::Result<Graph> {
        let mut nodes = self.nodes.clone();
        let mut t = self.t.clone();
        for (n, s, f) in extra {
            nodes.push(n);
            t.push(s);
            t.push(f);
        }
        Graph::assemble(
            &self.source,
            t,
            edges,
            nodes,
            self.anchors.clone(),
            self.dropped.clone(),
        )
    }

    /// 2026-10-23: Incoming edge indices of event `v`.
    pub fn incoming(&self, v: u32) -> &[u32] {
        let v = v as usize;
        &self.in_edges[self.in_off[v] as usize..self.in_off[v + 1] as usize]
    }

    /// 2026-10-23: Outgoing edge indices of event `v`.
    pub fn outgoing(&self, v: u32) -> &[u32] {
        let v = v as usize;
        &self.out_edges[self.out_off[v] as usize..self.out_off[v + 1] as usize]
    }

    /// 2026-10-23: The measured weight of edge `k`: target time minus source time.
    pub fn gap(&self, k: usize) -> f64 {
        let e = self.edges[k];
        self.t[e.dst as usize] - self.t[e.src as usize]
    }
}
