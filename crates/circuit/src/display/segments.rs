// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The segments a plan is drawn in: one per layer, one per run of blocks outside the
//! layers, and one per layer boundary a fused group crosses (the cross-layer residual add and
//! input norm), so that group is drawn once, in one frame spanning both layers.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - Every node of the plan's section is in exactly one segment, in execution order.
//! - A boundary segment holds exactly the nodes of one group whose nodes sit in two layers; the
//!   layer segments on either side no longer hold them.
//! - A segment's `enter` is the first edge its first node reads from outside it, and `exit` the
//!   first output of its last node that is read outside it.

use std::collections::BTreeSet;

use crate::fuser::{FusionPlan, section_of};
use crate::ir::{Circuit, EdgeIdx, NodeIdx};

/// 2026-09-28: What a segment is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SegKind {
    /// 2026-09-28: One decoder layer.
    Layer(usize),
    /// 2026-09-28: A fused group crossing from layer `.0` into layer `.1`.
    Boundary(usize, usize),
    /// 2026-09-28: Blocks outside the layers.
    Other,
}

/// 2026-09-28: A run of nodes drawn as one diagram.
pub(super) struct Segment {
    pub title: String,
    pub kind: SegKind,
    pub nodes: Vec<NodeIdx>,
    pub enter: Option<(EdgeIdx, String)>,
    pub exit: Option<(EdgeIdx, String)>,
}

impl Segment {
    /// 2026-09-28: The layer a summary groups this segment under: its own, or for a boundary
    /// the layer it enters.
    pub fn key_layer(&self) -> Option<usize> {
        match self.kind {
            SegKind::Layer(l) | SegKind::Boundary(_, l) => Some(l),
            SegKind::Other => None,
        }
    }

    /// 2026-09-28: Whether `--layer n` shows this segment: layer n and the boundaries on
    /// either side of it.
    pub fn touches(&self, n: usize) -> bool {
        match self.kind {
            SegKind::Layer(l) => l == n,
            SegKind::Boundary(a, b) => a == n || b == n,
            SegKind::Other => false,
        }
    }
}

/// 2026-09-28: The segments of `plan`, in execution order.
pub(super) fn segments(c: &Circuit, plan: &FusionPlan) -> Vec<Segment> {
    let section = section_of(plan.mode);
    let mut base: Vec<Segment> = Vec::new();
    for b in c.blocks.iter().filter(|b| b.section == section) {
        let kind = b.layer.map_or(SegKind::Other, SegKind::Layer);
        match base.last_mut() {
            Some(s) if s.kind == kind => {
                if kind == SegKind::Other {
                    s.title = format!("{} / {}", s.title, b.template);
                }
                s.nodes.extend(b.first..b.end);
            }
            _ => base.push(Segment {
                title: b.template.clone(),
                kind,
                nodes: (b.first..b.end).collect(),
                enter: None,
                exit: None,
            }),
        }
    }
    let layer_of = |n: NodeIdx| c.nodes[n].layer;
    let mut crossing: Vec<(usize, usize, Vec<NodeIdx>)> = Vec::new();
    let mut taken = BTreeSet::new();
    for grp in &plan.groups {
        let layers: BTreeSet<Option<usize>> = grp.nodes.iter().map(|&n| layer_of(n)).collect();
        if let (2, Some(Some(a)), Some(Some(b))) = (
            layers.len(),
            layers.first().copied(),
            layers.last().copied(),
        ) {
            taken.extend(grp.nodes.iter().copied());
            let mut nodes = grp.nodes.clone();
            nodes.sort_unstable();
            crossing.push((a, b, nodes));
        }
    }
    let mut out = Vec::new();
    for s in base {
        let from = match s.kind {
            SegKind::Layer(l) => Some(l),
            _ => None,
        };
        let nodes: Vec<NodeIdx> = s
            .nodes
            .iter()
            .copied()
            .filter(|n| !taken.contains(n))
            .collect();
        if !nodes.is_empty() {
            out.push(Segment { nodes, ..s });
        }
        for (a, b, nodes) in crossing.iter().filter(|(a, _, _)| Some(*a) == from) {
            out.push(Segment {
                title: "layer boundary".into(),
                kind: SegKind::Boundary(*a, *b),
                nodes: nodes.clone(),
                enter: None,
                exit: None,
            });
        }
    }
    let streams: BTreeSet<EdgeIdx> = c
        .blocks
        .iter()
        .flat_map(|b| b.stream_in.into_iter().chain(b.stream_out))
        .collect();
    for i in 0..out.len() {
        let inside: BTreeSet<NodeIdx> = out[i].nodes.iter().copied().collect();
        let seg_of = |n: NodeIdx| out.iter().position(|s| s.nodes.contains(&n));
        let first = out[i].nodes[0];
        let last = *out[i].nodes.last().unwrap_or(&first);
        // 2026-09-28: A boundary shows the edge the layer before hands it (the FFN output);
        // the residual stream it also reads is marked on the add itself.
        let outside = |e: &EdgeIdx| c.edges[*e].producer.is_none_or(|p| !inside.contains(&p));
        let inputs = &c.nodes[first].inputs;
        let enter = match out[i].kind {
            SegKind::Boundary(..) => inputs
                .iter()
                .copied()
                .find(|e| outside(e) && !streams.contains(e))
                .or_else(|| inputs.iter().copied().find(outside)),
            _ => inputs.iter().copied().find(outside),
        };
        let exit = c.nodes[last]
            .outputs
            .iter()
            .copied()
            .find(|&e| c.edges[e].consumers.iter().any(|x| !inside.contains(x)));
        let kind = out[i].kind;
        let boundary =
            |o: Option<usize>| o.is_some_and(|o| matches!(out[o].kind, SegKind::Boundary(..)));
        let enter = enter.map(|e| {
            let other = c.edges[e].producer.and_then(seg_of);
            let label = if let SegKind::Boundary(..) = kind {
                "from the previous layer".to_string()
            } else if streams.contains(&e) {
                "stream in".to_string()
            } else if boundary(other) {
                "from the layer boundary".to_string()
            } else {
                String::new()
            };
            (e, label)
        });
        let exit = exit.map(|e| {
            let other = c.edges[e]
                .consumers
                .iter()
                .find(|x| !inside.contains(x))
                .and_then(|&x| seg_of(x));
            let label = match kind {
                _ if streams.contains(&e) => "stream out".to_string(),
                SegKind::Boundary(..) => "into the next layer".to_string(),
                _ if boundary(other) => "into the layer boundary".to_string(),
                _ => String::new(),
            };
            (e, label)
        });
        out[i].enter = enter;
        out[i].exit = exit;
    }
    out
}
