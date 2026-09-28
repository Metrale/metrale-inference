// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The rows of one diagram: each node becomes a row, except that a MoE block's
//! routed-expert and shared-expert paths are paired side by side between a fan-out and a
//! fan-in, and consecutive rows of one multi-node group share a frame.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - Every node of the segment appears in exactly one row.
//! - A frame covers a contiguous run of rows all of whose nodes belong to one group; a group
//!   whose rows are not contiguous gets one frame per run.

use std::collections::VecDeque;

use crate::fuser::FusionPlan;
use crate::ir::{Circuit, LinearRole, NodeIdx, OpKind};

/// 2026-09-28: One row of a diagram.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Row {
    /// 2026-09-28: One node across the full width.
    Single(NodeIdx),
    /// 2026-09-28: A routed-expert node and a shared-expert node side by side.
    Pair(Option<NodeIdx>, Option<NodeIdx>),
    /// 2026-09-28: The split into the routed and shared paths.
    FanOut,
    /// 2026-09-28: The join back into one path.
    FanIn,
}

impl Row {
    pub(super) fn nodes(&self) -> Vec<NodeIdx> {
        match self {
            Row::Single(n) => vec![*n],
            Row::Pair(a, b) => a.iter().chain(b.iter()).copied().collect(),
            Row::FanOut | Row::FanIn => Vec::new(),
        }
    }
}

fn is_shared(c: &Circuit, n: NodeIdx) -> bool {
    match c.nodes[n].op {
        OpKind::Linear(
            LinearRole::SharedGateUp | LinearRole::SharedDown | LinearRole::SharedGate,
        ) => true,
        OpKind::SiluMul => c.nodes[n].inputs.iter().any(|&e| {
            c.edges[e]
                .producer
                .is_some_and(|p| c.nodes[p].op == OpKind::Linear(LinearRole::SharedGateUp))
        }),
        _ => false,
    }
}

fn is_routed(c: &Circuit, n: NodeIdx) -> bool {
    match c.nodes[n].op {
        OpKind::ExpertGateUp | OpKind::ExpertDown => true,
        OpKind::SiluMul => c.nodes[n].inputs.iter().any(|&e| {
            c.edges[e]
                .producer
                .is_some_and(|p| c.nodes[p].op == OpKind::ExpertGateUp)
        }),
        _ => false,
    }
}

/// 2026-09-28: Rows for `nodes` (in execution order).
pub(super) fn rows(c: &Circuit, nodes: &[NodeIdx]) -> Vec<Row> {
    let mut shared: VecDeque<NodeIdx> =
        nodes.iter().copied().filter(|&n| is_shared(c, n)).collect();
    let mut out = Vec::new();
    let mut in_fan = false;
    for &n in nodes {
        if is_shared(c, n) {
            continue;
        }
        if is_routed(c, n) {
            if !in_fan {
                out.push(Row::FanOut);
                in_fan = true;
            }
            out.push(Row::Pair(Some(n), shared.pop_front()));
            continue;
        }
        if in_fan {
            while let Some(s) = shared.pop_front() {
                out.push(Row::Pair(None, Some(s)));
            }
            out.push(Row::FanIn);
            in_fan = false;
        }
        out.push(Row::Single(n));
    }
    // 2026-09-28: A shared-expert node with no routed path beside it stays visible.
    for s in shared {
        out.push(Row::Single(s));
    }
    out
}

/// 2026-09-28: The multi-node group a row belongs to, if all its nodes share one.
pub(super) fn row_group(row: &Row, plan: &FusionPlan, group_of: &[Option<usize>]) -> Option<usize> {
    let nodes = row.nodes();
    let g = group_of[*nodes.first()?]?;
    let all = nodes.iter().all(|&n| group_of[n] == Some(g));
    (all && plan.groups[g].nodes.len() > 1).then_some(g)
}

/// 2026-09-28: Frame assignment per row: `Some(group)` inside that group's frame. Fan rows
/// take the frame of the rows around them when both sides share it.
pub(super) fn frames(
    rows: &[Row],
    plan: &FusionPlan,
    group_of: &[Option<usize>],
) -> Vec<Option<usize>> {
    let mut f: Vec<Option<usize>> = rows.iter().map(|r| row_group(r, plan, group_of)).collect();
    for i in 0..rows.len() {
        if matches!(rows[i], Row::FanOut | Row::FanIn) {
            let before = i.checked_sub(1).and_then(|j| f[j]);
            let after = f.get(i + 1).copied().flatten();
            f[i] = if before.is_some() && before == after {
                before
            } else {
                None
            };
        }
    }
    f
}
