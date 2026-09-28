// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The buffer planner: an arena offset for every materialised edge of a plan,
//! sized once for the widest padded row count, so no buffer is reallocated after boot (the
//! CUDA-graph constraint: captured pointers must stay valid).
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - Two edges whose live ranges overlap never share a byte.
//! - An edge is live from the group that writes it to the last group that reads it; an edge
//!   read or written outside the plan (its inputs from another section, and the declared
//!   outputs) is live for the whole plan.
//! - Offsets are aligned to [`ALIGN`] bytes.

use std::collections::BTreeMap;

use crate::fuser::{EdgeState, FusionPlan};
use crate::ir::{Circuit, EdgeIdx};

/// 2026-09-28: Offset alignment, in bytes.
pub const ALIGN: u64 = 256;

/// 2026-09-28: One placed edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slot {
    /// 2026-09-28: The edge.
    pub edge: EdgeIdx,
    /// 2026-09-28: Byte offset in the arena.
    pub offset: u64,
    /// 2026-09-28: Bytes at the sized row count.
    pub bytes: u64,
    /// 2026-09-28: First group index it is live in.
    pub start: usize,
    /// 2026-09-28: Last group index it is live in.
    pub end: usize,
}

/// 2026-09-28: The arena layout of one plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferPlan {
    /// 2026-09-28: Placed edges, by edge index.
    pub slots: Vec<Slot>,
    /// 2026-09-28: Arena size in bytes.
    pub arena_bytes: u64,
    /// 2026-09-28: Sum of the materialised edges' sizes, before reuse.
    pub materialized_bytes: u64,
}

/// 2026-09-28: Why a plan could not be laid out.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PlanError {
    /// 2026-09-28: A row or byte count overflowed, or a dim is not a multiple of a scale group.
    #[error("edge `{0}` has no byte size at the sized row count")]
    Size(String),
}

/// 2026-09-28: Live range of every materialised edge in `plan`, keyed by edge.
pub fn live_ranges(circuit: &Circuit, plan: &FusionPlan) -> BTreeMap<EdgeIdx, (usize, usize)> {
    let mut group_of = vec![None; circuit.nodes.len()];
    for (g, grp) in plan.groups.iter().enumerate() {
        for &n in &grp.nodes {
            group_of[n] = Some(g);
        }
    }
    let last = plan.groups.len().saturating_sub(1);
    let mut out = BTreeMap::new();
    for (e, edge) in circuit.edges.iter().enumerate() {
        if plan.edge_states[e] != Some(EdgeState::Materialized) {
            continue;
        }
        let start = edge.producer.and_then(|p| group_of[p]).unwrap_or(0);
        let end = if edge.is_output {
            last
        } else {
            edge.consumers
                .iter()
                .filter_map(|&c| group_of[c])
                .max()
                .unwrap_or(last)
                .max(start)
        };
        out.insert(e, (start, end));
    }
    out
}

/// 2026-09-28: Place every materialised edge of `plan`, sized for `max_rows` rows.
///
/// First fit over edges sorted by (start, larger first, edge index): each edge takes the
/// lowest aligned offset that overlaps no placed edge whose live range meets its own.
pub fn plan_buffers(
    circuit: &Circuit,
    plan: &FusionPlan,
    max_rows: u64,
) -> Result<BufferPlan, PlanError> {
    let mut dims = circuit.dims.clone();
    dims.insert("n".into(), max_rows);
    let mut items = Vec::new();
    let mut materialized_bytes = 0u64;
    for (e, (start, end)) in live_ranges(circuit, plan) {
        let edge = &circuit.edges[e];
        let size_err = || PlanError::Size(edge.id.clone());
        let rows = edge.rows.eval(&dims).map_err(|_| size_err())?;
        let bytes = plan.edge_formats[e]
            .bytes(rows, edge.dim_value)
            .ok_or_else(size_err)?;
        materialized_bytes = materialized_bytes.checked_add(bytes).ok_or_else(size_err)?;
        items.push((start, end, bytes, e));
    }
    items.sort_by(|a, b| a.0.cmp(&b.0).then(b.2.cmp(&a.2)).then(a.3.cmp(&b.3)));
    let mut slots: Vec<Slot> = Vec::with_capacity(items.len());
    let mut arena_bytes = 0u64;
    for (start, end, bytes, edge) in items {
        let mut busy: Vec<(u64, u64)> = slots
            .iter()
            .filter(|s| s.start <= end && start <= s.end)
            .map(|s| (s.offset, s.offset + s.bytes))
            .collect();
        busy.sort_unstable();
        let mut offset = 0u64;
        for (lo, hi) in busy {
            if offset + bytes <= lo {
                break;
            }
            offset = offset.max(hi.div_ceil(ALIGN) * ALIGN);
        }
        arena_bytes = arena_bytes.max(offset + bytes);
        slots.push(Slot {
            edge,
            offset,
            bytes,
            start,
            end,
        });
    }
    slots.sort_by_key(|s| s.edge);
    Ok(BufferPlan {
        slots,
        arena_bytes,
        materialized_bytes,
    })
}

#[cfg(test)]
#[path = "planner_tests.rs"]
mod planner_tests;
