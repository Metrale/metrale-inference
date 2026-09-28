// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The buffer planner: an arena offset for every materialised edge of a plan,
//! sized once for the widest padded row count, so no buffer is reallocated after boot (the
//! CUDA-graph constraint: captured pointers must stay valid).
//!
//! An executor may add a [`Layout`]: edges it binds to buffers of its own, outputs a kernel
//! writes in place over one of its inputs, and edges a kernel writes back to back.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - Two edges whose live ranges overlap never share a byte, unless a [`Layout`] alias joins
//!   them; the edges of one alias class share their bytes by construction.
//! - An edge is live from the group that writes it to the last group that reads it; an edge
//!   read or written outside the plan (its inputs from another section, and the declared
//!   outputs) is live for the whole plan. An alias class is live over the union of its edges.
//! - The edges of a pack are placed back to back, in pack order, with no padding between.
//! - Offsets of unpacked storage are aligned to [`ALIGN`] bytes; a pack's first edge is.

use std::collections::{BTreeMap, BTreeSet};

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
    /// 2026-09-28: A [`Layout`] constraint the plan cannot honour.
    #[error("layout: {0}")]
    Layout(String),
}

/// 2026-09-28: Storage constraints an executor puts on a plan's materialised edges.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Layout {
    /// 2026-09-28: Edges bound to the executor's own buffers; the planner places none of them.
    pub external: BTreeSet<EdgeIdx>,
    /// 2026-09-28: `(output, input)`: a kernel writes `output` in place over `input`, so the
    /// two share storage. They must have the same size.
    pub aliases: Vec<(EdgeIdx, EdgeIdx)>,
    /// 2026-09-28: Edges a kernel writes back to back, in this order, from one pointer.
    pub packs: Vec<Vec<EdgeIdx>>,
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

/// 2026-09-28: Place every materialised edge of `plan`, sized for `max_rows` rows, with no
/// layout constraints.
pub fn plan_buffers(
    circuit: &Circuit,
    plan: &FusionPlan,
    max_rows: u64,
) -> Result<BufferPlan, PlanError> {
    plan_buffers_with(circuit, plan, max_rows, &Layout::default())
}

/// 2026-09-28: One unit of placement: alias classes laid back to back.
struct Item {
    /// 2026-09-28: `(class root, offset within the item)`.
    classes: Vec<(EdgeIdx, u64)>,
    bytes: u64,
    start: usize,
    end: usize,
    /// 2026-09-28: Lowest edge index in the item: the deterministic tie-break.
    key: EdgeIdx,
}

fn root(parent: &mut BTreeMap<EdgeIdx, EdgeIdx>, e: EdgeIdx) -> EdgeIdx {
    let mut r = e;
    while parent[&r] != r {
        r = parent[&r];
    }
    parent.insert(e, r);
    r
}

/// 2026-09-28: Place every materialised edge of `plan` that `layout` does not bind, sized for
/// `max_rows` rows.
///
/// Aliased edges form one class; packs lay classes back to back into one item; every other
/// class is an item of its own. Items are placed first fit, sorted by (start, larger first,
/// lowest edge): each takes the lowest aligned offset that overlaps no placed item whose live
/// range meets its own.
pub fn plan_buffers_with(
    circuit: &Circuit,
    plan: &FusionPlan,
    max_rows: u64,
    layout: &Layout,
) -> Result<BufferPlan, PlanError> {
    let mut dims = circuit.dims.clone();
    dims.insert("n".into(), max_rows);
    let ranges = live_ranges(circuit, plan);
    let id = |e: EdgeIdx| circuit.edges.get(e).map_or("?", |x| x.id.as_str());
    let layout_err = |msg: String| PlanError::Layout(msg);
    for &e in layout
        .external
        .iter()
        .chain(layout.aliases.iter().flat_map(|(a, b)| [a, b]))
        .chain(layout.packs.iter().flatten())
    {
        if !ranges.contains_key(&e) {
            return Err(layout_err(format!("edge `{}` is not materialised", id(e))));
        }
    }
    let mut size = BTreeMap::new();
    let mut materialized_bytes = 0u64;
    for &e in ranges.keys() {
        let edge = &circuit.edges[e];
        let size_err = || PlanError::Size(edge.id.clone());
        let rows = edge.rows.eval(&dims).map_err(|_| size_err())?;
        let bytes = plan.edge_formats[e]
            .bytes(rows, edge.dim_value)
            .ok_or_else(size_err)?;
        materialized_bytes = materialized_bytes.checked_add(bytes).ok_or_else(size_err)?;
        if !layout.external.contains(&e) {
            size.insert(e, bytes);
        }
    }
    let mut parent: BTreeMap<EdgeIdx, EdgeIdx> = size.keys().map(|&e| (e, e)).collect();
    for &(out, inp) in &layout.aliases {
        if layout.external.contains(&out) || layout.external.contains(&inp) {
            return Err(layout_err(format!(
                "alias `{}` over `{}` names an external edge",
                id(out),
                id(inp)
            )));
        }
        if size[&out] != size[&inp] {
            return Err(layout_err(format!(
                "alias `{}` ({} B) over `{}` ({} B): sizes differ",
                id(out),
                size[&out],
                id(inp),
                size[&inp]
            )));
        }
        let (a, b) = (root(&mut parent, out), root(&mut parent, inp));
        parent.insert(a.max(b), a.min(b));
    }
    let edges: Vec<EdgeIdx> = size.keys().copied().collect();
    let mut class_range: BTreeMap<EdgeIdx, (usize, usize, EdgeIdx)> = BTreeMap::new();
    for &e in &edges {
        let r = root(&mut parent, e);
        let (s, t) = ranges[&e];
        let c = class_range.entry(r).or_insert((s, t, e));
        *c = (c.0.min(s), c.1.max(t), c.2.min(e));
    }
    let mut packed = BTreeSet::new();
    let mut items = Vec::new();
    for pack in &layout.packs {
        let mut item = Item {
            classes: Vec::new(),
            bytes: 0,
            start: usize::MAX,
            end: 0,
            key: EdgeIdx::MAX,
        };
        for &e in pack {
            if layout.external.contains(&e) {
                return Err(layout_err(format!("pack names external edge `{}`", id(e))));
            }
            let r = root(&mut parent, e);
            if !packed.insert(r) {
                return Err(layout_err(format!("edge `{}` is packed twice", id(e))));
            }
            let (s, t, k) = class_range[&r];
            item.classes.push((r, item.bytes));
            item.bytes += size[&e];
            item.start = item.start.min(s);
            item.end = item.end.max(t);
            item.key = item.key.min(k);
        }
        items.push(item);
    }
    for (&r, &(s, t, k)) in &class_range {
        if !packed.contains(&r) {
            items.push(Item {
                classes: vec![(r, 0)],
                bytes: size[&r],
                start: s,
                end: t,
                key: k,
            });
        }
    }
    items.sort_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then(b.bytes.cmp(&a.bytes))
            .then(a.key.cmp(&b.key))
    });
    let mut placed: Vec<(u64, u64, usize, usize)> = Vec::with_capacity(items.len());
    let mut class_at: BTreeMap<EdgeIdx, (u64, usize, usize)> = BTreeMap::new();
    let mut arena_bytes = 0u64;
    for item in &items {
        let mut busy: Vec<(u64, u64)> = placed
            .iter()
            .filter(|p| p.2 <= item.end && item.start <= p.3)
            .map(|p| (p.0, p.1))
            .collect();
        busy.sort_unstable();
        let mut offset = 0u64;
        for (lo, hi) in busy {
            if offset + item.bytes <= lo {
                break;
            }
            offset = offset.max(hi.div_ceil(ALIGN) * ALIGN);
        }
        arena_bytes = arena_bytes.max(offset + item.bytes);
        placed.push((offset, offset + item.bytes, item.start, item.end));
        for &(r, within) in &item.classes {
            class_at.insert(r, (offset + within, item.start, item.end));
        }
    }
    let mut slots = Vec::with_capacity(edges.len());
    for e in edges {
        let (offset, start, end) = class_at[&root(&mut parent, e)];
        slots.push(Slot {
            edge: e,
            offset,
            bytes: size[&e],
            start,
            end,
        });
    }
    Ok(BufferPlan {
        slots,
        arena_bytes,
        materialized_bytes,
    })
}

#[cfg(test)]
#[path = "planner_tests.rs"]
mod planner_tests;
