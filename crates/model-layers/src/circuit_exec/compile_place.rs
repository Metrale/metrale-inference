// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Where a plan's edges and launches go, split from `compile.rs`: the storage
//! constraints of a plan ([`layout`]), the model buffer of an external edge
//! ([`external_buffer`]), a placement of every edge ([`Placement`]) and the segments of a
//! program ([`segments`]).
//!
//! Owner: model-layers circuit executor.
//! Invariants: see `compile.rs`.

use anyhow::{Context, Result, bail};
use metrale_circuit::planner::Layout;
use metrale_circuit::{Circuit, FusionPlan, Group, Mode};
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::emitters::emitter;
use super::super::fixed::Fixed;
use super::super::program::{Segment, SegmentOf};
use super::GroupRef;

/// 2026-10-03: Where each edge of a plan lives: its buffer and, when not packed contiguously,
/// its row stride in bytes. `None` for an edge the plan does not materialise.
pub struct Placement {
    pub ptrs: Vec<Option<DevicePtr>>,
    pub strides: Vec<Option<u64>>,
}

/// 2026-09-28: The storage constraints of `plan`.
pub fn layout(circuit: &Circuit, plan: &FusionPlan) -> Result<Layout> {
    let mut layout = Layout::default();
    let materialized =
        |e: usize| plan.edge_states[e] == Some(metrale_circuit::EdgeState::Materialized);
    for b in &circuit.blocks {
        for e in b.stream_in.into_iter().chain(b.stream_out) {
            if materialized(e) {
                layout.external.insert(e);
            }
        }
    }
    // 2026-09-30: The declared outputs (`Edge::is_output`); `external_buffer` places each by
    // the model buffer it binds to and refuses one that binds none.
    for (e, edge) in circuit.edges.iter().enumerate() {
        if edge.is_output && materialized(e) {
            layout.external.insert(e);
        }
    }
    for (index, group) in plan.groups.iter().enumerate() {
        let g = GroupRef {
            circuit,
            group,
            index,
        };
        emitter(&group.emitter)?.constrain(&g, &mut layout)?;
    }
    Ok(layout)
}

/// 2026-09-29: The model buffer an external edge is: a stream edge is `hidden`; otherwise
/// (2026-09-30) the buffer its declared output binds (`Edge::binds`, LIFECYCLE-DESIGN.md 3.4).
/// An output that binds none is refused.
pub(in super::super) fn external_buffer(
    circuit: &Circuit,
    e: usize,
    fixed: &Fixed,
    mode: Mode,
) -> Result<DevicePtr> {
    let edge = &circuit.edges[e];
    let stream = circuit
        .blocks
        .iter()
        .any(|b| b.stream_in == Some(e) || b.stream_out == Some(e));
    if stream {
        return Ok(fixed.hidden);
    }
    use metrale_circuit::model_buffer::ModelBuffer;
    match edge.binds {
        Some(ModelBuffer::Logits) => Ok(fixed.logits),
        Some(ModelBuffer::Tokens) if mode == Mode::VerifyBatch => Ok(fixed.verify_batch_tokens),
        Some(ModelBuffer::Tokens) => Ok(fixed.tokens),
        Some(ModelBuffer::DraftEmbed) => fixed
            .draft
            .as_ref()
            .map(|d| d.embed)
            .with_context(|| format!("`{}`: no draft embedding buffer", edge.id)),
        None => bail!(
            "`{}` is read outside the program but binds no model buffer: an unbound output",
            edge.id
        ),
    }
}

/// 2026-10-03: Where group `g`'s launches belong: its first node's layer, else a head step
/// (the `head` and `mtp_out` blocks, by the node's op), else the embedding.
pub(super) fn segment_of(circuit: &Circuit, g: &Group) -> SegmentOf {
    let Some(&first) = g.nodes.first() else {
        return SegmentOf::Embed;
    };
    let n = &circuit.nodes[first];
    match (n.layer, n.block.as_str()) {
        (Some(l), _) => SegmentOf::Layer(l),
        (None, "head" | "mtp_out") => SegmentOf::Head(n.op),
        (None, _) => SegmentOf::Embed,
    }
}

/// 2026-10-03: Merge per-launch owners into contiguous segments.
pub(super) fn segments(owners: &[SegmentOf]) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::new();
    for (i, &of) in owners.iter().enumerate() {
        match out.last_mut() {
            Some(s) if s.of == of => s.launches.end = i + 1,
            _ => out.push(Segment {
                of,
                launches: i..i + 1,
            }),
        }
    }
    out
}
