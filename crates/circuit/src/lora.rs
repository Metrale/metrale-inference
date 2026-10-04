// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: LoRA adapters as circuit nodes (LIFECYCLE-DESIGN.md 15.10). [`adapt`] rewrites an
//! instantiated circuit so that after every adapted projection of the target's forward a
//! `lora_shrink` (`x · A`, rank wide) and a `lora_expand` (`y + scale · xa · B`) run, and every
//! reader of the projection's output reads the adapted output instead. The executor's LoRA
//! emitters launch them; the rules of FUSIONS.toml's LoRA section choose the kernels per mode.
//!
//! Owner: metrale-circuit (FEATURES workstream).
//! Invariants:
//! - A model served without adapters is never adapted, so its circuit, plans and digests are
//!   today's exactly; the policy then carries no LoRA setting either.
//! - Only the target's forward (`Section::Main`) is adapted: legacy never adapts the MTP draft
//!   head (`install_lora_layers` walks the target's layers only).
//! - An adapted projection's output edge keeps its id and is read by the expand alone; the
//!   adapted output is a new edge of the same format and shape.
//! - A role legacy cannot adapt is refused, never skipped: the circuit must not claim an
//!   adapter it does not run.

use std::collections::{BTreeMap, BTreeSet};

use crate::dims::DimExpr;
use crate::follow::{Followers, push_edge};
use crate::format::Format;
use crate::ir::{Circuit, Edge, EdgeIdx, LinearRole, Node, NodeIdx, OpKind};

/// 2026-10-03: The dim every LoRA edge's width is written in.
pub const RANK_DIM: &str = "lora_rank";

/// 2026-10-03: The policy setting a LoRA build states (`on`), so its plans' settings line
/// discloses the adapters; a build without adapters states nothing.
pub const ACTIVE_SETTING: &str = "lora_active";

/// 2026-10-03: The projections legacy can adapt in the target's layers
/// (`metrale_model_layers::lora::key`): attention q/k/v/o, the dense FFN's gate, up and down,
/// and the GatedDeltaNet out_proj. The GatedDeltaNet input projections, the shared expert, the
/// MTP head, the embedding and the head are refused there.
pub const ADAPTABLE: [LinearRole; 7] = [
    LinearRole::Q,
    LinearRole::K,
    LinearRole::V,
    LinearRole::O,
    LinearRole::GateUp,
    LinearRole::Down,
    LinearRole::GdnOut,
];

/// 2026-10-03: What the loaded adapters adapt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoraSpec {
    /// 2026-10-03: The pool's padded rank (`LoraWeights::max_rank`).
    pub rank: u64,
    /// 2026-10-03: Per target layer, the projections its adapters fold into: attention's that
    /// the pool routes, the FFN's and out_proj's that the active adapter has a pair for (legacy
    /// skips a layer whose active pair is absent). A layer not listed is not adapted.
    pub layers: BTreeMap<usize, BTreeSet<LinearRole>>,
}

impl LoraSpec {
    /// 2026-10-03: `roles` on every layer in `layers`.
    pub fn uniform(
        rank: u64,
        layers: impl IntoIterator<Item = usize>,
        roles: &[LinearRole],
    ) -> Self {
        LoraSpec {
            rank,
            layers: layers
                .into_iter()
                .map(|l| (l, roles.iter().copied().collect()))
                .collect(),
        }
    }
}

/// 2026-10-03: Why a circuit could not be adapted.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LoraError {
    /// 2026-10-03: A rank of zero, or no adapted projection at all (then serve without the
    /// overlay).
    #[error("a LoRA spec needs a rank above 0 and at least one role (rank {rank}, {roles} roles)")]
    Empty {
        /// 2026-10-03: The rank given.
        rank: u64,
        /// 2026-10-03: The (layer, role) pairs given.
        roles: usize,
    },
    /// 2026-10-03: A role legacy cannot adapt.
    #[error("LoRA cannot adapt `linear:{0}` (adaptable: q, k, v, o, gate_up, down, gdn_out)")]
    Role(&'static str),
    /// 2026-10-03: An adapted role that layer's forward does not project.
    #[error("layer {1} has no `linear:{0}` node to adapt")]
    Absent(&'static str, usize),
    /// 2026-10-03: A projection whose output is read outside the program.
    #[error("`{0}`'s output is read outside the program; its adapter cannot follow it")]
    External(String),
    /// 2026-10-03: A projection that reads a quantized activation (a declared W8A8 or W4A4
    /// projection): legacy leaves that arm when an adapter is installed (`w8a8_decode_arm.rs`),
    /// which a plan at the declared precision cannot.
    #[error("`{0}` reads a {1} activation; LoRA adapts only 16-bit-activation projections")]
    QuantizedInput(String, Format),
    /// 2026-10-03: The shrink's width does not evaluate (a rank that overflows).
    #[error("the LoRA shrink width: {0}")]
    Width(String),
    /// 2026-10-03: The circuit already names the rank dim.
    #[error("the circuit already defines `{RANK_DIM}`: adapted twice?")]
    Twice,
}

/// 2026-10-03: Device bytes of the adapter pool an adapted circuit binds: per shrink node, `slots`
/// adapters of BF16 `A` (`[rank, k_in]` per half: gate and up are two) and `B` (`[n_out, rank]`),
/// every adapter padded to the pool's rank (`LoraWeights::max_rank`); the memory model's term.
pub fn pool_bytes(circuit: &Circuit, slots: u64) -> u64 {
    circuit
        .nodes
        .iter()
        .filter(|n| n.op == OpKind::LoraShrink)
        .map(|n| {
            let k_in = circuit.edges[n.inputs[0]].dim_value;
            let xa = circuit.edges[n.outputs[0]].dim_value;
            let n_out: u64 = circuit
                .nodes
                .iter()
                .filter(|m| m.op == OpKind::LoraExpand && m.inputs[0] == n.outputs[0])
                .map(|m| circuit.edges[m.outputs[0]].dim_value)
                .sum();
            slots * (xa * k_in + n_out * circuit.dims[RANK_DIM]) * 2
        })
        .sum()
}

/// 2026-10-03: The shrink's width for `role`: two adapters (gate, up) share the gate|up
/// projection.
fn xa_width(role: LinearRole) -> &'static str {
    match role {
        LinearRole::GateUp => "lora_rank*2",
        _ => RANK_DIM,
    }
}

/// 2026-10-03: `circuit` with the adapters of `spec` after each adapted projection.
pub fn adapt(circuit: &Circuit, spec: &LoraSpec) -> Result<Circuit, LoraError> {
    let roles: usize = spec.layers.values().map(BTreeSet::len).sum();
    if spec.rank == 0 || roles == 0 {
        return Err(LoraError::Empty {
            rank: spec.rank,
            roles,
        });
    }
    if let Some(r) = spec
        .layers
        .values()
        .flatten()
        .find(|r| !ADAPTABLE.contains(r))
    {
        return Err(LoraError::Role(r.name()));
    }
    if circuit.dims.contains_key(RANK_DIM) {
        return Err(LoraError::Twice);
    }
    let main = crate::follow::main_nodes(circuit);
    let wants = |n: &Node| match (n.op, n.layer) {
        (OpKind::Linear(r), Some(l)) => spec.layers.get(&l).is_some_and(|rs| rs.contains(&r)),
        _ => false,
    };
    let adapted: BTreeSet<NodeIdx> = (0..circuit.nodes.len())
        .filter(|&n| main[n] && wants(&circuit.nodes[n]))
        .collect();
    for (&l, rs) in &spec.layers {
        for r in rs {
            let present = adapted.iter().any(|&n| {
                circuit.nodes[n].op == OpKind::Linear(*r) && circuit.nodes[n].layer == Some(l)
            });
            if !present {
                return Err(LoraError::Absent(r.name(), l));
            }
        }
    }
    let mut base = circuit.clone();
    base.dims.insert(RANK_DIM.to_string(), spec.rank);
    crate::follow::follow(&base, &adapted, |out, node| adapter_of(out, node, spec))
}

/// 2026-10-03: The shrink and expand that follow adapted `node`, their edges pushed into `out`.
fn adapter_of(out: &mut Circuit, node: &Node, spec: &LoraSpec) -> Result<Followers, LoraError> {
    let OpKind::Linear(role) = node.op else {
        unreachable!("only projections are adapted")
    };
    let (&x, &y) = match (node.inputs.first(), node.outputs.first()) {
        (Some(x), Some(y)) => (x, y),
        _ => return Err(LoraError::External(node.id.clone())),
    };
    let base = out.edges[y].clone();
    let stream = out
        .blocks
        .iter()
        .any(|b| b.stream_in == Some(y) || b.stream_out == Some(y));
    if base.is_output || stream {
        return Err(LoraError::External(node.id.clone()));
    }
    if !out.edges[x].format.is_plain() {
        return Err(LoraError::QuantizedInput(
            node.id.clone(),
            out.edges[x].format,
        ));
    }
    let width = DimExpr::parse(xa_width(role)).expect("a fixed expression");
    let dim_value = width
        .eval(&out.dims)
        .map_err(|e| LoraError::Width(e.to_string()))?;
    let xa = push_edge(
        out,
        Edge {
            id: format!("{}.lora_xa", node.id),
            format: Format::Bf16,
            rows: base.rows.clone(),
            dim: width,
            dim_value,
            producer: None,
            consumers: Vec::new(),
            is_output: false,
            binds: None,
        },
    );
    let adapted = push_edge(
        out,
        Edge {
            id: format!("{}.lora", node.id),
            producer: None,
            consumers: Vec::new(),
            ..base
        },
    );
    let params = BTreeMap::from([
        ("role".to_string(), role.name().to_string()),
        ("rank".to_string(), spec.rank.to_string()),
    ]);
    let make = |suffix: &str, op: OpKind, inputs: Vec<EdgeIdx>, output: EdgeIdx| Node {
        id: format!("{}_lora_{suffix}", node.id),
        local: format!("{}_lora_{suffix}", node.local),
        op,
        inputs,
        outputs: vec![output],
        weight: Some(Format::Bf16),
        binding: Vec::new(),
        params: params.clone(),
        layer: node.layer,
        block: node.block.clone(),
        state: Vec::new(),
    };
    Ok(Followers {
        nodes: vec![
            make("a", OpKind::LoraShrink, vec![x], xa),
            make("b", OpKind::LoraExpand, vec![xa, y], adapted),
        ],
        replaces: Some((y, adapted)),
        leader: None,
    })
}

#[cfg(test)]
#[path = "lora_tests.rs"]
mod lora_tests;
