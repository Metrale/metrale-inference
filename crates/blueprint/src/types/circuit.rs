// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The circuit instance as wire records: indices are `u32` (BU64), op and format
//! names their canonical spellings.
//!
//! Owner: blueprint.
//! Invariants: wire types only (see `types/mod.rs`); [`LayerKind`], [`Section`] and
//! [`StateAccess`] mirror `metrale_circuit`'s and are converted exhaustively at the adapter.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

/// 2026-10-22: The instantiated circuit.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CircuitSpec {
    /// 2026-10-22: The arch, e.g. `glm5_next`.
    pub arch: String,
    /// 2026-10-22: The dims every expression was evaluated under.
    pub dims: BTreeMap<String, u64>,
    /// 2026-10-22: Per text-stack layer.
    pub layer_kinds: Vec<LayerKind>,
    /// 2026-10-22: In execution order.
    pub nodes: Vec<NodeSpec>,
    /// 2026-10-22: Every tensor between nodes.
    pub edges: Vec<EdgeSpec>,
    /// 2026-10-22: Blocks in execution order.
    pub blocks: Vec<BlockSpec>,
    /// 2026-10-22: The state kept between steps.
    pub states: Vec<StateSpec>,
}

/// 2026-10-22: A decoder layer's kind (mirror of `metrale_circuit::LayerKind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LayerKind {
    /// 2026-10-22: GatedDeltaNet or KDA.
    LinearAttention,
    /// 2026-10-22: Softmax attention.
    FullAttention,
    /// 2026-10-22: Mamba2.
    Mamba,
    /// 2026-10-22: MoE-only layer.
    Moe,
    /// 2026-10-22: Indexer-selected latent attention.
    SparseAttention,
    /// 2026-10-22: Sliding-window attention.
    SlidingAttention,
}

impl LayerKind {
    /// 2026-10-22: The export spelling (`sparse_attention` for the HF
    /// `deepseek_sparse_attention`).
    pub const fn name(self) -> &'static str {
        match self {
            LayerKind::LinearAttention => "linear_attention",
            LayerKind::FullAttention => "full_attention",
            LayerKind::Mamba => "mamba",
            LayerKind::Moe => "moe",
            LayerKind::SparseAttention => "sparse_attention",
            LayerKind::SlidingAttention => "sliding_attention",
        }
    }
}

/// 2026-10-22: Main forward or draft head.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Section {
    /// 2026-10-22: The target model.
    Main,
    /// 2026-10-22: The MTP draft head.
    Draft,
}

/// 2026-10-22: How a node touches a state (mirror of `metrale_circuit::state::StateAccess`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateAccess {
    /// 2026-10-22: Reads it.
    Read,
    /// 2026-10-22: Writes new units.
    Write,
    /// 2026-10-22: Reads and replaces in place.
    Update,
    /// 2026-10-22: Copies into the verify intermediates.
    Snapshot,
}

impl StateAccess {
    /// 2026-10-22: The spelling.
    pub const fn name(self) -> &'static str {
        match self {
            StateAccess::Read => "read",
            StateAccess::Write => "write",
            StateAccess::Update => "update",
            StateAccess::Snapshot => "snapshot",
        }
    }
}

/// 2026-10-22: One op instance.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeSpec {
    /// 2026-10-22: Unique id, e.g. `l3.attn.o_proj`.
    pub id: String,
    /// 2026-10-22: The id inside its block template.
    pub local: String,
    /// 2026-10-22: The op's canonical spelling (`OpKind::name`).
    pub op: String,
    /// 2026-10-22: Input edges.
    pub inputs: Vec<u32>,
    /// 2026-10-22: Output edges.
    pub outputs: Vec<u32>,
    /// 2026-10-22: Weight format spelling, for ops reading a weight.
    pub weight: Option<String>,
    /// 2026-10-22: Checkpoint modules read.
    pub binding: Vec<String>,
    /// 2026-10-22: Op parameters.
    pub params: BTreeMap<String, String>,
    /// 2026-10-22: The layer, if any.
    pub layer: Option<u32>,
    /// 2026-10-22: The block template.
    pub block: String,
    /// 2026-10-22: States touched: index into `states`, and how.
    pub state: Vec<(u32, StateAccess)>,
}

/// 2026-10-22: One tensor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeSpec {
    /// 2026-10-22: Unique id.
    pub id: String,
    /// 2026-10-22: Declared element format spelling.
    pub format: String,
    /// 2026-10-22: Row expression text (reads `n`).
    pub rows: String,
    /// 2026-10-22: Feature dimension expression text.
    pub dim: String,
    /// 2026-10-22: `dim` evaluated.
    pub dim_value: u64,
    /// 2026-10-22: The writer.
    pub producer: Option<u32>,
    /// 2026-10-22: The readers.
    pub consumers: Vec<u32>,
    /// 2026-10-22: Read outside the circuit.
    pub is_output: bool,
    /// 2026-10-22: The model buffer an output lands in.
    pub binds: Option<String>,
}

/// 2026-10-22: One instantiated block.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockSpec {
    /// 2026-10-22: Template name.
    pub template: String,
    /// 2026-10-22: Layer, if any.
    pub layer: Option<u32>,
    /// 2026-10-22: Main or draft.
    pub section: Section,
    /// 2026-10-22: First node.
    pub first: u32,
    /// 2026-10-22: One past the last node.
    pub end: u32,
}

/// 2026-10-22: One state kept between steps.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateSpec {
    /// 2026-10-22: `l0.gdn.h`.
    pub id: String,
    /// 2026-10-22: Template-local id.
    pub local: String,
    /// 2026-10-22: Block template.
    pub block: String,
    /// 2026-10-22: Layer, if any.
    pub layer: Option<u32>,
    /// 2026-10-22: Main or draft.
    pub section: Section,
    /// 2026-10-22: Kind spelling.
    pub kind: String,
    /// 2026-10-22: Storage format spelling.
    pub format: String,
    /// 2026-10-22: Elements per unit.
    pub elements: u64,
    /// 2026-10-22: Lifetime spelling.
    pub lifetime: String,
}
