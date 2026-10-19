// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The circuit IR: a closed op vocabulary, edges with a format and a shape, nodes
//! bound to checkpoint modules, and the instantiated graph of one model.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - [`OpKind`] is closed. A template naming anything else is a load error, never an
//!   opaque pass-through.
//! - Heavy ops ([`OpKind::is_heavy`]) are opaque: the fuser fuses at their edges or through a
//!   pre-written kernel, and never generates them.
//! - Nodes are stored in execution order, which is a topological order: every input of a
//!   node is produced by an earlier node or is a circuit input.
//! - An edge's fused or materialised state belongs to a plan (`FusionPlan::edge_states`),
//!   not to the circuit, because one circuit has one plan per mode and row count.

use std::collections::BTreeMap;

use crate::dims::DimExpr;
use crate::format::Format;

mod ops;
mod roles;
pub use ops::{OpKind, OpParseError};
pub use roles::LinearRole;

/// 2026-09-28: Index of an edge in [`Circuit::edges`].
pub type EdgeIdx = usize;
/// 2026-09-28: Index of a node in [`Circuit::nodes`].
pub type NodeIdx = usize;

/// 2026-09-28: One tensor between nodes: `rows x dim` in `format`. `rows` is an expression
/// over `n`, the step's row count, and the arch dims (e.g. `n*top_k`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edge {
    /// 2026-09-28: Unique id, e.g. `l3.attn.q` or `stream.4`.
    pub id: String,
    /// 2026-09-28: Element format.
    pub format: Format,
    /// 2026-09-28: Row expression; reads `n`.
    pub rows: DimExpr,
    /// 2026-09-28: Feature dimension expression.
    pub dim: DimExpr,
    /// 2026-09-28: `dim` evaluated under the arch dims.
    pub dim_value: u64,
    /// 2026-09-28: The node that writes it; `None` for none (refused by the loader).
    pub producer: Option<NodeIdx>,
    /// 2026-09-28: The nodes that read it, in node order.
    pub consumers: Vec<NodeIdx>,
    /// 2026-09-28: Read outside the circuit (the logits); never fused away.
    pub is_output: bool,
    /// 2026-09-30: The model buffer a declared output lands in; `None` for an output declared
    /// without one (the executor refuses such a program) and for every other edge.
    pub binds: Option<crate::model_buffer::ModelBuffer>,
}

/// 2026-09-28: One op instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    /// 2026-09-28: Unique id, e.g. `l3.attn.o_proj`.
    pub id: String,
    /// 2026-09-28: The id inside its block template, e.g. `o_proj`.
    pub local: String,
    /// 2026-09-28: The op.
    pub op: OpKind,
    /// 2026-09-28: Input edges, in the op's argument order.
    pub inputs: Vec<EdgeIdx>,
    /// 2026-09-28: Output edges.
    pub outputs: Vec<EdgeIdx>,
    /// 2026-09-28: Weight format, for ops that read a linear weight.
    pub weight: Option<Format>,
    /// 2026-09-28: Checkpoint modules this node reads (`layers.3.mlp.down_proj`), in order.
    pub binding: Vec<String>,
    /// 2026-09-28: Op parameters from the template, e.g. `top_k = "8"`.
    pub params: BTreeMap<String, String>,
    /// 2026-09-28: The layer this node belongs to; `None` in the prologue and epilogue.
    pub layer: Option<usize>,
    /// 2026-09-28: The block template it was instantiated from.
    pub block: String,
    /// 2026-09-30: The states it touches: indices into [`Circuit::states`], and how.
    pub state: Vec<(usize, crate::state::StateAccess)>,
}

/// 2026-09-28: A decoder layer's kind, as `ModelConfig::layer_type` reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LayerKind {
    /// 2026-09-28: A GatedDeltaNet layer. 2026-10-08: Or a KDA layer (`glm5_next`): the HF
    /// spelling is the same, and the circuit's layout picks the mixer.
    LinearAttention,
    /// 2026-09-28: A softmax attention layer.
    FullAttention,
    /// 2026-09-29: A Mamba2 layer (Nemotron-H `mamba`).
    Mamba,
    /// 2026-09-29: A layer whose only mixer is a MoE FFN (Nemotron-H `moe`).
    Moe,
    /// 2026-10-08: A sparse-attention layer: latent attention over the tokens an indexer
    /// selects (`deepseek_sparse_attention`, GLM-5).
    SparseAttention,
    /// 2026-10-18: A sliding-window attention layer: softmax over the last `window` tokens only
    /// (`sliding_attention`; DeepSeek-V4.1 layers whose `compress_ratios` entry is 0).
    SlidingAttention,
}

impl LayerKind {
    /// 2026-09-28: The spelling in HF configs and circuit layouts.
    pub fn name(self) -> &'static str {
        match self {
            LayerKind::LinearAttention => "linear_attention",
            LayerKind::FullAttention => "full_attention",
            LayerKind::Mamba => "mamba",
            LayerKind::Moe => "moe",
            LayerKind::SparseAttention => "deepseek_sparse_attention",
            LayerKind::SlidingAttention => "sliding_attention",
        }
    }

    /// 2026-09-28: Parse the HF spelling.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "linear_attention" => Some(LayerKind::LinearAttention),
            "full_attention" => Some(LayerKind::FullAttention),
            "mamba" => Some(LayerKind::Mamba),
            "moe" => Some(LayerKind::Moe),
            "deepseek_sparse_attention" => Some(LayerKind::SparseAttention),
            "sliding_attention" => Some(LayerKind::SlidingAttention),
            _ => None,
        }
    }
}

/// 2026-09-28: What instantiation needs from a model config: its layer kinds in order and
/// its named dims. The caller fills it from `ModelConfig`, so this crate does not depend on
/// the config tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchShape {
    /// 2026-09-28: Kind of each text-stack layer, index = layer.
    pub layer_kinds: Vec<LayerKind>,
    /// 2026-09-28: Named dims (`hidden`, `head_dim`, ...), the names the templates read.
    pub dims: BTreeMap<String, u64>,
}

/// 2026-09-28: Which forward a block belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Section {
    /// 2026-09-28: The target model's forward.
    Main,
    /// 2026-09-28: The speculative draft head (MTP), which reads the main stream.
    Draft,
}

/// 2026-09-28: One instantiated block: its template and the node range it produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockInstance {
    /// 2026-09-28: Template name.
    pub template: String,
    /// 2026-09-28: Layer index; `None` for the prologue, epilogue and draft.
    pub layer: Option<usize>,
    /// 2026-09-28: Main or draft.
    pub section: Section,
    /// 2026-09-28: First node.
    pub first: NodeIdx,
    /// 2026-09-28: One past the last node.
    pub end: NodeIdx,
    /// 2026-09-28: The stream edge entering the block, if it reads one.
    pub stream_in: Option<EdgeIdx>,
    /// 2026-09-28: The stream edge leaving it, if it writes one.
    pub stream_out: Option<EdgeIdx>,
}

/// 2026-09-28: An instantiated circuit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Circuit {
    /// 2026-09-28: The arch id of the TOML it came from (`qwen3_5`).
    pub arch: String,
    /// 2026-09-28: The TOML's one-line description.
    pub description: String,
    /// 2026-09-28: Nodes in execution order.
    pub nodes: Vec<Node>,
    /// 2026-09-28: All edges.
    pub edges: Vec<Edge>,
    /// 2026-09-28: Blocks in execution order.
    pub blocks: Vec<BlockInstance>,
    /// 2026-09-28: Layer kinds, from the arch shape.
    pub layer_kinds: Vec<LayerKind>,
    /// 2026-09-28: The dims every expression was evaluated under.
    pub dims: BTreeMap<String, u64>,
    /// 2026-09-30: The state every block keeps between steps, in block order.
    pub states: Vec<crate::state::StateDecl>,
}

impl Circuit {
    /// 2026-09-28: Find an edge by id.
    pub fn edge(&self, id: &str) -> Option<EdgeIdx> {
        self.edges.iter().position(|e| e.id == id)
    }

    /// 2026-09-28: Find a node by id.
    pub fn node(&self, id: &str) -> Option<NodeIdx> {
        self.nodes.iter().position(|n| n.id == id)
    }

    /// 2026-10-08: `(out, k)` of the weight matrix node `n` reads: `k` its first input's width
    /// and `out` the sum of its output widths, except for latent attention, whose weight is
    /// `kv_b_proj`, `[q_heads * (mla_qk + mla_v), kv_lora]`. One definition for the roofline,
    /// the footprint and the memory model. `None` when a dim latent attention needs is
    /// missing or the product overflows.
    /// 2026-10-18: A grouped linear (param `groups`, DeepSeek-V4 `wo_a`) multiplies each of its
    /// `groups` input slices by its own `out / groups` rows, so `k` is the input width over
    /// `groups`; `None` when `groups` is not a positive divisor of it.
    pub fn weight_shape(&self, n: &Node) -> Option<(u64, u64)> {
        if n.op == OpKind::MlaAttention {
            let d = |k: &str| self.dims.get(k).copied();
            let per_head = d("mla_qk")?.checked_add(d("mla_v")?)?;
            return Some((d("q_heads")?.checked_mul(per_head)?, d("kv_lora")?));
        }
        let k = n.inputs.first().map_or(0, |&e| self.edges[e].dim_value);
        let k = match (n.op, n.params.get("groups")) {
            (OpKind::Linear(_), Some(g)) => {
                let g = g
                    .parse::<u64>()
                    .ok()
                    .filter(|g| *g > 0 && k.is_multiple_of(*g))?;
                k / g
            }
            _ => k,
        };
        let out = n.outputs.iter().map(|&e| self.edges[e].dim_value).sum();
        Some((out, k))
    }
}

#[cfg(test)]
mod tests;
