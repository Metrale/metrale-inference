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

mod roles;
pub use roles::LinearRole;

/// 2026-09-28: The closed op vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OpKind {
    /// 2026-09-28: Token embedding lookup; a source node.
    Embed,
    /// 2026-09-28: RMSNorm with a learned weight.
    RmsNorm,
    /// 2026-09-28: RMSNorm of one input gated by SiLU of another (GDN output norm).
    GatedRmsNorm,
    /// 2026-09-28: Elementwise sum of two streams.
    ResidualAdd,
    /// 2026-09-28: A copy between buffers.
    Copy,
    /// 2026-09-28: Concatenate inputs along the feature dim.
    Concat,
    /// 2026-09-28: Split or deinterleave a packed projection output.
    Split,
    /// 2026-09-28: A weight-bearing projection.
    Linear(LinearRole),
    /// 2026-09-28: `silu(gate) * up` over a packed gate|up input.
    SiluMul,
    /// 2026-09-29: `relu(x)^2` of an ungated up projection.
    Relu2,
    /// 2026-09-28: Quantize activations to the given format.
    ActQuant(Format),
    /// 2026-09-28: Per-head RMSNorm of Q or K.
    QkNorm,
    /// 2026-09-28: Rotary position embedding.
    Rope,
    /// 2026-09-28: Append K and V to the paged cache.
    KvWrite,
    /// 2026-09-28: Paged softmax attention over the cache.
    PagedAttention,
    /// 2026-09-28: `x * sigmoid(gate)` (attention output gate).
    SigmoidGateMul,
    /// 2026-09-28: GDN beta and decay from the `ba` projection.
    GdnGates,
    /// 2026-09-28: Causal depthwise conv1d step with its rolling state.
    Conv1dUpdate,
    /// 2026-09-28: L2 normalisation of Q and K heads.
    L2Norm,
    /// 2026-09-28: The gated delta rule recurrence over the recurrent state.
    GdnRecurrence,
    /// 2026-09-29: The Mamba2 selective state-space update: one step of the SSM state per row
    /// (`dt` softplus, `A`, the grouped `B`/`C` projections and the `D` skip).
    SsmUpdate,
    /// 2026-09-28: MoE router logits.
    Router,
    /// 2026-09-28: Top-k expert selection and weights.
    TopK,
    /// 2026-09-28: Routed experts' gate and up projections.
    ExpertGateUp,
    /// 2026-09-28: Routed experts' down projections.
    ExpertDown,
    /// 2026-09-28: Weighted sum of the routed experts and the gated shared expert.
    Blend,
    /// 2026-09-28: Cross-rank reduction of an expert-parallel FFN output.
    EpReduce,
    /// 2026-09-28: The model's final RMSNorm.
    FinalNorm,
    /// 2026-09-28: Vocabulary projection.
    LmHead,
    /// 2026-09-28: Greedy token selection over the logits.
    Argmax,
    /// 2026-09-29: A snapshot of the recurrent state its input's producer updates, taken after
    /// each row but the last into the step's checkpoint slots (the verify's rollback points).
    /// It has no output.
    StateSnapshot,
    /// 2026-10-08: Clamp a packed gate|up row before `silu_mul`: the gate to at most the limit,
    /// the up half to [-limit, limit] (GLM-5 `swiglu_limit`; the limit is a runtime argument).
    SwigluClamp,
    /// 2026-10-08: LayerNorm with a learned weight and bias (mean-centred, unlike RMSNorm).
    LayerNorm,
    /// 2026-10-08: The sparse-attention indexer's key pooling: each completed run of
    /// `index_kpool` tokens becomes one pool key, a per-channel softmax over the run's gate
    /// scores plus the learned position embedding weighting its keys. The incomplete run
    /// waits in a per-sequence tail. It has no output. 2026-10-10: Or (DeepSeek-V4's
    /// compressors, `ratio` its run length) the completed pools are its output rows, one per
    /// `ratio` input rows, which later nodes norm, rotate and write; with `overlap = "true"`
    /// each input row carries two series and a pool spans its own run and the previous one
    /// (width `2 * ratio`, stride `ratio`).
    KpoolCompress,
    /// 2026-10-08: The sparse-attention indexer's selection: per row, the head-weighted ReLU
    /// scores of the indexer query against every pool key, and the top `index_topk /
    /// index_kpool` pools. The pool ids are its output; the incomplete tail is always attended.
    /// 2026-10-10: Or (DeepSeek-V4, `tail = "window"`) the top `index_topk` pools, the
    /// incomplete run being covered by the sliding window instead.
    IndexSelect,
    /// 2026-10-08: Multi-head latent attention: per-head queries absorbed into the shared KV
    /// latent through `kv_b_proj`'s key half, softmax over the latent cache rows the selection
    /// names, and the result taken back to `v_head_dim` per head through its value half.
    MlaAttention,
    /// 2026-10-08: Broadcast a hidden row into `hc` hyper-connection residual streams.
    HcExpand,
    /// 2026-10-08: The hyper-connection pre-mix: from the streams and their `hc_mix`
    /// projection, the pre weights (sigmoid), the post weights and the Sinkhorn-normalised
    /// combination matrix; the sublayer input is the pre-weighted sum of the streams.
    HcPre,
    /// 2026-10-08: The hyper-connection post-mix: every new stream is its post weight times the
    /// sublayer output plus the combination of the old streams.
    HcPost,
    /// 2026-10-08: The streams' mean: one hidden row per token again (before the final norm).
    /// 2026-10-10: Or, with `weights = "sigmoid_mix"` (DeepSeek-V4's `hc_head`), the streams
    /// summed under learned weights: `sigmoid(mix * scale + base) + hc_eps` per stream, `mix`
    /// the `hc_mix` projection of the RMS-normed streams (the pre half of [`OpKind::HcPre`]
    /// without the post and combination mixes); it then reads the streams and the mix.
    HcContract,
    /// 2026-10-10: DeepSeek-V4 shared-KV attention: every query head attends, with one learned
    /// sink logit per head in its softmax denominator, the sliding window's rows and the
    /// compressed rows its layer names (`compressed`: `none`; `all` the causally complete
    /// ones; `selected` the indexer's top-k); each row is both the key and the value of the
    /// one KV head. Unlike [`OpKind::PagedAttention`] (separate K and V sides, no sink, one
    /// paged cache) and [`OpKind::MlaAttention`] (absorbed through `kv_b_proj`), it reads two
    /// caches, the per-sequence window and the per-token compressed rows.
    CompressedAttention,
}

const PLAIN_OPS: [(OpKind, &str); 39] = [
    (OpKind::Embed, "embed"),
    (OpKind::RmsNorm, "rms_norm"),
    (OpKind::GatedRmsNorm, "gated_rms_norm"),
    (OpKind::ResidualAdd, "residual_add"),
    (OpKind::Copy, "copy"),
    (OpKind::Concat, "concat"),
    (OpKind::Split, "split"),
    (OpKind::SiluMul, "silu_mul"),
    (OpKind::Relu2, "relu2"),
    (OpKind::QkNorm, "qk_norm"),
    (OpKind::Rope, "rope"),
    (OpKind::KvWrite, "kv_write"),
    (OpKind::PagedAttention, "paged_attention"),
    (OpKind::SigmoidGateMul, "sigmoid_gate_mul"),
    (OpKind::GdnGates, "gdn_gates"),
    (OpKind::Conv1dUpdate, "conv1d_update"),
    (OpKind::L2Norm, "l2_norm"),
    (OpKind::GdnRecurrence, "gdn_recurrence"),
    (OpKind::SsmUpdate, "ssm_update"),
    (OpKind::Router, "router"),
    (OpKind::TopK, "top_k"),
    (OpKind::ExpertGateUp, "expert_gate_up"),
    (OpKind::ExpertDown, "expert_down"),
    (OpKind::Blend, "blend"),
    (OpKind::EpReduce, "ep_reduce"),
    (OpKind::FinalNorm, "final_norm"),
    (OpKind::LmHead, "lm_head"),
    (OpKind::Argmax, "argmax"),
    (OpKind::StateSnapshot, "state_snapshot"),
    (OpKind::SwigluClamp, "swiglu_clamp"),
    (OpKind::LayerNorm, "layer_norm"),
    (OpKind::KpoolCompress, "kpool_compress"),
    (OpKind::IndexSelect, "index_select"),
    (OpKind::MlaAttention, "mla_attention"),
    (OpKind::HcExpand, "hc_expand"),
    (OpKind::HcPre, "hc_pre"),
    (OpKind::HcPost, "hc_post"),
    (OpKind::HcContract, "hc_contract"),
    (OpKind::CompressedAttention, "compressed_attention"),
];

impl OpKind {
    /// 2026-09-28: Resolve an op name. `linear` needs `role`, `act_quant` needs `format`;
    /// every other op refuses both.
    pub fn parse(
        name: &str,
        role: Option<&str>,
        format: Option<Format>,
    ) -> Result<Self, OpParseError> {
        match (name, role, format) {
            ("linear", Some(r), None) => LinearRole::parse(r)
                .map(OpKind::Linear)
                .ok_or_else(|| OpParseError::UnknownRole(r.to_string())),
            ("linear", None, _) => Err(OpParseError::MissingRole),
            ("act_quant", None, Some(f)) => Ok(OpKind::ActQuant(f)),
            ("act_quant", _, None) => Err(OpParseError::MissingFormat),
            _ => {
                let op = PLAIN_OPS
                    .iter()
                    .find(|(_, n)| *n == name)
                    .map(|(k, _)| *k)
                    .ok_or_else(|| OpParseError::UnknownOp(name.to_string()))?;
                if role.is_some() || format.is_some() {
                    return Err(OpParseError::StrayQualifier(name.to_string()));
                }
                Ok(op)
            }
        }
    }

    /// 2026-09-28: The op's base name (`linear`, `act_quant`, `rms_norm`, ...).
    pub fn base_name(&self) -> &'static str {
        match self {
            OpKind::Linear(_) => "linear",
            OpKind::ActQuant(_) => "act_quant",
            other => PLAIN_OPS
                .iter()
                .find(|(k, _)| k == other)
                .map(|(_, n)| *n)
                .unwrap_or("?"),
        }
    }

    /// 2026-09-28: The canonical spelling with its qualifier, e.g. `linear:down`,
    /// `act_quant:fp8/token`.
    pub fn name(&self) -> String {
        match self {
            OpKind::Linear(r) => format!("linear:{}", r.name()),
            OpKind::ActQuant(f) => format!("act_quant:{}", f.name()),
            _ => self.base_name().to_string(),
        }
    }

    /// 2026-09-28: An op that is never generated: projections, attention, the recurrence,
    /// the experts and the vocabulary projection. Fusion happens at its edges. 2026-10-08: Also
    /// the indexer's selection, latent attention and the hyper-connection pre-mix (an iterative
    /// Sinkhorn normalisation). 2026-10-10: And DeepSeek-V4's shared-KV attention.
    pub fn is_heavy(&self) -> bool {
        matches!(
            self,
            OpKind::Linear(_)
                | OpKind::PagedAttention
                | OpKind::GdnRecurrence
                | OpKind::SsmUpdate
                | OpKind::Router
                | OpKind::ExpertGateUp
                | OpKind::ExpertDown
                | OpKind::LmHead
                | OpKind::IndexSelect
                | OpKind::MlaAttention
                | OpKind::HcPre
                | OpKind::CompressedAttention
        )
    }

    /// 2026-09-28: Ops that read a weight whose format the precision resolver answers.
    /// 2026-10-08: Latent attention reads `kv_b_proj` ([`Circuit::weight_shape`]).
    pub fn reads_linear_weight(&self) -> bool {
        matches!(
            self,
            OpKind::Linear(_)
                | OpKind::ExpertGateUp
                | OpKind::ExpertDown
                | OpKind::LmHead
                | OpKind::Router
                | OpKind::MlaAttention
        )
    }
}

/// 2026-09-28: Why an op spelling was refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum OpParseError {
    /// 2026-09-28: Not in the vocabulary.
    #[error("unknown op `{0}`")]
    UnknownOp(String),
    /// 2026-09-28: `linear` with a role outside [`LinearRole`].
    #[error("unknown linear role `{0}`")]
    UnknownRole(String),
    /// 2026-09-28: `linear` without `role`.
    #[error("`linear` needs a `role`")]
    MissingRole,
    /// 2026-09-28: `act_quant` without `format`.
    #[error("`act_quant` needs a `format`")]
    MissingFormat,
    /// 2026-09-28: `role` or `format` on an op that takes neither.
    #[error("op `{0}` takes no `role` or `format`")]
    StrayQualifier(String),
}

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
    /// 2026-10-10: A DeepSeek-V4 layer whose attention reads only its sliding window
    /// (`sliding_attention`, compress ratio 0): no compressor, no indexer.
    SlidingAttention,
    /// 2026-10-10: A DeepSeek-V4 layer whose attention also reads the indexer-selected rows of
    /// a 4x compressed KV (`compressed_sparse_attention`, CSA): an overlapping-window
    /// compressor, and an indexer with its own compressor.
    CompressedSparseAttention,
    /// 2026-10-10: A DeepSeek-V4 layer whose attention also reads every row of a 128x
    /// compressed KV (`heavily_compressed_attention`, HCA): a non-overlapping compressor, no
    /// indexer.
    HeavilyCompressedAttention,
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
            LayerKind::CompressedSparseAttention => "compressed_sparse_attention",
            LayerKind::HeavilyCompressedAttention => "heavily_compressed_attention",
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
            "compressed_sparse_attention" => Some(LayerKind::CompressedSparseAttention),
            "heavily_compressed_attention" => Some(LayerKind::HeavilyCompressedAttention),
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
    /// missing or the product overflows. 2026-10-10: A grouped output projection
    /// ([`LinearRole::OGroup`]) holds one `[out / o_groups, k / o_groups]` block per group, so
    /// its weight is `[out, k / o_groups]`; `None` when `o_groups` is missing, zero, or does
    /// not divide `k`.
    pub fn weight_shape(&self, n: &Node) -> Option<(u64, u64)> {
        let d = |k: &str| self.dims.get(k).copied();
        if n.op == OpKind::MlaAttention {
            let per_head = d("mla_qk")?.checked_add(d("mla_v")?)?;
            return Some((d("q_heads")?.checked_mul(per_head)?, d("kv_lora")?));
        }
        let k = n.inputs.first().map_or(0, |&e| self.edges[e].dim_value);
        let out = n.outputs.iter().map(|&e| self.edges[e].dim_value).sum();
        if n.op == OpKind::Linear(LinearRole::OGroup) {
            let g = d("o_groups").filter(|&g| g > 0 && k.is_multiple_of(g))?;
            return Some((out, k / g));
        }
        Some((out, k))
    }
}

#[cfg(test)]
mod tests;
