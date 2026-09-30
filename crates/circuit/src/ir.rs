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

/// 2026-09-28: What a linear (GEMV/GEMM) node projects. Closed, like [`OpKind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LinearRole {
    /// 2026-09-28: Attention Q with its output gate, interleaved per head.
    Q,
    /// 2026-09-28: Attention K.
    K,
    /// 2026-09-28: Attention V.
    V,
    /// 2026-09-28: Attention output projection.
    O,
    /// 2026-09-28: GatedDeltaNet `in_proj_qkv` and `in_proj_z` in one projection.
    Qkvz,
    /// 2026-09-28: GatedDeltaNet `in_proj_b` and `in_proj_a` (the beta and decay inputs).
    Ba,
    /// 2026-09-28: GatedDeltaNet `out_proj`.
    GdnOut,
    /// 2026-09-28: MLP gate and up projections.
    GateUp,
    /// 2026-09-28: MLP down projection.
    Down,
    /// 2026-09-28: MoE shared-expert gate and up projections.
    SharedGateUp,
    /// 2026-09-28: MoE shared-expert down projection.
    SharedDown,
    /// 2026-09-28: MoE shared-expert scalar gate (`shared_expert_gate`).
    SharedGate,
    /// 2026-09-28: MTP head input projection (`fc`, embedding and hidden concatenated).
    MtpFc,
    /// 2026-09-29: Mamba2 `in_proj`: z, x, B, C and dt in one projection.
    MambaIn,
    /// 2026-09-29: Mamba2 `out_proj`.
    MambaOut,
    /// 2026-09-29: An ungated MoE shared expert's up projection.
    SharedUp,
    /// 2026-09-30: A latent MoE's projection from the hidden width into the experts' latent
    /// width (`fc1_latent_proj`, Nemotron-3 Super).
    MoeLatentIn,
    /// 2026-09-30: A latent MoE's projection of the routed sum back to the hidden width
    /// (`fc2_latent_proj`).
    MoeLatentOut,
}

const ROLES: [(LinearRole, &str); 18] = [
    (LinearRole::Q, "q"),
    (LinearRole::K, "k"),
    (LinearRole::V, "v"),
    (LinearRole::O, "o"),
    (LinearRole::Qkvz, "qkvz"),
    (LinearRole::Ba, "ba"),
    (LinearRole::GdnOut, "gdn_out"),
    (LinearRole::GateUp, "gate_up"),
    (LinearRole::Down, "down"),
    (LinearRole::SharedGateUp, "shared_gate_up"),
    (LinearRole::SharedDown, "shared_down"),
    (LinearRole::SharedGate, "shared_gate"),
    (LinearRole::MtpFc, "mtp_fc"),
    (LinearRole::MambaIn, "mamba_in"),
    (LinearRole::MambaOut, "mamba_out"),
    (LinearRole::SharedUp, "shared_up"),
    (LinearRole::MoeLatentIn, "moe_latent_in"),
    (LinearRole::MoeLatentOut, "moe_latent_out"),
];

impl LinearRole {
    /// 2026-09-28: The role spelled in templates and rules.
    pub fn parse(s: &str) -> Option<Self> {
        ROLES.iter().find(|(_, n)| *n == s).map(|(r, _)| *r)
    }

    /// 2026-09-28: The canonical spelling.
    pub fn name(self) -> &'static str {
        ROLES
            .iter()
            .find(|(r, _)| *r == self)
            .map(|(_, n)| *n)
            .unwrap_or("?")
    }
}

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
}

const PLAIN_OPS: [(OpKind, &str); 29] = [
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
    /// the experts and the vocabulary projection. Fusion happens at its edges.
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
        )
    }

    /// 2026-09-28: Ops that read a weight whose format the precision resolver answers.
    pub fn reads_linear_weight(&self) -> bool {
        matches!(
            self,
            OpKind::Linear(_)
                | OpKind::ExpertGateUp
                | OpKind::ExpertDown
                | OpKind::LmHead
                | OpKind::Router
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
    /// 2026-09-28: A GatedDeltaNet layer.
    LinearAttention,
    /// 2026-09-28: A softmax attention layer.
    FullAttention,
    /// 2026-09-29: A Mamba2 layer (Nemotron-H `mamba`).
    Mamba,
    /// 2026-09-29: A layer whose only mixer is a MoE FFN (Nemotron-H `moe`).
    Moe,
}

impl LayerKind {
    /// 2026-09-28: The spelling in HF configs and circuit layouts.
    pub fn name(self) -> &'static str {
        match self {
            LayerKind::LinearAttention => "linear_attention",
            LayerKind::FullAttention => "full_attention",
            LayerKind::Mamba => "mamba",
            LayerKind::Moe => "moe",
        }
    }

    /// 2026-09-28: Parse the HF spelling.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "linear_attention" => Some(LayerKind::LinearAttention),
            "full_attention" => Some(LayerKind::FullAttention),
            "mamba" => Some(LayerKind::Mamba),
            "moe" => Some(LayerKind::Moe),
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
}
