// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The closed op vocabulary ([`OpKind`]) and its spelling, moved out of `ir.rs`
//! unchanged to keep that file short.
//!
//! Owner: metrale-circuit.
//! Invariants: see [`crate::ir`]; [`OpKind`] is closed.

use crate::prelude::*;

use crate::format::Format;

use super::LinearRole;

/// 2026-09-28: The closed op vocabulary.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
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
    /// 2026-10-23: The sum of the group's partials on every rank of the group (`params.group`),
    /// derived where a partial meets a consumer not linear in it ([`crate::parallel`]).
    AllReduce,
    /// 2026-10-23: The group's slices concatenated on every rank of the group, derived where a
    /// sliced edge meets a consumer that needs it whole.
    AllGather,
    /// 2026-10-23: The group's first rank's tensor sent to every rank (`params.root`), derived
    /// at the boundary of a part placed on fewer ranks than N.
    Broadcast,
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
    /// waits in a per-sequence tail. It has no output.
    KpoolCompress,
    /// 2026-10-08: The sparse-attention indexer's selection: per row, the head-weighted ReLU
    /// scores of the indexer query against every pool key, and the top `index_topk /
    /// index_kpool` pools. The pool ids are its output; the incomplete tail is always attended.
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
    HcContract,
    /// 2026-10-18: Sparse attention over one shared latent row per token (keys and values are
    /// the latent itself, no `kv_b_proj`), with a learned per-head sink logit: the last
    /// `window` rows from a per-sequence ring it updates, plus (param `compressed`, a state
    /// name) the compressed rows the indexer selects, read from its own state or from the
    /// nearest earlier layer's state of that name. DeepSeek-V4.1 attention.
    LatentSparseAttention,
    /// 2026-10-18: Pool every `ratio` tokens' projected latents into one, weighted by a
    /// per-channel softmax over the run's gate scores (no position term); the incomplete run
    /// waits in the per-sequence tail it updates. The pooled latent is its output.
    KvCompress,
    /// 2026-10-18: Hashed n-gram embedding lookup (Engram): the rows of a large embedding
    /// table named by token-hash ids the host computes; a source node like [`OpKind::Embed`].
    EngramLookup,
    /// 2026-10-18: The Engram residual: per stream, a sigmoid gate from the normalised dot of
    /// the stream with its key slice, times the shared value slice, added to the stream.
    EngramGate,
}

pub(super) const PLAIN_OPS: [(OpKind, &str); 45] = [
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
    (OpKind::AllReduce, "all_reduce"),
    (OpKind::AllGather, "all_gather"),
    (OpKind::Broadcast, "broadcast"),
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
    (OpKind::LatentSparseAttention, "latent_sparse_attention"),
    (OpKind::KvCompress, "kv_compress"),
    (OpKind::EngramLookup, "engram_lookup"),
    (OpKind::EngramGate, "engram_gate"),
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
    /// Sinkhorn normalisation).
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
                | OpKind::LatentSparseAttention
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
