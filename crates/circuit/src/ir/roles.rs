// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The closed list of what a linear node projects ([`LinearRole`]), split out of
//! `ir.rs` unchanged apart from the roles added for `glm5_next` (KDA, MLA, the sparse-attention
//! indexer and the hyper-connection mixes).
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - Every role has one spelling (`ROLES`); a spelling outside the list is a load error.
//! - New roles are appended, so the derived order of the existing ones does not change.

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
    /// 2026-10-08: KDA (Kimi delta attention) `b_proj`: the per-head beta input.
    KdaB,
    /// 2026-10-08: KDA `f_a_proj`: the decay gate's low-rank down projection.
    KdaFA,
    /// 2026-10-08: KDA `f_b_proj`: the decay gate's low-rank up projection, one input per
    /// channel.
    KdaFB,
    /// 2026-10-08: KDA `g_a_proj`: the output gate's low-rank down projection.
    KdaGA,
    /// 2026-10-08: KDA `g_b_proj`: the output gate's low-rank up projection.
    KdaGB,
    /// 2026-10-08: MLA `q_a_proj`: the hidden row into the query latent (`q_lora_rank`).
    MlaQA,
    /// 2026-10-08: MLA `q_b_proj`: the normed query latent into the per-head queries.
    MlaQB,
    /// 2026-10-08: MLA `kv_a_proj_with_mqa`: the hidden row into the shared KV latent.
    MlaKvA,
    /// 2026-10-08: Sparse-attention indexer `wq_b`: the query latent into the indexer heads.
    IndexQ,
    /// 2026-10-08: Indexer `wk`: the hidden row into the indexer key.
    IndexK,
    /// 2026-10-08: Indexer `weights_proj`: one weight per indexer head.
    IndexWeights,
    /// 2026-10-08: Indexer `index_kpool_compress_gate`: the per-channel pool gate scores.
    IndexGate,
    /// 2026-10-08: Hyper-connection `hc_*_fn`: the flattened residual streams into the
    /// `(2 + hc) * hc` pre, post and combination mixes.
    HcMix,
}

const ROLES: [(LinearRole, &str); 31] = [
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
    (LinearRole::KdaB, "kda_b"),
    (LinearRole::KdaFA, "kda_f_a"),
    (LinearRole::KdaFB, "kda_f_b"),
    (LinearRole::KdaGA, "kda_g_a"),
    (LinearRole::KdaGB, "kda_g_b"),
    (LinearRole::MlaQA, "mla_q_a"),
    (LinearRole::MlaQB, "mla_q_b"),
    (LinearRole::MlaKvA, "mla_kv_a"),
    (LinearRole::IndexQ, "index_q"),
    (LinearRole::IndexK, "index_k"),
    (LinearRole::IndexWeights, "index_weights"),
    (LinearRole::IndexGate, "index_gate"),
    (LinearRole::HcMix, "hc_mix"),
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
