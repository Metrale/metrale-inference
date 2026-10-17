// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: LoRA environment switches (`METRALE_LORA_*`), the
//! full-attention layer list, and `validate_peft_config`, the adapter-config
//! check that needs `--max-lora-rank`.
//!
//! Owner: model-layers (lora).
//! Invariants: 2026-10-17: the model-crate readers take the resolved settings; the two
//! `*_env` readers below are kept only for crates/server, which does not hold the settings yet.

use anyhow::{Result, bail};
use metrale_config::{LayerType, ModelConfig, PeftAdapterConfig};
use metrale_settings::InferenceServerSettings;

use super::LoraModule;

/// 2026-09-25: `METRALE_LORA_ROTATE` (`1` or `true`, any case). It, or `lora_peer_env`, sets the
/// model's `lora_rotatable`, which permits runtime adapter rotation and swap whatever the number
/// of resident adapters. It does not make decode eager; only `METRALE_LORA_EAGER` does.
/// `met serve` refuses `--lora-stageable-disk` unless this or a peer is set. 2026-10-17: The
/// model reads `model.lora_rotate`; this reader is the server's until it holds the settings.
pub fn lora_rotate_env() -> bool {
    std::env::var("METRALE_LORA_ROTATE").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// 2026-09-25: `METRALE_LORA_PEER`, the host:port of the weight peer that
/// stages adapters, when set and non-empty. Setting it also sets the model's
/// `lora_rotatable` (see `lora_rotate_env`). 2026-10-17: The server's reader; the model
/// reads [`lora_peer`].
pub fn lora_peer_env() -> Option<String> {
    std::env::var("METRALE_LORA_PEER")
        .ok()
        .filter(|s| !s.is_empty())
}

/// 2026-10-17: [`lora_peer_env`] from the resolved `storage.lora_peer`.
pub fn lora_peer(s: &InferenceServerSettings) -> Option<&str> {
    s.storage.lora_peer.as_deref().filter(|s| !s.is_empty())
}

/// 2026-10-17: The routed-expert and router LoRA policy, resolved once from the settings and
/// passed to every adapter audit (`expert_pack::validate`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertLoraPolicy {
    /// 2026-09-25: `METRALE_LORA_EXPERTS` (`kernels.lora_experts`): load routed-expert and router
    /// LoRA deltas. Off, an adapter with expert or router tensors is refused at load.
    pub enabled: bool,
    /// 2026-09-25: The padded rank of the expert and router LoRA pools,
    /// `METRALE_LORA_EXPERT_RANK` (a positive integer; otherwise 16). It is separate from
    /// `--max-lora-rank`, which sizes the slot pool. An adapter whose `r` exceeds it is refused.
    pub max_rank: usize,
}

impl ExpertLoraPolicy {
    /// 2026-10-17: The policy from `kernels.lora_experts` and `kernels.lora_expert_rank`.
    #[must_use]
    pub fn from_settings(s: &InferenceServerSettings) -> Self {
        Self {
            enabled: s.kernels.lora_experts,
            max_rank: s
                .kernels
                .lora_expert_rank
                .as_deref()
                .and_then(|v| v.parse().ok())
                .filter(|&r: &usize| r > 0)
                .unwrap_or(16),
        }
    }
}

/// 2026-09-25: `METRALE_LORA_PREFILL_BGMV=1`: the attention QKV
/// prefill applies LoRA with the per-row `apply_lora_bgmv` instead of
/// `apply_lora_delta` when the call has a slot buffer and a route
/// (`layers/qwen3_attention/prefill/paged_qkv.rs`).
pub fn prefill_bgmv_forced(s: &InferenceServerSettings) -> bool {
    s.scheduler.lora_prefill_bgmv
}

/// 2026-09-25: `METRALE_LORA_NO_BATCH_VERIFY=1`: while an adapter
/// is loaded, cross-sequence batched verify is not eligible (model-engine
/// `trait_impl/verify_e.rs`).
pub fn no_batch_verify(s: &InferenceServerSettings) -> bool {
    s.speculative.lora_no_batch_verify
}

pub fn full_attention_layers(cfg: &ModelConfig) -> Vec<usize> {
    (0..cfg.num_hidden_layers)
        .filter(|&i| cfg.layer_type(i) == LayerType::FullAttention)
        .collect()
}

/// 2026-09-25: Adapter-config checks that need `--max-lora-rank`: the rank
/// fits the pool, and every target module is one `LoraModule` or the router
/// (`gate`). The parse-time checks are in `metrale_config::parse_peft_adapter_config`.
pub fn validate_peft_config(peft: &PeftAdapterConfig, max_lora_rank: usize) -> Result<()> {
    if peft.r > max_lora_rank {
        bail!(
            "REJECT[rank-exceeds-pool]: r={} > --max-lora-rank={}",
            peft.r,
            max_lora_rank
        );
    }
    let mut unsupported: Vec<&str> = Vec::new();
    for t in &peft.target_modules {
        let last = t.rsplit('.').next().unwrap_or(t);
        // 2026-09-25: `gate` is the MoE router, not `gate_proj`. Expert
        // projections use the dense leaf names, which `LoraModule::ALL` lists.
        let ok = last == "gate" || LoraModule::ALL.iter().any(|m| m.peft_name() == last);
        if !ok {
            unsupported.push(t.as_str());
        }
    }
    if !unsupported.is_empty() {
        if !allow_partial_targets() {
            bail!(
                "REJECT[unsupported-target]: target_modules {unsupported:?} \
                 (allowed: q_proj k_proj v_proj o_proj gate_proj up_proj down_proj gate). \
                 Set METRALE_LORA_ALLOW_PARTIAL=1 to load anyway, applying only the \
                 supported modules — the adapter will then be PARTIALLY applied and \
                 will not reproduce its training behaviour."
            );
        }
        tracing::warn!(
            "LoRA PARTIAL LOAD (METRALE_LORA_ALLOW_PARTIAL=1): target_modules \
             {unsupported:?} are NOT supported and will be SKIPPED. Their \
             trained deltas will not be applied; output will differ from the \
             adapter's intent. Supported: q_proj k_proj v_proj o_proj \
             gate_proj up_proj down_proj gate."
        );
    }
    Ok(())
}

/// 2026-09-25: `METRALE_LORA_ALLOW_PARTIAL` (`1` or `true`, any case): load an
/// adapter that names target modules the engine cannot apply, skipping those.
/// Re-exported from metrale-config, whose parse-time check reads the same
/// switch, so both checks see one value.
pub use metrale_config::allow_partial_targets;

#[cfg(test)]
#[path = "env_settings_tests.rs"]
mod settings_tests;
