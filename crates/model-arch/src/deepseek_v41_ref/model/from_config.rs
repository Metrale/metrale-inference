// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: `ModelCfg` from the engine's parsed `ModelConfig`, so the CPU reference and the
//! GPU path read one config (the `deepseek_v41` parser) rather than two.
//!
//! Owner: model-arch, DeepSeek-V4.1 reference.
//! Invariants: only the backbone layers are kept (`compress_ratios` also lists the MTP
//! layers that follow them).

use metrale_config::ModelConfig;

use super::super::moe::MoeCfg;
use super::{ModelCfg, Semantics};

/// 2026-10-19: The router temperature. `config.json` has no key for it, so the reference's
/// `ModelArgs.gate_temp` declaration is what the checkpoint runs with.
pub const REFERENCE_GATE_TEMP: f32 = 1.0;

impl ModelCfg {
    /// 2026-10-19: `max_seq` is a runtime limit (it sizes the caches), not model shape.
    pub fn from_model_config(m: &ModelConfig, max_seq: usize, semantics: Semantics) -> Self {
        let n_layers = m.num_hidden_layers;
        ModelCfg {
            dim: m.hidden_size,
            hc: m.hc_mult,
            iters: m.hc_sinkhorn_iters,
            hc_eps: m.hc_eps,
            eps: m.rms_norm_eps as f32,
            n_layers,
            vocab: m.vocab_size,
            max_seq,
            n_heads: m.num_attention_heads,
            head_dim: m.head_dim,
            rope_dim: m.qk_rope_head_dim,
            q_rank: m.q_lora_rank,
            o_rank: m.o_lora_rank,
            groups: m.o_groups,
            window: m.sliding_window as usize,
            ratios: m.compress_ratios[..n_layers].to_vec(),
            kv_sources: m.kv_source_layer_ids.clone(),
            index_sources: m.index_source_layer_ids.clone(),
            cand_src: m.candidate_source_layer_id.map_or(-1, |l| l as i64),
            cand_topk_blocks: m.candidate_topk_blocks,
            cand_block: m.candidate_block_size,
            index_heads: m.index_n_heads,
            index_hd: m.index_head_dim,
            index_topk: m.index_topk,
            rope_theta: m.rope_theta as f32,
            compress_rope_theta: m.compress_rope_theta,
            rope_factor: m.yarn_factor,
            orig_seq: m.yarn_original_max_position_embeddings,
            beta_fast: m.yarn_beta_fast,
            beta_slow: m.yarn_beta_slow,
            moe: MoeCfg {
                dim: m.hidden_size,
                inter: m.moe_intermediate_size,
                n_routed: m.n_routed_experts,
                topk: m.num_experts_per_tok,
                gate_temp: REFERENCE_GATE_TEMP,
                norm_topk_prob: m.norm_topk_prob,
                route_scale: m.routed_scaling_factor as f32,
                swiglu_limit: m.swiglu_limit,
            },
            semantics,
        }
    }
}

#[cfg(test)]
#[path = "from_config_tests.rs"]
mod tests;
