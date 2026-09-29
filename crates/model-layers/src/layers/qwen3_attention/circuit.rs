// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `CircuitBindings` for `Qwen3AttentionLayer`: the full-attention layer's weights by
//! slot, its heads, RoPE, KV-cache layer and decode route, and the features the circuit does not
//! model.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants:
//! - Every arm of `decode_inner` / `attention_forward` other than the gated, per-head-norm,
//!   interleaved-MRoPE one the rules encode is reported in `unmodelled`.
//! - `paged_decode_plain` is the layer's own routing answer (`bf16_decode_is_plain`), never a
//!   restatement of it.

use std::collections::BTreeMap;

use metrale_circuit::LinearRole;

use super::Qwen3AttentionLayer;
use crate::circuit_exec::{
    AttnFacts, BoundWeight, CircuitBindings, CircuitLayer, MixerFacts, RopeFacts, WeightSlot,
};
use crate::weight_map::{DenseWeight, QuantWeight};

/// 2026-09-28: A Q/K/V projection as `attention_forward` reads it: the quantized weight when the
/// loader installed one, else the dense `attn` weight.
fn proj(
    w: Option<&QuantWeight>,
    dense: DenseWeight,
    what: &str,
    unmodelled: &mut Vec<String>,
) -> BoundWeight {
    match w {
        Some(QuantWeight::Nvfp4(q)) => BoundWeight::Nvfp4(*q),
        Some(QuantWeight::Dense(d)) => BoundWeight::Dense(*d),
        Some(QuantWeight::Fp8(_)) => {
            unmodelled.push(format!("an FP8 {what} projection"));
            BoundWeight::Dense(dense)
        }
        Some(QuantWeight::PackedQ2(_)) => {
            unmodelled.push(format!("a packed-Q2 {what} projection"));
            BoundWeight::Dense(dense)
        }
        None => BoundWeight::Dense(dense),
    }
}

impl CircuitBindings for Qwen3AttentionLayer {
    fn circuit_layer(
        &self,
        config: &metrale_config::ModelConfig,
        levers: &crate::layers::ops::ModelLevers,
    ) -> Option<CircuitLayer> {
        let mut unmodelled = Vec::new();
        let arms = [
            (self.hc.is_some(), "hyper-connections"),
            (self.lora.is_some(), "an attention LoRA adapter"),
            (self.mla.is_some(), "multi-head latent attention"),
            (self.qsa.is_some(), "a QSA indexer"),
            (self.v_norm_weight.is_some(), "a V norm"),
            (
                self.head_gate_weight.is_some(),
                "a per-head gate projection",
            ),
            (
                self.post_attn_out_norm.is_some(),
                "a post-attention output norm",
            ),
            (self.post_ffn_out_norm.is_some(), "a post-FFN output norm"),
            (self.layer_scalar.is_some(), "a layer scalar"),
            (self.moe_ffn.is_some(), "a second (MoE) FFN"),
            (
                self.shortcut_carry_in.is_some() || self.shortcut_carry_out.is_some(),
                "a shortcut carry",
            ),
            (
                self.attn.q_norm_full.is_some() || self.attn.k_norm_full.is_some(),
                "a full-width Q/K norm",
            ),
            (!self.yarn_inv_freq.is_null(), "YaRN RoPE"),
            (
                self.rope_proportional && self.rope_proportional_k.0 != 0,
                "proportional RoPE",
            ),
            (self.o_dense_bf16.is_some(), "a BF16 output projection"),
            (
                self.w8a8.is_some(),
                "declared W8A8 attention projections (not bound yet)",
            ),
        ];
        for (present, what) in arms {
            if present {
                unmodelled.push(what.to_string());
            }
        }
        let dense = BoundWeight::Dense;
        let mut weights = BTreeMap::from([
            (WeightSlot::InputNorm, dense(self.input_norm)),
            (WeightSlot::PostNorm, dense(self.post_attn_norm)),
            (WeightSlot::QNorm, dense(self.attn.q_norm)),
            (WeightSlot::KNorm, dense(self.attn.k_norm)),
        ]);
        let q = proj(
            self.q_weight.as_ref(),
            self.attn.q_proj,
            "q",
            &mut unmodelled,
        );
        let k = proj(
            self.k_weight.as_ref(),
            self.attn.k_proj,
            "k",
            &mut unmodelled,
        );
        let v = proj(
            self.v_weight.as_ref(),
            self.attn.v_proj,
            "v",
            &mut unmodelled,
        );
        weights.insert(WeightSlot::Linear(LinearRole::Q), q);
        weights.insert(WeightSlot::Linear(LinearRole::K), k);
        weights.insert(WeightSlot::Linear(LinearRole::V), v);
        if matches!(
            self.o_weight,
            Some(QuantWeight::Fp8(_)) | Some(QuantWeight::PackedQ2(_))
        ) {
            unmodelled.push("an FP8 or packed-Q2 output projection".to_string());
        }
        weights.insert(
            WeightSlot::Linear(LinearRole::O),
            BoundWeight::Nvfp4(self.attn.o_proj),
        );
        self.ffn.circuit_bind(&mut weights, &mut unmodelled);
        let nq = self
            .num_q_heads_override
            .unwrap_or(config.num_attention_heads) as u32;
        let nkv = self
            .num_kv_heads_override
            .unwrap_or(config.num_key_value_heads) as u32;
        let hd = self.head_dim_override.unwrap_or(config.head_dim) as u32;
        let facts = AttnFacts {
            attn_layer_idx: self.attn_layer_idx,
            kv_dtype: self.kv_dtype,
            num_q_heads: nq,
            num_kv_heads: nkv,
            head_dim: hd,
            gated: self.gated,
            rope: RopeFacts {
                mrope_interleaved: self.mrope_interleaved && self.rope_mrope_interleaved_k.0 != 0,
                theta: self.rope_theta_override.unwrap_or(config.rope_theta as f32),
                rotary_dim: self
                    .rotary_dim_override
                    .unwrap_or(config.rotary_dim() as u32),
            },
            sliding_window: self.sliding_window.unwrap_or(0),
            softmax_scale: self.effective_attn_scale(hd),
            paged_decode_plain: self.bf16_decode_is_plain(nq, nkv, hd, levers.max_decode_seqs),
        };
        Some(CircuitLayer {
            mixer: MixerFacts::Attention(facts),
            weights,
            unmodelled,
        })
    }
}
