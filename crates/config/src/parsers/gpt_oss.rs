// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Strict pinned GPT-OSS-20B runtime configuration, not loader support.
use crate::{
    GptOssActivation, GptOssAttention, GptOssExpertBias, GptOssExpertFormat, GptOssNorm,
    GptOssPolicy, GptOssRope, GptOssRouting, LayerType, ModelConfig, finalize_config,
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};

fn exact(raw: &Value, key: &str, expected: Value) -> Result<()> {
    let value = raw
        .get(key)
        .with_context(|| format!("gpt_oss requires `{key}`"))?;
    let equal = if expected.is_number() {
        value.as_f64() == expected.as_f64()
    } else {
        value == &expected
    };
    ensure!(
        equal,
        "gpt_oss unsupported `{key}`: expected {expected}, got {value}"
    );
    Ok(())
}

fn known_keys(raw: &Value, allowed: &[&str], path: &str) -> Result<()> {
    let object = raw
        .as_object()
        .with_context(|| format!("gpt_oss {path} must be an object"))?;
    for key in object.keys() {
        ensure!(
            allowed.contains(&key.as_str()),
            "gpt_oss unsupported {path} field `{key}`"
        );
    }
    Ok(())
}

pub(crate) fn parse_gpt_oss(raw: &Value) -> Result<ModelConfig> {
    known_keys(
        raw,
        &[
            "architectures",
            "attention_bias",
            "attention_dropout",
            "eos_token_id",
            "experts_per_token",
            "head_dim",
            "hidden_act",
            "hidden_size",
            "initial_context_length",
            "initializer_range",
            "intermediate_size",
            "layer_types",
            "max_position_embeddings",
            "model_type",
            "num_attention_heads",
            "num_experts_per_tok",
            "num_hidden_layers",
            "num_key_value_heads",
            "num_local_experts",
            "num_experts",
            "output_router_logits",
            "pad_token_id",
            "quantization_config",
            "rms_norm_eps",
            "rope_scaling",
            "rope_theta",
            "router_aux_loss_coef",
            "sliding_window",
            "swiglu_limit",
            "tie_word_embeddings",
            "transformers_version",
            "use_cache",
            "vocab_size",
        ],
        "config",
    )?;
    for (key, value) in [
        ("hidden_size", 2880),
        ("num_hidden_layers", 24),
        ("intermediate_size", 2880),
        ("vocab_size", 201088),
        ("num_attention_heads", 64),
        ("num_key_value_heads", 8),
        ("head_dim", 64),
        ("num_local_experts", 32),
        ("num_experts_per_tok", 4),
        ("experts_per_token", 4),
        ("initial_context_length", 4096),
        ("max_position_embeddings", 131072),
        ("sliding_window", 128),
    ] {
        ensure!(
            raw[key].as_u64() == Some(value),
            "gpt_oss requires pinned integer `{key}` = {value}"
        );
    }
    if raw.get("num_experts").is_some() {
        exact(raw, "num_experts", json!(32))?;
    }
    for (key, value) in [
        ("rope_theta", 150000.0),
        ("rms_norm_eps", 1e-5),
        ("swiglu_limit", 7.0),
        ("attention_dropout", 0.0),
    ] {
        exact(raw, key, json!(value))?;
    }
    for (key, value) in [
        ("model_type", json!("gpt_oss")),
        ("architectures", json!(["GptOssForCausalLM"])),
        ("hidden_act", json!("silu")),
        ("attention_bias", json!(true)),
        ("tie_word_embeddings", json!(false)),
    ] {
        exact(raw, key, value)?;
    }
    ensure!(
        raw["eos_token_id"].as_u64() == Some(200002),
        "gpt_oss requires pinned assistant-turn EOS 200002"
    );
    ensure!(
        raw["pad_token_id"].as_u64() == Some(199999),
        "gpt_oss requires pinned pad token 199999"
    );
    let rope = &raw["rope_scaling"];
    known_keys(
        rope,
        &[
            "beta_fast",
            "beta_slow",
            "factor",
            "original_max_position_embeddings",
            "rope_type",
            "truncate",
        ],
        "rope_scaling",
    )?;
    for (key, value) in [
        ("beta_fast", 32.0),
        ("beta_slow", 1.0),
        ("factor", 32.0),
        ("original_max_position_embeddings", 4096.0),
    ] {
        exact(rope, key, json!(value))?;
    }
    exact(rope, "rope_type", json!("yarn"))?;
    exact(rope, "truncate", json!(false))?;
    let expected_layers: Vec<_> = (0..24)
        .map(|i| {
            if i % 2 == 0 {
                "sliding_attention"
            } else {
                "full_attention"
            }
        })
        .collect();
    exact(raw, "layer_types", json!(expected_layers))?;
    // 2026-10-07: Precision parser validates the complete MXFP4 declaration, including exclusions.
    exact(&raw["quantization_config"], "quant_method", json!("mxfp4"))?;
    let mut config: ModelConfig =
        serde_json::from_value(raw.clone()).context("gpt_oss runtime config")?;
    config.num_experts = raw["num_local_experts"]
        .as_u64()
        .context("num_local_experts")? as usize;
    config.moe_intermediate_size = config.intermediate_size;
    config.norm_topk_prob = true;
    config.scoring_func = "softmax".into();
    config.attn_gated = false;
    config.qk_norm_type = "none".into();
    config.weight_prefix = "model".into();
    config.yarn_factor = rope["factor"].as_f64().context("factor")? as f32;
    config.yarn_beta_fast = rope["beta_fast"].as_f64().context("beta_fast")? as f32;
    config.yarn_beta_slow = rope["beta_slow"].as_f64().context("beta_slow")? as f32;
    config.yarn_original_max_position_embeddings = rope["original_max_position_embeddings"]
        .as_u64()
        .context("original_max_position_embeddings must be an integer")?
        as usize;
    config.yarn_attention_factor = (1.0 + 0.1 * f64::from(config.yarn_factor).ln()) as f32;
    config.gpt_oss = Some(GptOssPolicy {
        routing: GptOssRouting::TopKLogitsThenSoftmax,
        attention: GptOssAttention::BiasedGqaWithDenominatorSink,
        activation: GptOssActivation::InterleavedAsymmetricSwiGlu { alpha: 1.702 },
        expert_bias: GptOssExpertBias::GateUpAndDownBeforeRoutingReduction,
        norm: GptOssNorm::Fp32NormalizationAndScaleBeforeCast,
        rope: GptOssRope::HalfSplitYarnWithoutTruncatedCorrectionRange,
        expert_format: GptOssExpertFormat::Mxfp4E2m1Group32E8m0,
    });
    ensure!(
        config.layer_types.first() == Some(&LayerType::SlidingAttention),
        "gpt_oss first layer must slide"
    );
    finalize_config(&mut config, raw)?;
    Ok(config)
}

#[cfg(test)]
#[path = "gpt_oss_tests.rs"]
mod tests;
