// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `config.json` parser for `model_type = "laguna"` (Poolside Laguna).
//!
//! Owner: config (model parsers).
//! Invariants:
//! - A parse that succeeds has a YaRN full-attention rope and a default sliding-attention rope
//!   with `partial_rotary_factor` 1.0, and every per-layer Q-head count is a multiple of
//!   `num_key_value_heads`; `num_attention_heads` is the largest of them.
//! - 2026-10-07: An integer (compressed-tensors `pack-quantized`) checkpoint parses only when
//!   [`admit_packed_int`] admits its block and only routed experts are integer-quantized.

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use super::super::{ModelConfig, finalize_config};
use crate::LayerPrecision;
use crate::precision_plan::packed_int::{admit_packed_int, routed_expert_scheme};

fn required<'a>(raw: &'a Value, key: &str) -> Result<&'a Value> {
    raw.get(key)
        .with_context(|| format!("laguna config missing required field `{key}`"))
}

fn required_f64(raw: &Value, key: &str) -> Result<f64> {
    required(raw, key)?
        .as_f64()
        .with_context(|| format!("laguna config field `{key}` must be numeric"))
}

pub(crate) fn parse_laguna(raw: &Value) -> Result<ModelConfig> {
    let mut normalized = raw.clone();
    let object = normalized
        .as_object_mut()
        .context("laguna config.json must be an object")?;

    let eos = required(raw, "eos_token_id")?;
    let primary_eos = match eos {
        Value::Number(n) => n.as_u64(),
        Value::Array(ids) => ids.first().and_then(Value::as_u64),
        _ => None,
    }
    .context("laguna eos_token_id must be an integer or non-empty integer array")?;
    object.insert("eos_token_id".into(), Value::from(primary_eos));

    let mut config: ModelConfig =
        serde_json::from_value(normalized).context("Failed to parse laguna config.json")?;
    ensure!(
        config.hidden_size > 0,
        "laguna hidden_size must be non-zero"
    );
    ensure!(config.head_dim > 0, "laguna head_dim must be non-zero");
    ensure!(
        config.num_key_value_heads > 0,
        "laguna num_key_value_heads must be non-zero"
    );
    ensure!(
        config.num_experts > 0,
        "laguna num_experts must be non-zero"
    );
    ensure!(
        config.num_experts_per_tok > 0,
        "laguna num_experts_per_tok must be non-zero"
    );
    let max_heads = config
        .num_attention_heads_per_layer
        .iter()
        .copied()
        .max()
        .context("laguna num_attention_heads_per_layer must not be empty")?;
    ensure!(
        config
            .num_attention_heads_per_layer
            .iter()
            .all(|heads| heads.is_multiple_of(config.num_key_value_heads)),
        "every laguna Q-head count must be divisible by num_key_value_heads"
    );
    config.num_attention_heads = max_heads;

    let full_rope = required(raw, "rope_parameters")?
        .get("full_attention")
        .context("laguna rope_parameters missing full_attention")?;
    ensure!(
        full_rope.get("rope_type").and_then(Value::as_str) == Some("yarn"),
        "laguna full_attention rope_type must be yarn"
    );
    config.rope_theta = required_f64(full_rope, "rope_theta")?;
    config.partial_rotary_factor = required_f64(full_rope, "partial_rotary_factor")?;
    config.yarn_factor = required_f64(full_rope, "factor")? as f32;
    config.yarn_beta_slow = required_f64(full_rope, "beta_slow")? as f32;
    config.yarn_beta_fast = required_f64(full_rope, "beta_fast")? as f32;
    config.yarn_original_max_position_embeddings =
        required(full_rope, "original_max_position_embeddings")?
            .as_u64()
            .context("laguna original_max_position_embeddings must be an integer")?
            as usize;
    config.yarn_attention_factor = required_f64(full_rope, "attention_factor")? as f32;

    let sliding_rope = required(raw, "rope_parameters")?
        .get("sliding_attention")
        .context("laguna rope_parameters missing sliding_attention")?;
    ensure!(
        sliding_rope.get("rope_type").and_then(Value::as_str) == Some("default"),
        "laguna sliding_attention rope_type must be default"
    );
    ensure!(
        required_f64(sliding_rope, "partial_rotary_factor")? == 1.0,
        "laguna sliding_attention partial_rotary_factor must be 1.0"
    );

    config.routed_scaling_factor = required_f64(raw, "moe_routed_scaling_factor")?;
    config.scoring_func = "sigmoid".to_string();
    config.use_routing_bias = true;
    config.qk_norm_type = "per_head".to_string();
    config.attn_gated = false;
    config.weight_prefix.clear();
    config.mtp_num_hidden_layers = 0;
    config.num_mtp_modules = 0;
    config.mtp_transformer_layers = 0;

    finalize_config(&mut config, raw)?;
    admit_packed_int_layout(&config, raw)?;
    Ok(config)
}

/// 2026-10-07: A compressed-tensors integer checkpoint (Laguna-XS-2.1-INT4) is admitted only
/// in the layout the packed-int expert kernels serve: every MoE layer's routed experts share
/// one admitted INT4/INT8 g128 scheme, and every other linear module (attention, the head
/// gate, the dense layer, the router, the shared expert, lm_head) is unquantized. Float and
/// unquantized checkpoints pass through unchanged.
fn admit_packed_int_layout(config: &ModelConfig, raw: &Value) -> Result<()> {
    let Some(qc) = raw.get("quantization_config") else {
        return Ok(());
    };
    if admit_packed_int(qc)?.is_empty() {
        return Ok(());
    }
    let plan = &config
        .quantization_config
        .as_ref()
        .context("laguna integer checkpoint without a parsed quantization_config")?
        .precision;
    let unquantized = |module: String| -> Result<()> {
        let p = plan.resolve(&module);
        ensure!(
            p == LayerPrecision::UNQUANTIZED,
            "laguna {module} declares {}; only routed experts may be integer-quantized",
            p.label()
        );
        Ok(())
    };
    unquantized("lm_head".into())?;
    for i in 0..config.num_hidden_layers {
        let lp = format!("model.layers.{i}");
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj", "g_proj"] {
            unquantized(format!("{lp}.self_attn.{proj}"))?;
        }
        let mlp = format!("{lp}.mlp");
        if config.mlp_only_layers.contains(&i) {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                unquantized(format!("{mlp}.{proj}"))?;
            }
            continue;
        }
        unquantized(format!("{mlp}.gate"))?;
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            unquantized(format!("{mlp}.shared_expert.{proj}"))?;
        }
        routed_expert_scheme(plan, &mlp, config.num_experts)?
            .with_context(|| format!("laguna {mlp} routed experts are not integer-quantized"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::LayerType;

    const CONFIG: &str = r#"{
        "model_type": "laguna",
        "hidden_size": 3072,
        "intermediate_size": 12288,
        "num_hidden_layers": 2,
        "vocab_size": 100352,
        "num_attention_heads": 48,
        "num_attention_heads_per_layer": [48, 72],
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_experts": 256,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "norm_topk_prob": true,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [0],
        "layer_types": ["full_attention", "sliding_attention"],
        "sliding_window": 512,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 0.000001,
        "bos_token_id": 2,
        "eos_token_id": [2, 24],
        "tie_word_embeddings": false,
        "gating": "per-head",
        "moe_routed_scaling_factor": 2.5,
        "rope_parameters": {
            "full_attention": {
                "rope_type": "yarn",
                "rope_theta": 500000.0,
                "factor": 32.0,
                "original_max_position_embeddings": 8192,
                "beta_slow": 1.0,
                "beta_fast": 32.0,
                "attention_factor": 1.3465735902799727,
                "partial_rotary_factor": 0.5
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0
            }
        }
    }"#;

    fn parse_fixture() -> crate::ModelConfig {
        crate::parse_config(CONFIG).expect("parse laguna")
    }

    #[test]
    fn parses_heterogeneous_attention_layout() {
        let config = parse_fixture();
        assert_eq!(config.model_type, "laguna");
        assert_eq!(config.num_attention_heads, 72);
        assert_eq!(config.num_attention_heads_per_layer, [48, 72]);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(
            config.layer_types,
            [LayerType::FullAttention, LayerType::SlidingAttention]
        );
        assert_eq!(config.sliding_window, 512);
        assert_eq!(config.qk_norm_type, "per_head");
    }

    #[test]
    fn parses_laguna_moe_contract() {
        let config = parse_fixture();
        assert_eq!(config.mlp_only_layers, [0]);
        assert_eq!(config.num_experts, 256);
        assert_eq!(config.num_experts_per_tok, 10);
        assert_eq!(config.moe_intermediate_size, 1024);
        assert_eq!(config.shared_expert_intermediate_size, 1024);
        assert!(config.norm_topk_prob);
        assert_eq!(config.scoring_func, "sigmoid");
        assert!(config.use_routing_bias);
        assert_eq!(config.routed_scaling_factor, 2.5);
    }

    #[test]
    fn parses_laguna_rope_and_eos_contract() {
        let config = parse_fixture();
        assert_eq!(config.rope_theta, 500000.0);
        assert_eq!(config.partial_rotary_factor, 0.5);
        assert_eq!(config.rotary_dim(), 64);
        assert_eq!(config.yarn_factor, 32.0);
        assert_eq!(config.yarn_beta_slow, 1.0);
        assert_eq!(config.yarn_beta_fast, 32.0);
        assert_eq!(config.yarn_original_max_position_embeddings, 8192);
        assert_eq!(config.yarn_attention_factor, 1.3465736);
        assert_eq!(config.eos_token_id, 2);
    }

    const INT4: &str = include_str!("../precision_plan/fixtures/poolside_laguna_xs_2_1_int4.json");

    fn int4_with(edit: impl FnOnce(&mut serde_json::Value)) -> anyhow::Result<crate::ModelConfig> {
        let mut raw: serde_json::Value = serde_json::from_str(INT4).expect("INT4 fixture");
        edit(&mut raw);
        crate::parse_config(&raw.to_string())
    }

    /// 2026-10-07: poolside/Laguna-XS-2.1-INT4 @ 4b7e28ab parses with its integer plan; the
    /// shape matches the XS target pin (model_type laguna, hidden_size 2048).
    #[test]
    fn parses_laguna_xs_int4_checkpoint() {
        let config = int4_with(|_| {}).expect("parse INT4 laguna");
        assert_eq!(config.model_type, "laguna");
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_experts_per_tok, 8);
        let qc = config.quantization_config.expect("quantization_config");
        assert_eq!(qc.quant_method, "compressed-tensors");
        assert_eq!(qc.quant_algo, "INT4");
        assert_eq!(qc.format, "pack-quantized");
    }

    #[test]
    fn rejects_int4_checkpoint_with_quantized_attention() {
        let error = int4_with(|raw| {
            raw["quantization_config"]["ignore"]
                .as_array_mut()
                .expect("ignore list")
                .retain(|e| e.as_str() != Some(r"re:.*\.self_attn\.q_proj$"));
            raw["quantization_config"]["config_groups"]["group_0"]["targets"]
                .as_array_mut()
                .expect("targets")
                .push(serde_json::json!(r"re:.*self_attn\.q_proj$"));
        })
        .expect_err("integer attention must be refused");
        assert!(
            format!("{error:#}").contains("model.layers.0.self_attn.q_proj declares W4A16"),
            "{error:#}"
        );
    }

    #[test]
    fn rejects_int4_checkpoint_with_unquantized_moe_layer() {
        let error = int4_with(|raw| {
            raw["quantization_config"]["config_groups"]
                .as_object_mut()
                .expect("groups")
                .remove("group_1");
        })
        .expect_err("BF16 routed experts have no packed-int kernel");
        assert!(
            format!("{error:#}").contains("model.layers.31.mlp routed experts"),
            "{error:#}"
        );
    }

    /// 2026-10-07: An exact-name target outranks group_0's pattern, so one expert of layer 5
    /// would be INT8 among INT4 siblings; one grouped kernel cannot serve that layer.
    #[test]
    fn rejects_int4_checkpoint_with_mixed_expert_schemes_in_a_layer() {
        let error = int4_with(|raw| {
            let groups = &mut raw["quantization_config"]["config_groups"];
            let mut one = groups["group_1"].clone();
            one["targets"] = serde_json::json!(["model.layers.5.mlp.experts.7.down_proj"]);
            groups["group_2"] = one;
        })
        .expect_err("mixed schemes in one layer must be refused");
        assert!(
            format!("{error:#}").contains("model.layers.5.mlp.experts.7.down_proj declares"),
            "{error:#}"
        );
    }

    #[test]
    fn rejects_int4_checkpoint_with_asymmetric_experts() {
        let error = int4_with(|raw| {
            raw["quantization_config"]["config_groups"]["group_1"]["weights"]["symmetric"] =
                serde_json::json!(false);
        })
        .expect_err("asymmetric experts must be refused");
        assert!(
            format!("{error:#}").contains("group_1.weights.symmetric"),
            "{error:#}"
        );
    }

    #[test]
    fn rejects_missing_per_layer_heads() {
        let mut invalid: serde_json::Value =
            serde_json::from_str(CONFIG).expect("parse Laguna fixture");
        invalid
            .as_object_mut()
            .expect("Laguna fixture is an object")
            .remove("num_attention_heads_per_layer");
        let error =
            crate::parse_config(&invalid.to_string()).expect_err("must reject missing heads");
        assert_eq!(
            error.to_string(),
            "laguna num_attention_heads_per_layer must not be empty"
        );
    }
}
