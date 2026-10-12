// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Pinned checkpoint and changed-math refusal controls.
use crate::precision_plan::{Granularity, PlanSource};
use crate::{GptOssActivation, GptOssRope, GptOssRouting, LayerType, parse_config};
const FIXTURE: &str =
    include_str!("../../../circuit/tests/fixtures/checkpoints/openai--gpt-oss-20b/config.json");

#[test]
fn gpt_oss_pinned_shape_and_policies_survive_runtime_parse() {
    let c = parse_config(FIXTURE).unwrap();
    assert_eq!(
        (c.hidden_size, c.num_hidden_layers, c.head_dim),
        (2880, 24, 64)
    );
    assert_eq!((c.num_attention_heads, c.num_key_value_heads), (64, 8));
    assert_eq!(
        (
            c.num_experts,
            c.num_experts_per_tok,
            c.moe_intermediate_size
        ),
        (32, 4, 2880)
    );
    assert_eq!(c.layer_types[0], LayerType::SlidingAttention);
    assert_eq!(c.layer_types[1], LayerType::FullAttention);
    assert_eq!(c.sliding_window, 128);
    assert!(!c.attn_gated);
    assert!(!c.tie_word_embeddings);
    assert_eq!(c.swiglu_limit, 7.0);
    assert_eq!(c.yarn_factor, 32.0);
    assert_eq!(c.yarn_original_max_position_embeddings, 4096);
    assert_eq!(c.yarn_attention_factor, 1.0 + 0.1 * 32_f32.ln());
    assert_eq!(c.eos_ids(), &[200002]);
    let policy = c.gpt_oss.unwrap();
    assert_eq!(policy.routing, GptOssRouting::TopKLogitsThenSoftmax);
    assert_eq!(
        policy.activation,
        GptOssActivation::InterleavedAsymmetricSwiGlu { alpha: 1.702 }
    );
    assert_eq!(
        policy.rope,
        GptOssRope::HalfSplitYarnWithoutTruncatedCorrectionRange
    );
    let q = c.quantization_config.unwrap();
    assert_eq!(q.precision.source, PlanSource::Mxfp4);
    for name in [
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.23.mlp.experts.down_proj",
    ] {
        let p = q.precision.resolve(name);
        assert_eq!(p.weight.unwrap().granularity, Granularity::Group(32));
        assert!(p.activation.is_none());
    }
    for name in [
        "model.layers.0.self_attn.q_proj",
        "model.layers.1.mlp.router",
        "lm_head",
        "model.embed_tokens",
    ] {
        assert!(q.precision.resolve(name).weight.is_none());
    }
}

#[test]
fn gpt_oss_missing_required_math_is_not_defaulted() {
    for field in [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "num_local_experts",
        "experts_per_token",
        "num_experts_per_tok",
        "intermediate_size",
        "vocab_size",
        "attention_bias",
        "attention_dropout",
        "hidden_act",
        "layer_types",
        "initial_context_length",
        "max_position_embeddings",
        "rms_norm_eps",
        "rope_scaling",
        "rope_theta",
        "sliding_window",
        "swiglu_limit",
        "tie_word_embeddings",
        "quantization_config",
        "eos_token_id",
    ] {
        let mut d: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
        d.as_object_mut().unwrap().remove(field);
        assert!(
            parse_config(&d.to_string()).is_err(),
            "missing {field} accepted"
        );
    }
}

#[test]
fn gpt_oss_changed_math_and_conflicting_aliases_fail_closed() {
    for (pointer, value) in [
        ("/attention_bias", serde_json::json!(false)),
        ("/experts_per_token", serde_json::json!(8)),
        ("/num_experts", serde_json::json!(256)),
        ("/num_local_experts", serde_json::json!(128)),
        ("/swiglu_limit", serde_json::json!(0)),
        ("/rope_scaling/truncate", serde_json::json!(true)),
        ("/rope_scaling/factor", serde_json::json!(8.0)),
        (
            "/quantization_config/quant_method",
            serde_json::json!("nvfp4"),
        ),
        ("/eos_token_id", serde_json::json!([200002, 200007])),
    ] {
        let mut d: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
        if pointer == "/num_experts" {
            d["num_experts"] = value;
        } else {
            *d.pointer_mut(pointer).unwrap() = value;
        }
        assert!(
            parse_config(&d.to_string()).is_err(),
            "accepted mutation {pointer}"
        );
    }
    let mut d: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    d["layer_types"][0] = "linear_attention".into();
    assert!(parse_config(&d.to_string()).is_err());
    d = serde_json::from_str(FIXTURE).unwrap();
    d["num_experts"] = 32.into();
    assert_eq!(parse_config(&d.to_string()).unwrap().num_experts, 32);
}

#[test]
fn gpt_oss_unknown_policies_are_not_silently_ignored() {
    for (object, key) in [
        ("", "attention_sinks"),
        ("rope_scaling", "attention_factor"),
        ("quantization_config", "group_size"),
    ] {
        let mut d: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
        if object.is_empty() {
            d[key] = false.into();
        } else {
            d[object][key] = false.into();
        }
        assert!(parse_config(&d.to_string()).is_err());
    }
    assert!(crate::ModelConfig::qwen3_next_80b_nvfp4().gpt_oss.is_none());
}

#[test]
fn gpt_oss_common_consumers_keep_bf16_head_and_reasoning_capability() {
    let mut c = parse_config(FIXTURE).unwrap();
    assert!(c.skip_lm_head_quantization());
    assert!(c.capabilities().supports_thinking);
    assert!(!c.capabilities().has_ssm_layers);
    assert!(c.capabilities().has_moe_layers);
    c.quantization_config
        .as_mut()
        .unwrap()
        .precision
        .ignore
        .retain(|target| target.text() != "lm_head");
    let expert_precision = c
        .quantization_config
        .as_ref()
        .unwrap()
        .precision
        .resolve("model.layers.0.mlp.experts.down_proj");
    c.quantization_config
        .as_mut()
        .unwrap()
        .precision
        .rules
        .push(crate::precision_plan::Rule {
            targets: vec![crate::precision_plan::Target::Exact("lm_head".into())],
            precision: expert_precision,
        });
    assert!(
        !c.skip_lm_head_quantization(),
        "consumer follows declared head precision"
    );
    c.lm_head_bf16_override = Some(true);
    assert!(c.skip_lm_head_quantization());
    c.lm_head_bf16_override = Some(false);
    assert!(
        !c.skip_lm_head_quantization(),
        "explicit operator downcast remains explicit"
    );
}
