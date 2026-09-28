// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The shipped `kernel_caps()` against Qwen/Qwen3.6-35B-A3B-FP8's own
//! `quantization_config` under `declared`: the MoE expert and block-scaled attention/GDN W8A8
//! families stay off (W8A16) until the full BFCL draw validates them, while the dense
//! per-channel family is on for unsloth/Qwen3.8-27B-NVFP4.

use metrale_config::weight_quantization::ActFormat;
use metrale_config::{
    DeclaredPrecisionPlan, W4a4Downcast, WeightQuantPolicy, WeightQuantTier, WeightQuantization,
};

fn plan(text: &str) -> DeclaredPrecisionPlan {
    let raw: serde_json::Value = serde_json::from_str(text).expect("fixture parses");
    DeclaredPrecisionPlan::from_quantization_config(&raw).expect("plan")
}

const L: &str = "model.language_model.layers";

#[test]
fn shipped_caps_hold_the_moe_and_block_scaled_w8a8_and_keep_the_dense() {
    let caps = metrale_model_layers::layers::kernel_caps();
    let declared =
        WeightQuantTier::new(WeightQuantization::Declared, W4a4Downcast::Off).expect("tier");
    let moe = plan(include_str!(
        "../../config/src/precision_plan/fixtures/qwen3_6_35b_a3b_fp8.json"
    ));
    let pol = WeightQuantPolicy::new(declared, &moe, caps);
    for m in [
        format!("{L}.0.mlp.experts.0.gate_proj"),
        format!("{L}.0.mlp.shared_expert.up_proj"),
        format!("{L}.3.self_attn.q_proj"),
        format!("{L}.0.linear_attn.in_proj_qkv"),
    ] {
        assert_eq!(
            pol.fp8_block_scaled_decode_act(&m),
            Some(ActFormat::Bf16),
            "{m}"
        );
        assert!(pol.declares_fp8_activations(&m), "{m}");
    }
    let dense = plan(include_str!(
        "../../config/src/precision_plan/fixtures/unsloth_qwen3_8_27b_nvfp4.json"
    ));
    let d = WeightQuantPolicy::new(declared, &dense, caps);
    for m in [
        format!("{L}.3.self_attn.q_proj"),
        format!("{L}.0.linear_attn.in_proj_qkv"),
        format!("{L}.60.mlp.gate_proj"),
    ] {
        assert_eq!(d.fp8_decode_act(&m), Some(ActFormat::Fp8), "{m}");
    }
}
