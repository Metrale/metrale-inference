// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The shipped `kernel_caps()` against Qwen/Qwen3.6-35B-A3B-FP8's own
//! `quantization_config` under `declared`: 2026-09-29, the MoE expert W8A8 family is on
//! (validated by the full BFCL draw and agentic-webserver), as the dense per-channel family is
//! for unsloth/Qwen3.8-27B-NVFP4, and so is the block-scaled attention/GDN W8A8 family; with the
//! model's `expert_down_w8a16` exception the expert down projections decode W8A16 while still
//! declaring FP8 activations; the `nvfp4` tier asks for none of them.

use metrale_config::weight_quantization::{AboveDeclared, ActFormat};
use metrale_config::{
    DeclaredPrecisionPlan, W4a4Downcast, WeightQuantPolicy, WeightQuantTier, WeightQuantization,
};

fn plan(text: &str) -> DeclaredPrecisionPlan {
    let raw: serde_json::Value = serde_json::from_str(text).expect("fixture parses");
    DeclaredPrecisionPlan::from_quantization_config(&raw).expect("plan")
}

const L: &str = "model.language_model.layers";

#[test]
fn shipped_caps_run_every_w8a8_family_under_declared() {
    let caps = metrale_model_layers::layers::kernel_caps();
    let declared =
        WeightQuantTier::new(WeightQuantization::Declared, W4a4Downcast::Off).expect("tier");
    let moe = plan(include_str!(
        "../../config/src/precision_plan/fixtures/qwen3_6_35b_a3b_fp8.json"
    ));
    let pol = WeightQuantPolicy::new(declared, &moe, caps);
    let nvfp4 = WeightQuantTier::new(WeightQuantization::Nvfp4, W4a4Downcast::Off).expect("tier");
    let nv = WeightQuantPolicy::new(nvfp4, &moe, caps);
    for m in [
        format!("{L}.0.mlp.experts.0.gate_proj"),
        format!("{L}.0.mlp.shared_expert.up_proj"),
    ] {
        assert_eq!(
            pol.fp8_block_scaled_decode_act(&m),
            Some(ActFormat::Fp8),
            "{m}"
        );
        assert!(pol.declares_fp8_activations(&m), "{m}");
        assert_eq!(nv.fp8_block_scaled_decode_act(&m), None, "{m}");
    }
    // 2026-09-29: The block-scaled attention/GDN projections decode FP8 (W8A8) under
    // `declared`; `nvfp4` asks for nothing.
    for m in [
        format!("{L}.3.self_attn.q_proj"),
        format!("{L}.0.linear_attn.in_proj_qkv"),
    ] {
        assert_eq!(
            pol.fp8_block_scaled_decode_act(&m),
            Some(ActFormat::Fp8),
            "{m}"
        );
        assert_eq!(nv.fp8_block_scaled_decode_act(&m), None, "{m}");
    }
    // 2026-09-29: The 35B's `expert_down_w8a16`: its expert down projections decode BF16 and
    // still declare FP8; gate and up stay FP8.
    let above = pol.with_above_declared(AboveDeclared {
        expert_down_w8a16: true,
    });
    for m in [
        format!("{L}.0.mlp.experts.0.down_proj"),
        format!("{L}.0.mlp.shared_expert.down_proj"),
    ] {
        assert_eq!(above.fp8_decode_act(&m), Some(ActFormat::Bf16), "{m}");
        assert!(above.declares_fp8_activations(&m), "{m}");
    }
    assert_eq!(
        above.fp8_decode_act(&format!("{L}.0.mlp.experts.0.gate_proj")),
        Some(ActFormat::Fp8)
    );
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
