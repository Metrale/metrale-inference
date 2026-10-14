// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: `shared_expert_declared_bf16` on the shape of Inferact/Qwen3.8-Flash-Next-NVFP4's
//! own `quantization_config` (ModelOpt NVFP4, the shared expert in `ignore`): BF16 under the
//! `declared` tier when all three projections are ignored, NVFP4 otherwise, and never under the
//! `nvfp4` tier; and `experts_declared_nvfp4` (2026-10-13), which marks the routed experts for
//! the grouped NVFP4 decode only where the plan is followed.

use metrale_config::weight_quantization::KernelCaps;
use metrale_config::{
    QuantizationConfig, W4a4Downcast, WeightQuantPolicy, WeightQuantTier, WeightQuantization,
};

use super::{experts_declared_nvfp4, shared_expert_declared_bf16};

const L0: &str = "model.language_model.layers.0";
const L1: &str = "model.language_model.layers.1";

/// 2026-10-13: Layer 0 ignores its whole shared expert, as the checkpoint does; layer 1 ignores
/// only `gate_proj` and `up_proj`, so its `down_proj` stays NVFP4.
fn checkpoint() -> QuantizationConfig {
    let ignore: Vec<String> = ["gate_proj", "up_proj", "down_proj"]
        .iter()
        .map(|p| format!("{L0}.mlp.shared_expert.{p}"))
        .chain(
            ["gate_proj", "up_proj"]
                .iter()
                .map(|p| format!("{L1}.mlp.shared_expert.{p}")),
        )
        .chain(["lm_head".to_string(), format!("{L0}.mlp.gate")])
        .collect();
    metrale_config::parse_quantization_config(&serde_json::json!({ "quantization_config": {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "group_size": 16,
        "with_input_scale": true,
        "ignore": ignore,
    }}))
    .expect("parses")
    .expect("declares")
}

fn decide(tier: WeightQuantization, qc: Option<&QuantizationConfig>, lp: &str) -> bool {
    let policy = WeightQuantPolicy::for_checkpoint(
        WeightQuantTier::new(tier, W4a4Downcast::Off).expect("tier"),
        qc,
        KernelCaps::default(),
    );
    shared_expert_declared_bf16(&policy, lp)
}

#[test]
fn a_shared_expert_the_checkpoint_ignores_runs_bf16_under_declared() {
    let qc = checkpoint();
    assert!(decide(WeightQuantization::Declared, Some(&qc), L0));
}

#[test]
fn one_quantized_projection_keeps_the_shared_expert_nvfp4() {
    let qc = checkpoint();
    assert!(!decide(WeightQuantization::Declared, Some(&qc), L1));
}

#[test]
fn the_nvfp4_tier_and_an_undeclared_checkpoint_keep_the_engine_default() {
    let qc = checkpoint();
    assert!(!decide(WeightQuantization::Nvfp4, Some(&qc), L0));
    assert!(!decide(WeightQuantization::Declared, None, L0));
}

fn experts_decide(tier: WeightQuantization, qc: Option<&QuantizationConfig>) -> bool {
    let policy = WeightQuantPolicy::for_checkpoint(
        WeightQuantTier::new(tier, W4a4Downcast::Off).expect("tier"),
        qc,
        KernelCaps::default(),
    );
    experts_declared_nvfp4(&policy, L0)
}

#[test]
fn declared_nvfp4_experts_are_marked_only_where_the_plan_is_followed() {
    let qc = checkpoint();
    assert!(experts_decide(WeightQuantization::Declared, Some(&qc)));
    assert!(!experts_decide(WeightQuantization::Nvfp4, Some(&qc)));
    assert!(!experts_decide(WeightQuantization::Declared, None));
}
