// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `lm_head_flags`: what `--lm-head-dtype` sets for each head the
//! weight-quantization policy chooses, on unsloth/Qwen3.8-27B-NVFP4's own
//! `quantization_config` (FP8-declared head).
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.

use metrale_config::weight_quantization::{KernelCaps, LmHeadChoice, LmHeadFormat};
use metrale_config::{
    QuantizationConfig, W4a4Downcast, WeightQuantPolicy, WeightQuantTier, WeightQuantization,
};

use super::lm_head_flags;

fn unsloth() -> QuantizationConfig {
    let block: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../config/src/precision_plan/fixtures/unsloth_qwen3_8_27b_nvfp4.json"
    ))
    .expect("fixture parses");
    metrale_config::parse_quantization_config(&serde_json::json!({ "quantization_config": block }))
        .expect("parses")
        .expect("declares")
}

fn choice(tier: WeightQuantization, batched_head: bool) -> LmHeadChoice {
    let qc = unsloth();
    let caps = KernelCaps {
        fp8_lm_head_batched: batched_head,
        ..KernelCaps::default()
    };
    WeightQuantPolicy::for_checkpoint(
        WeightQuantTier::new(tier, W4a4Downcast::Off).expect("tier"),
        Some(&qc),
        caps,
    )
    .lm_head()
}

/// 2026-09-28: `(None, false)`: the engine's per-model default (NVFP4 on this model).
const ENGINE_DEFAULT: (Option<bool>, bool) = (None, false);
const FP8: (Option<bool>, bool) = (Some(false), true);

/// 2026-09-28: Path A: under `declared` without the batched FP8 head, `default` keeps the
/// engine's head; Path B: with it, the declared FP8 head.
#[test]
fn declared_takes_the_fp8_head_only_with_the_batched_kernel() {
    let without = choice(WeightQuantization::Declared, false);
    assert_eq!(without, LmHeadChoice::PendingFp8Kernel);
    assert_eq!(lm_head_flags("default", without).unwrap(), ENGINE_DEFAULT);
    let with = choice(WeightQuantization::Declared, true);
    assert_eq!(with, LmHeadChoice::Declared(LmHeadFormat::Fp8));
    assert_eq!(lm_head_flags("default", with).unwrap(), FP8);
}

/// 2026-09-28: Path C: the `nvfp4` tier keeps the engine's head either way.
#[test]
fn nvfp4_keeps_the_engine_head() {
    for batched in [false, true] {
        let c = choice(WeightQuantization::Nvfp4, batched);
        assert_eq!(c, LmHeadChoice::EngineDefault);
        assert_eq!(lm_head_flags("default", c).unwrap(), ENGINE_DEFAULT);
    }
}

/// 2026-09-28: An explicit `--lm-head-dtype` wins over every choice, and an unknown value is
/// refused.
#[test]
fn an_explicit_head_overrides_every_choice() {
    for c in [
        LmHeadChoice::EngineDefault,
        LmHeadChoice::PendingFp8Kernel,
        LmHeadChoice::Declared(LmHeadFormat::Fp8),
        LmHeadChoice::Declared(LmHeadFormat::Bf16),
    ] {
        assert_eq!(
            lm_head_flags("bf16", c).unwrap(),
            (Some(true), false),
            "{c:?}"
        );
        assert_eq!(
            lm_head_flags("nvfp4", c).unwrap(),
            (Some(false), false),
            "{c:?}"
        );
        assert_eq!(lm_head_flags("fp8", c).unwrap(), FP8, "{c:?}");
        assert!(lm_head_flags("fp4", c).is_err(), "{c:?}");
    }
}

/// 2026-09-28: The production caps (`kernel_caps()`) carry the batched FP8 head, so under
/// `declared` `default` takes the checkpoint's FP8 head; under `nvfp4` the engine's head stays.
#[test]
fn the_shipped_caps_take_the_declared_fp8_head() {
    let qc = unsloth();
    let caps = metrale_model_layers::layers::kernel_caps();
    let pick = |tier| {
        WeightQuantPolicy::for_checkpoint(
            WeightQuantTier::new(tier, W4a4Downcast::Off).expect("tier"),
            Some(&qc),
            caps,
        )
        .lm_head()
    };
    let declared = pick(WeightQuantization::Declared);
    assert_eq!(declared, LmHeadChoice::Declared(LmHeadFormat::Fp8));
    assert_eq!(lm_head_flags("default", declared).unwrap(), FP8);
    let nvfp4 = pick(WeightQuantization::Nvfp4);
    assert_eq!(nvfp4, LmHeadChoice::EngineDefault);
    assert_eq!(lm_head_flags("default", nvfp4).unwrap(), ENGINE_DEFAULT);
}
