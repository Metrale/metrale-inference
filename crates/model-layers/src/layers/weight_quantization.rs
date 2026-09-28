// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The `--weight-quantization` tier in force for this process
//! (`metrale_config::WeightQuantTier`): `declared` or `nvfp4`, and under `nvfp4` the
//! `--w4a4-downcast` lever. Loaders pair it with a model's plan in a
//! `metrale_config::WeightQuantPolicy`; decode dispatch reads it beside each weight's stamp
//! (`QuantizedWeight::act`). It has no environment fallback.
//!
//! Owner: model-layers (quantization dispatch).
//! Invariants:
//! - The first publication or read wins (`OnceLock`). The serve publishes before the model is
//!   built; anything that reads first (a test, an example) fixes the default, `declared`.

use std::sync::OnceLock;

use metrale_config::WeightQuantTier;

static TIER: OnceLock<WeightQuantTier> = OnceLock::new();

/// 2026-09-28: Publish `--weight-quantization` and its lever. Returns the tier in force; a
/// caller that gets a different one should warn.
pub fn set_weight_quantization_from_cli(tier: WeightQuantTier) -> WeightQuantTier {
    let _ = TIER.set(tier);
    *TIER.get().expect("just set")
}

/// 2026-09-28: The tier in force: `declared` unless the serve published another.
pub fn weight_quantization() -> WeightQuantTier {
    *TIER.get_or_init(WeightQuantTier::default)
}

/// 2026-09-28: Which decode families the policy may rely on: the dense/non-expert W8A8 family
/// (`ops::w8a8_decode`), the MoE expert W8A8 family (`moe/fp8_grouped_tc_w8a8.rs`), and the
/// batched FP8 lm_head. `WeightQuantPolicy::fp8_decode_act` reports FP8 activations for the
/// layers each W8A8 family serves; each still checks at run time that its kernels resolved and
/// the shapes fit, and falls back to W8A16 otherwise. `fp8_lm_head_batched` stays false until a
/// batched FP8 head is verified, so `declared` keeps the engine's default head.
pub fn kernel_caps() -> metrale_config::weight_quantization::KernelCaps {
    metrale_config::weight_quantization::KernelCaps {
        w8a8_decode: true,
        w8a8_moe_decode: true,
        fp8_lm_head_batched: false,
    }
}
