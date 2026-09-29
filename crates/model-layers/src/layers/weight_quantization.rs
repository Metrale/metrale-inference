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

/// 2026-09-28: Which decode families the policy may rely on. On: the dense per-channel W8A8
/// family (`ops::w8a8_decode`). 2026-09-29: On as well, validated on Qwen3.6-35B-A3B-FP8 by the
/// full BFCL draw and agentic-webserver (`declared` against `nvfp4`, one binary, dgx3): the MoE
/// expert W8A8 family (`moe/fp8_grouped_tc_w8a8.rs`) and the block-scaled attention/GDN W8A8
/// adoption (`qwen35/load_layers/w8a8_adopt.rs`). Also on: the batched FP8 lm_head (`model/lm_head_fp8_rows.rs`, one weight pass per 128 rows
/// at every row count), so `declared` takes a checkpoint's declared FP8 head. Each W8A8
/// family still checks at run time that its kernels resolved and the shapes fit.
pub fn kernel_caps() -> metrale_config::weight_quantization::KernelCaps {
    metrale_config::weight_quantization::KernelCaps {
        w8a8_decode: true,
        w8a8_moe_decode: true,
        w8a8_block_scaled_decode: true,
        fp8_lm_head_batched: true,
    }
}
