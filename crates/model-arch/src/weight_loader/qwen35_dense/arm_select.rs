// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The per-layer arm predicates `load_layers` builds by and `prune_after_load`
//! frees by, moved from `qwen35_dense.rs`. Each lever comes from the settings passed in, so
//! the load and the prune read the same values.
//!
//! Owner: model-arch weight loader (Qwen3.5 dense).
//! Invariants: the load and the prune pass the same settings.

use metrale_config::ModelConfig;
use metrale_model_layers::weight_map::Nvfp4Variant;
use metrale_model_weights::weights::WeightStore;
use metrale_settings::InferenceServerSettings;

use super::{proj_is_fp8_any_scale, proj_is_native_fp8, proj_q2_group};

/// 2026-09-25: `METRALE_DENSE_FP8=1` enables the native FP8 dense-FFN and attention arms
/// (`ffn_fp8_arm_selected`, `attn_fp8`); off by default: on an FP8 checkpoint it gives correct text
/// but decodes slower than the NVFP4 path.
pub(super) fn dense_fp8_enabled(s: &InferenceServerSettings) -> bool {
    s.kernels.dense_fp8
}

/// 2026-09-25: Whether the native block-scaled FP8 GDN arm runs for this SSM layer.
/// `load_layers` and `prune_after_load` both call it; the second frees the store tensors
/// the first copied into the fused `[QKV|Z]` weight. The keep-packed Q2 arm is tested
/// first in `load_layers`, so its condition is part of this one.
pub(super) fn gdn_fp8_arm_selected(
    store: &WeightStore,
    la: &str,
    tp_size: usize,
    s: &InferenceServerSettings,
) -> bool {
    let q2 = tp_size.max(1) == 1
        && !s.kernels.no_q2_gdn
        && proj_q2_group(store, &format!("{la}.in_proj_qkv")).is_some()
        && proj_q2_group(store, &format!("{la}.in_proj_z")).is_some();
    !q2 && !s.kernels.no_gdn_fp8
        && proj_is_fp8_any_scale(store, &format!("{la}.in_proj_qkv"))
        && proj_is_fp8_any_scale(store, &format!("{la}.in_proj_z"))
        && proj_is_fp8_any_scale(store, &format!("{la}.out_proj"))
}

/// 2026-09-25: The dense-FFN width: `intermediate_size`, or `moe_intermediate_size` when
/// that is 0; the same order as `load_dense_ffn` (`weight_map/fp8_lut.rs`).
pub(super) fn ffn_inter(config: &ModelConfig) -> usize {
    if config.intermediate_size > 0 {
        config.intermediate_size
    } else {
        config.moe_intermediate_size
    }
}

/// 2026-09-25: Whether the native FP8 dense-FFN arm runs for this layer:
/// `METRALE_DENSE_FP8=1`, TP=1, the `Fp8Dequanted` variant and a block-scaled FP8
/// `mlp.gate_proj`. `load_layers` calls it, and `prune_after_load` through
/// `ffn_gateup_fused_selected`.
pub(super) fn ffn_fp8_arm_selected(
    store: &WeightStore,
    config: &ModelConfig,
    variant: Nvfp4Variant,
    lp: &str,
    s: &InferenceServerSettings,
) -> bool {
    dense_fp8_enabled(s)
        && config.tp_world_size.max(1) == 1
        && matches!(variant, Nvfp4Variant::Fp8Dequanted)
        && proj_is_native_fp8(store, &format!("{lp}.mlp.gate_proj"))
}

/// 2026-09-25: Whether this layer's gate and up are fused into one `[2*inter, hidden]`
/// block-scaled FP8 weight: the FP8 FFN arm is selected, the target arms
/// `ffn_gateup_fused` (`[defaults]`, or `METRALE_FFN_GATEUP_FUSED`), the model has no
/// experts, both widths are multiples of 128 (so the concatenated scale grids meet on a
/// block boundary), and both tensors are `[inter, hidden]`. `prune_after_load` frees
/// the source tensors of exactly the layers this returns true for.
pub(super) fn ffn_gateup_fused_selected(
    store: &WeightStore,
    config: &ModelConfig,
    variant: Nvfp4Variant,
    lp: &str,
    s: &InferenceServerSettings,
) -> bool {
    let inter = ffn_inter(config);
    let hidden = config.hidden_size;
    // 2026-09-25: Checked here rather than at the call site because `prune_after_load`
    // frees on this answer: tensors whose shape is not `[inter, hidden]` from `config`
    // are not fused, and so are not freed.
    let on_disk = |name: &str| {
        store
            .get(&format!("{lp}.mlp.{name}.weight"))
            .is_ok_and(|w| w.shape == [inter, hidden])
    };
    ffn_fp8_arm_selected(store, config, variant, lp, s)
        && metrale_model_layers::layers::dense_ffn::gateup_fused::ffn_gateup_fused()
        && config.num_experts == 0
        && inter > 0
        && inter.is_multiple_of(128)
        && hidden.is_multiple_of(128)
        && on_disk("gate_proj")
        && on_disk("up_proj")
}

#[cfg(test)]
#[path = "arm_select_settings_tests.rs"]
mod settings_tests;
