// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The GLM MLP precision plan of one layer's group, from the published
//! `--weight-quantization` (the per-module stamp of the checkpoint's plan) and
//! `--activation-quantization` (the `moe` / `ffn` ladder), and the W4A4 kernels that resolved.
//! The decision itself is `glm5next_mlp::precision`; this only gathers its inputs.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_mlp::precision::{GroupPrecision, MlpGroup};
use metrale_config::{Nvfp4Act, ProjFamily, WeightQuantPolicy};

/// 2026-10-08: The plan of `group` at `layer`. The stamp combines the group's gate, up and down
/// modules (of expert `expert0` for a routed group): W4A4 only when all three declare FP4
/// activations. `has_scales`: whether every projection carries a static activation scale.
pub(super) fn group_precision(
    config: &ModelConfig,
    kernels: &Glm5NextMlpKernels,
    layer: usize,
    group: MlpGroup,
    expert0: usize,
    has_scales: bool,
) -> Result<GroupPrecision> {
    let policy = WeightQuantPolicy::for_checkpoint(
        metrale_model_layers::layers::weight_quantization(),
        config.quantization_config.as_ref(),
        metrale_model_layers::layers::kernel_caps(),
    );
    // 2026-10-11: `METRALE_GLM_BATCH_INVARIANT` refuses its conflicting levers here, where the
    // routed experts' kernels have resolved.
    if group == MlpGroup::RoutedExperts {
        crate::glm5next_batch_inv::check_at_load(kernels)?;
        // 2026-10-14: `METRALE_GLM_MOE_W4A4_GU_FUSED` refuses a load it cannot run here too.
        crate::glm5next_mlp::forward::check_gu_fused_at_load(kernels)?;
        // 2026-10-14: and `METRALE_GLM_MOE_W4A4_RASTER`.
        crate::glm5next_mlp::forward::check_raster_at_load(kernels)?;
        // 2026-10-15: and `METRALE_GLM_MOE_W4A4_NC`.
        crate::glm5next_mlp::forward::check_nc_at_load(kernels)?;
        // 2026-10-14: And `METRALE_GLM_MOE_W4A4_WIDE_SWEEP`.
        crate::glm5next_mlp::forward::check_wide_sweep_at_load(kernels)?;
    }
    let (base, family, w4a4_rows) = match group {
        // 2026-10-10: `METRALE_GLM_MOE_W4A4_GROUPED=1` lifts the routed W4A4 row cap with
        // the grouped W4A4 GEMM (`glm5next_mlp::forward::w4a4_grouped`).
        MlpGroup::RoutedExperts => (
            format!("mlp.experts.{expert0}"),
            ProjFamily::Moe,
            if crate::glm5next_mlp::forward::moe_w4a4_grouped()? {
                kernels.w4a4_expert_rows_grouped()
            } else {
                kernels.w4a4_expert_rows()
            },
        ),
        MlpGroup::DenseMlp => (
            "mlp".to_string(),
            ProjFamily::Ffn,
            kernels.w4a4_dense_rows(),
        ),
    };
    let stamp = Nvfp4Act::combine(
        ["gate_proj", "up_proj", "down_proj"]
            .map(|p| policy.nvfp4_act(&qualify(layer, &format!("{base}.{p}")))),
    );
    let plan = GroupPrecision::resolve(
        group,
        metrale_model_layers::layers::activation_quantization()
            .ladder(family)
            .clone(),
        stamp,
        w4a4_rows,
        has_scales,
    )
    .with_context(|| format!("glm5_next layer {layer}"))?;
    Ok(plan)
}

/// 2026-10-08: Log each distinct plan description once per process (the 42 routed layers share
/// one), with the row ranges that run above the declared W4A4.
pub(super) fn announce(p: &GroupPrecision, max_rows: usize) {
    static SEEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let d = p.describe(max_rows);
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&d) {
        tracing::info!("{d}");
        seen.push(d);
    }
}
