// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: One dense MLP site as `load_layers` binds it, moved verbatim out of `loader.rs`
//! (500-line cap).
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_mlp::Glm5NextDenseSite;
use crate::glm5next_mlp::build_w4a4::{act_scales_of, build_dense_nvfp4};
use crate::glm5next_mlp::precision::{MlpGroup, MlpKernel};

/// 2026-10-08: One dense MLP site: its precision plan, then the weight forms the plan reaches
/// within `max_rows`: the checkpoint's packed NVFP4, TP-sliced, for W4A4, and the BF16
/// dequantization for the 16-bit path (both when a ladder mixes them).
#[allow(clippy::too_many_arguments)]
pub(super) fn build_dense_site(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    config: &ModelConfig,
    mlp_cfg: &Glm5NextMlpConfig,
    kernels: &Glm5NextMlpKernels,
    src: &LayerSource,
    idx: usize,
    max_rows: usize,
) -> Result<Glm5NextDenseSite> {
    let act = ["gate_proj", "up_proj", "down_proj"]
        .map(|p| input_scale(gpu, store, idx, &format!("mlp.{p}")))
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let act = [act[0], act[1], act[2]];
    let precision = mlp_precision::group_precision(
        config,
        kernels,
        idx,
        MlpGroup::DenseMlp,
        0,
        act.iter().all(Option::is_some),
    )?;
    mlp_precision::announce(&precision, max_rows);
    let nvfp4 = if precision.reaches(MlpKernel::W4a4Static, max_rows) {
        let scales = act_scales_of(act, &format!("layer {idx} dense MLP"))?;
        let raw = |n: &str| src.raw(n);
        let w = build_dense_nvfp4(
            gpu,
            mlp_cfg.hidden,
            config.intermediate_size,
            mlp_cfg.dense_slice(),
            "mlp",
            &raw,
            scales,
        )?;
        Some((w, scales))
    } else {
        None
    };
    let bf16 = if precision.reaches(MlpKernel::Bf16, max_rows) {
        let load = |n: &str| src.f32(n);
        Some(mlp_build::build_dense_mlp(
            gpu,
            mlp_cfg,
            config.intermediate_size,
            mlp_cfg.dense_slice(),
            "mlp",
            &load,
        )?)
    } else {
        None
    };
    let grouped = match &nvfp4 {
        Some((w, _)) => crate::glm5next_mlp::forward::DenseGroupedTables::build(gpu, kernels, w)?,
        None => None,
    };
    Ok(Glm5NextDenseSite {
        bf16,
        nvfp4,
        precision,
        grouped,
    })
}
