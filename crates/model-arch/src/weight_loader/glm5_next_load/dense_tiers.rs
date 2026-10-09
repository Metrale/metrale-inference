// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: `--dense-quantization` registration for one GLM-5.3 layer: every BF16 projection
//! of its mixer and shared expert goes to the FP8 registry (`glm5next_fp8_dense`, `fp8`), or under
//! `w4a16` to the registry `glm5next_w4a16_dense::tier_of` names.
//!
//! Owner: model-arch weight loader (GLM-5.3).
//! Invariants:
//! - Under `fp8` every projection registers FP8 in the order the layer lists it, with the BF16
//!   weight kept beside, as before `w4a16` existed.
//! - A W4A16 projection's weight field ends up holding its NVFP4 key and its BF16 buffer is
//!   freed, after the quantization finished; nothing else in the layer still names the buffer
//!   (`glm5next_w4a16_dense::retarget` refuses otherwise).

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use crate::glm5next_layer::{Glm5NextMixer, Glm5NextMlpSite};
use crate::glm5next_mlp::Glm5NextMlpConfig;
use crate::glm5next_w4a16_dense::{self as w4a16, ProjTier};

/// 2026-10-09: One projection: `(weight, n, k, name)` with the shapes the forward launches.
type Proj = (DevicePtr, usize, usize, &'static str);

/// 2026-10-09: Split the layer's projections by tier: all FP8 unless `w4a16`, then by
/// `tier_of`. Pure, so the decision is tested without a GPU.
pub(super) fn split_by_tier(projs: Vec<Proj>, w4a16_on: bool) -> Result<(Vec<Proj>, Vec<Proj>)> {
    let (mut fp8, mut w4) = (Vec::new(), Vec::new());
    for p in projs {
        if w4a16_on && w4a16::tier_of(p.3)? == ProjTier::W4a16 {
            w4.push(p);
        } else {
            fp8.push(p);
        }
    }
    Ok((fp8, w4))
}

/// 2026-10-09: Register one layer's mixer and shared-expert projections under the published
/// dense tier (see the module invariants).
pub(super) fn register_dense_tiers(
    gpu: &dyn GpuBackend,
    mixer: &mut Glm5NextMixer,
    mlp: &mut Glm5NextMlpSite,
    mlp_cfg: &Glm5NextMlpConfig,
    idx: usize,
) -> Result<()> {
    let mut projs = match &*mixer {
        Glm5NextMixer::Kda { layer, .. } => layer.dense_projections(),
        Glm5NextMixer::Dsa(layer) => layer.dense_projections(),
    };
    if let Glm5NextMlpSite::Moe(w) = &*mlp {
        let (h, s) = (mlp_cfg.hidden, mlp_cfg.local_shared_intermediate);
        projs.push((w.shared.gate_proj, s, h, "shared_experts.gate_proj"));
        projs.push((w.shared.up_proj, s, h, "shared_experts.up_proj"));
        projs.push((w.shared.down_proj, h, s, "shared_experts.down_proj"));
    }
    register_projections(
        gpu,
        projs,
        &mut weight_slots(mixer, mlp),
        w4a16::enabled(),
        idx,
    )
}

/// 2026-10-09: Register `projs` (layer `idx`): FP8 for the FP8 part of [`split_by_tier`], with
/// the BF16 kept; W4A16 for the rest, each retargeted in `slots` to its NVFP4 key and its BF16
/// freed.
pub(super) fn register_projections(
    gpu: &dyn GpuBackend,
    projs: Vec<Proj>,
    slots: &mut [&mut DevicePtr],
    w4a16_on: bool,
    idx: usize,
) -> Result<()> {
    let (fp8, w4) = split_by_tier(projs, w4a16_on)?;
    if !fp8.is_empty() {
        let max_k = fp8.iter().map(|p| p.2).max().unwrap_or(0);
        crate::glm5next_fp8_dense::prepare(gpu, max_k)?;
        let quantize = gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?;
        for (w, n, k, name) in fp8 {
            crate::glm5next_fp8_dense::register(
                gpu,
                quantize,
                w,
                n,
                k,
                &format!("layer {idx} {name}"),
                0,
            )?;
        }
    }
    if w4.is_empty() {
        return Ok(());
    }
    let kernels = w4a16::Nvfp4QuantKernels::load(gpu)?;
    for (w, n, k, name) in w4 {
        let key = w4a16::register(gpu, &kernels, w, n, k, &format!("layer {idx} {name}"), 0)?;
        w4a16::retarget(slots, w, key)?;
        gpu.free(w)?;
    }
    Ok(())
}

/// 2026-10-09: Every weight field a W4A16 projection can live in: the KDA projections'
/// and the shared expert's. A DSA layer has none (`tier_of` keeps it FP8).
fn weight_slots<'a>(
    mixer: &'a mut Glm5NextMixer,
    mlp: &'a mut Glm5NextMlpSite,
) -> Vec<&'a mut DevicePtr> {
    let mut slots = Vec::new();
    if let Glm5NextMixer::Kda { layer, .. } = mixer {
        let w = &mut layer.weights;
        slots.extend([
            &mut w.q_proj.weight,
            &mut w.k_proj.weight,
            &mut w.v_proj.weight,
            &mut w.f_a.weight,
            &mut w.f_b.weight,
            &mut w.b_proj.weight,
            &mut w.g_a.weight,
            &mut w.g_b.weight,
            &mut w.o_proj.weight,
        ]);
    }
    if let Glm5NextMlpSite::Moe(w) = mlp {
        slots.extend([
            &mut w.shared.gate_proj,
            &mut w.shared.up_proj,
            &mut w.shared.down_proj,
        ]);
    }
    slots
}

#[cfg(test)]
#[path = "dense_tiers_tests.rs"]
mod tests;
