// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: `--dense-quantization` registration for one GLM-5.3 layer: every BF16 projection
//! of its mixer and shared expert goes to the FP8 registry (`glm5next_fp8_dense`, `fp8`), or under
//! `w4a16` to the registry `glm5next_w4a16_dense::tier_of` names. 2026-10-11: the DSA q_a /
//! absorbed q / absorbed o projections (`DSA_SITES`) take `--dsa-quantization`'s tier instead,
//! whatever the dense tier is: BF16 under `declared` (left unregistered), FP8 under `w8a8`, W4A16
//! under `w4a16` (`tier_for`).
//!
//! Owner: model-arch weight loader (GLM-5.3).
//! Invariants:
//! - Under `fp8` every projection registers FP8 in the order the layer lists it, with the BF16
//!   weight kept beside, as before `w4a16` existed.
//! - A W4A16 projection's weight field ends up holding its NVFP4 key and its BF16 buffer is
//!   freed, after the quantization finished; nothing else in the layer still names the buffer
//!   (`glm5next_w4a16_dense::retarget` refuses otherwise).

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::{DenseQuantization, DsaQuantization};
use metrale_model_layers::weight_map::QuantizedWeight;

use crate::glm5next_dense_levers::DenseLevers;
use crate::glm5next_layer::{Glm5NextMixer, Glm5NextMlpSite};
use crate::glm5next_mlp::Glm5NextMlpConfig;
use crate::glm5next_w4a16_dense::{self as w4a16, ProjTier};

/// 2026-10-09: One projection: `(weight, n, k, name)` with the shapes the forward launches.
type Proj = (DevicePtr, usize, usize, &'static str);

/// 2026-10-11: Where one projection is served.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Tier {
    /// 2026-10-11: Its checkpoint BF16 (not registered).
    Bf16,
    Fp8,
    W4a16,
}

/// 2026-10-11: The tier of the projection named `name` under `dense` and `dsa`. The
/// `DSA_SITES` follow `dsa` alone; every other projection follows `dense` (all FP8 under `fp8`,
/// `glm5next_w4a16_dense::tier_of` under `w4a16`, which refuses a name it has no decision for).
/// Pure, so the decision is tested without a GPU.
pub(super) fn tier_for(name: &str, dense: DenseQuantization, dsa: DsaQuantization) -> Result<Tier> {
    if w4a16::DSA_SITES.contains(&name) {
        return Ok(match dsa {
            DsaQuantization::Declared => Tier::Bf16,
            DsaQuantization::W8a8 => Tier::Fp8,
            DsaQuantization::W4a16 => Tier::W4a16,
        });
    }
    Ok(match dense {
        DenseQuantization::Declared => Tier::Bf16,
        DenseQuantization::Fp8 => Tier::Fp8,
        DenseQuantization::W4a16 => match w4a16::tier_of(name)? {
            ProjTier::W4a16 => Tier::W4a16,
            ProjTier::Fp8 => Tier::Fp8,
        },
    })
}

/// 2026-10-09: Split the layer's projections by tier into `(fp8, w4a16)`, keeping the layer's
/// order; BF16 projections are in neither. 2026-10-11: by [`tier_for`].
pub(super) fn split_by_tier(
    projs: Vec<Proj>,
    dense: DenseQuantization,
    dsa: DsaQuantization,
) -> Result<(Vec<Proj>, Vec<Proj>)> {
    let (mut fp8, mut w4) = (Vec::new(), Vec::new());
    for p in projs {
        match tier_for(p.3, dense, dsa)? {
            Tier::Bf16 => {}
            Tier::Fp8 => fp8.push(p),
            Tier::W4a16 => w4.push(p),
        }
    }
    Ok((fp8, w4))
}

/// 2026-10-11: Whether any projection of a layer can leave BF16: a dense or DSA tier below
/// declared. The loader registers only then.
pub(super) fn any_below_declared(dense: DenseQuantization, dsa: DsaQuantization) -> bool {
    dense.below_declared() || dsa.below_declared()
}

/// 2026-10-09: Register one layer's mixer and shared-expert projections under the published
/// dense tier (see the module invariants).
pub(super) fn register_dense_tiers(
    gpu: &dyn GpuBackend,
    mixer: &mut Glm5NextMixer,
    mlp: &mut Glm5NextMlpSite,
    mlp_cfg: &Glm5NextMlpConfig,
    levers: &DenseLevers,
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
        metrale_model_layers::layers::dense_quantization(),
        metrale_model_layers::layers::dsa_quantization(),
        levers,
        idx,
    )
}

/// 2026-10-09: Register `projs` (layer `idx`): FP8 for the FP8 part of [`split_by_tier`], with
/// the BF16 kept; W4A16 for the rest, each retargeted in `slots` to its NVFP4 key and its BF16
/// freed. 2026-10-15: each projection `METRALE_GLM_DEQUANT_PERSIST` names also keeps its BF16
/// dequantized copy (`glm5next_dequant_persist`). 2026-10-17: `levers` are the resolved dense
/// tier levers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn register_projections(
    gpu: &dyn GpuBackend,
    projs: Vec<Proj>,
    slots: &mut [&mut DevicePtr],
    dense: DenseQuantization,
    dsa: DsaQuantization,
    levers: &DenseLevers,
    idx: usize,
) -> Result<()> {
    let (fp8, w4) = split_by_tier(projs, dense, dsa)?;
    let persist = levers.persist;
    if !fp8.is_empty() {
        let max_k = fp8.iter().map(|p| p.2).max().unwrap_or(0);
        crate::glm5next_fp8_dense::prepare(gpu, max_k, levers.fp8_dequant_min_rows)?;
        let quantize = gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?;
        for (w, n, k, name) in fp8 {
            let what = format!("layer {idx} {name}");
            crate::glm5next_fp8_dense::register(gpu, quantize, w, n, k, &what, 0)?;
            if persist.keeps(true, name) {
                let seg = crate::glm5next_fp8_dense::row_weight(w)
                    .with_context(|| format!("{what}: no per-row FP8 weight to keep"))?;
                crate::glm5next_dequant_persist::keep_fp8(gpu, &seg, &what)?;
            }
        }
    }
    if w4.is_empty() {
        return Ok(());
    }
    let kernels = w4a16::Nvfp4QuantKernels::load(gpu, levers)?;
    for (w, n, k, name) in w4 {
        let what = format!("layer {idx} {name}");
        let key = w4a16::register(gpu, &kernels, w, n, k, &what, 0)?;
        if persist.keeps(false, name) {
            let [(weight, _), (weight_scale, _)] = w4a16::footprint(key)
                .with_context(|| format!("{what}: not registered under its key"))?;
            let q = QuantizedWeight {
                weight,
                weight_scale,
                ..QuantizedWeight::null()
            };
            crate::glm5next_dequant_persist::keep_w4a16(gpu, &q, n, k, &what)?;
        }
        w4a16::retarget(slots, w, key)?;
        gpu.free(w)?;
    }
    Ok(())
}

/// 2026-10-09: Every weight field a W4A16 projection can live in: the KDA projections', the
/// shared expert's, and the DSA projections `--dsa-quantization w4a16` moves.
fn weight_slots<'a>(
    mixer: &'a mut Glm5NextMixer,
    mlp: &'a mut Glm5NextMlpSite,
) -> Vec<&'a mut DevicePtr> {
    let mut slots = Vec::new();
    match mixer {
        Glm5NextMixer::Kda { layer, .. } => {
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
        Glm5NextMixer::Dsa(layer) => {
            let w = &mut layer.weights;
            slots.extend([&mut w.q_a_proj, &mut w.q_absorb, &mut w.o_absorb]);
        }
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
