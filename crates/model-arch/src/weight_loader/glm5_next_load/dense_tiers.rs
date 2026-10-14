// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: `--dense-quantization` registration for one GLM-5.3 layer: every BF16 projection
//! of its mixer and shared expert goes to the FP8 registry (`glm5next_fp8_dense`, `fp8`), or under
//! `w4a16` to the registry `glm5next_w4a16_dense::tier_of` names. 2026-10-11: the DSA q_a /
//! absorbed q / absorbed o projections (`DSA_SITES`) take `--dsa-quantization`'s tier instead,
//! whatever the dense tier is: BF16 under `declared` (left unregistered), FP8 under `w8a8`, W4A16
//! under `w4a16` (`tier_for`).
//!
//! 2026-10-14: Under a phase split (`decode=w4a16,prefill=fp8` / `decode=w4a16,prefill=w8a8`) a
//! projection whose decode tier is W4A16 and prompt-pass tier FP8 also gets an FP8 copy
//! (`glm5next_fp8_dense::register_prefill`), quantized from its BF16 before that is freed.
//!
//! Owner: model-arch weight loader (GLM-5.3).
//! Invariants:
//! - Under `fp8` every projection registers FP8 in the order the layer lists it, with the BF16
//!   weight kept beside, as before `w4a16` existed.
//! - A W4A16 projection's weight field ends up holding its NVFP4 key and its BF16 buffer is
//!   freed, after the quantization finished; nothing else in the layer still names the buffer
//!   (`glm5next_w4a16_dense::retarget` refuses otherwise).

use anyhow::Result;
use anyhow::bail;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::{DenseQuantization, DsaQuantization, Phased};

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
    use metrale_model_layers::layers as ml;
    register_projections(
        gpu,
        projs,
        &mut weight_slots(mixer, mlp),
        Phased {
            decode: ml::dense_quantization(),
            prefill: ml::dense_prefill_quantization(),
        },
        Phased {
            decode: ml::dsa_quantization(),
            prefill: ml::dsa_prefill_quantization(),
        },
        idx,
    )
}

/// 2026-10-14: Boot readback of a phase split: the per-phase tiers, and the prompt-pass FP8 copies
/// held beside the decode's W4A16 (count and GB, read back from the registry, not predicted).
pub(super) fn announce_prefill_copies() {
    use metrale_model_layers::layers::{self as ml, phased_name};
    let dense = Phased {
        decode: ml::dense_quantization(),
        prefill: ml::dense_prefill_quantization(),
    };
    let dsa = Phased {
        decode: ml::dsa_quantization(),
        prefill: ml::dsa_prefill_quantization(),
    };
    if !dense.is_split() && !dsa.is_split() {
        return;
    }
    let (count, bytes) = crate::glm5next_fp8_dense::registered_prefill();
    tracing::warn!(
        "glm5_next --dense-quantization {} --dsa-quantization {}: {count} W4A16 projections also \
         held as FP8 per-row ({:.2} GB) for the prompt pass only; decode and speculative verify \
         keep W4A16",
        phased_name(dense, DenseQuantization::name),
        phased_name(dsa, DsaQuantization::name),
        bytes as f64 / 1e9
    );
}

/// 2026-10-14: The names of the projections that are W4A16 at decode and FP8 in the prompt pass.
/// Errors on a phase pair the loader does not serve: every projection's two tiers must agree, or
/// be that pair. Pure.
pub(super) fn prefill_copies(
    projs: &[Proj],
    dense: Phased<DenseQuantization>,
    dsa: Phased<DsaQuantization>,
) -> Result<Vec<&'static str>> {
    let mut copies = Vec::new();
    for p in projs {
        match (
            tier_for(p.3, dense.decode, dsa.decode)?,
            tier_for(p.3, dense.prefill, dsa.prefill)?,
        ) {
            (Tier::W4a16, Tier::Fp8) => copies.push(p.3),
            (d, f) if d == f => {}
            (d, f) => bail!("{}: decode {d:?} with prefill {f:?} is not served", p.3),
        }
    }
    Ok(copies)
}

/// 2026-10-09: Register `projs` (layer `idx`): FP8 for the FP8 part of [`split_by_tier`], with
/// the BF16 kept; W4A16 for the rest, each retargeted in `slots` to its NVFP4 key and its BF16
/// freed. 2026-10-14: The split is by the decode tiers; [`prefill_copies`] adds the prompt
/// pass's FP8 copies.
pub(super) fn register_projections(
    gpu: &dyn GpuBackend,
    projs: Vec<Proj>,
    slots: &mut [&mut DevicePtr],
    dense: Phased<DenseQuantization>,
    dsa: Phased<DsaQuantization>,
    idx: usize,
) -> Result<()> {
    let copies = prefill_copies(&projs, dense, dsa)?;
    let (fp8, w4) = split_by_tier(projs, dense.decode, dsa.decode)?;
    let max_k = fp8
        .iter()
        .chain(w4.iter().filter(|p| copies.contains(&p.3)))
        .map(|p| p.2)
        .max();
    let quantize = match max_k {
        Some(max_k) => {
            crate::glm5next_fp8_dense::prepare(gpu, max_k)?;
            Some(gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?)
        }
        None => None,
    };
    if let Some(quantize) = quantize {
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
        let what = format!("layer {idx} {name}");
        let key = w4a16::register(gpu, &kernels, w, n, k, &what, 0)?;
        if let (true, Some(quantize)) = (copies.contains(&name), quantize) {
            crate::glm5next_fp8_dense::register_prefill(gpu, quantize, w, key, n, k, &what, 0)?;
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
