// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Several registered projections of one activation as one quantization and one
//! `w8a8_gemv_rowscale_mb1_ku8_split4` launch (`ops::w8a8_gemv_split4`), each into its own output:
//! the bytes [`super::proj`] writes for each. First user: the DSA input projections
//! (`glm5next_dsa::layer::proj_in_merged`).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - [`split_ready`] and [`proj_split`] decide from one plan (`plan`): what one answers, the
//!   other does.
//! - A declined call launches nothing; the caller runs [`super::proj`] per projection.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops::{
    SPLIT4_MAX_ROWS, SplitOut, W8a8Scale, split4_refusal, w8a8_act_quant, w8a8_gemv_split4,
};

use super::{Fp8Dense, enabled, state};

/// 2026-10-21: One output of [`proj_split`]: the BF16 weight the projection is registered under,
/// its output (`[m, n]` contiguous BF16) and `n`.
#[derive(Clone, Copy, Debug)]
pub struct SplitProj {
    pub b: DevicePtr,
    pub c: DevicePtr,
    pub n: usize,
}

/// 2026-10-21: The split launch's segments for `outs` at `m` rows of K = `k`, or `None` when it
/// cannot serve them: the split entry not built, a projection not registered as one per-row
/// segment, `m` outside 1..=8, the wide path taking `m` rows, or shapes that do not tile
/// (`split4_refusal`). A projection registered at another shape is an error, as in `proj`.
fn plan(s: &Fp8Dense, m: usize, k: usize, outs: &[SplitProj]) -> Result<Option<Vec<SplitOut>>> {
    if !(1..=SPLIT4_MAX_ROWS).contains(&m)
        || s.split4.0 == 0
        || crate::glm5next_fp8_wide::takes(s.wide_min_rows, m)
    {
        return Ok(None);
    }
    let mut segs = Vec::with_capacity(outs.len());
    for o in outs {
        let Some(w) = s.weights.get(&o.b.0) else {
            return Ok(None);
        };
        ensure!(
            (w.n() as usize, w.k() as usize) == (o.n, k),
            "--dense-quantization fp8: weight {:#x} registered as [{}, {}], launched as [{}, {k}]",
            o.b.0,
            w.n(),
            w.k(),
            o.n
        );
        let [seg] = w.segment_weights() else {
            return Ok(None);
        };
        segs.push(SplitOut {
            w: *seg,
            out: o.c,
            ldc: o.n as u32,
        });
    }
    Ok(split4_refusal(&segs, m).is_none().then_some(segs))
}

/// 2026-10-21: Whether [`proj_split`] would run `outs` at `m` rows (the registry in use and
/// the plan served), with nothing launched.
pub fn split_ready(m: usize, k: usize, outs: &[SplitProj]) -> Result<bool> {
    if !enabled() {
        return Ok(false);
    }
    let g = state().lock().unwrap_or_else(|e| e.into_inner());
    match g.as_ref() {
        Some(s) => Ok(plan(s, m, k, outs)?.is_some()),
        None => Ok(false),
    }
}

/// 2026-10-21: `c[m, n] = a[m, k] @ w[n, k]^T` for every projection of `outs`, as one
/// quantization of `a` (or the stable one the scratch holds) and one split launch.
/// `Ok(false)`, with nothing launched, where the registry is off or the plan declines.
pub fn proj_split(
    gpu: &dyn GpuBackend,
    a: DevicePtr,
    m: usize,
    k: usize,
    outs: &[SplitProj],
    stream: u64,
) -> Result<bool> {
    if !enabled() {
        return Ok(false);
    }
    proj_split_registered(gpu, a, m, k, outs, stream)
}

/// 2026-10-21: [`proj_split`] whatever the published tier.
fn proj_split_registered(
    gpu: &dyn GpuBackend,
    a: DevicePtr,
    m: usize,
    k: usize,
    outs: &[SplitProj],
    stream: u64,
) -> Result<bool> {
    let mut g = state().lock().unwrap_or_else(|e| e.into_inner());
    let Some(s) = g.as_mut() else {
        return Ok(false);
    };
    let Some(segs) = plan(s, m, k, outs)? else {
        return Ok(false);
    };
    let key = (a.0, m, k, stream);
    if !(s.stable == Some(a.0) && s.quantized == Some(key)) {
        let k32 = k as u32;
        w8a8_act_quant(gpu, &s.kernels, W8a8Scale::PerRow, a, k32, m, k32, &s.scratch, stream)?;
        s.quantized = Some(key);
    }
    w8a8_gemv_split4(gpu, s.split4, &segs, &s.scratch, m, stream)?;
    Ok(true)
}

#[cfg(test)]
#[path = "glm5next_fp8_dense_split_tests.rs"]
mod tests;
