// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The weight side of the GLM-5.3 W4A4 MLP: the checkpoint's packed NVFP4 dense MLP
//! sliced for one TP rank, and the static activation scales a W4A4 group runs under.
//!
//! NVFP4 as the checkpoint stores it: `[n, k / 2]` U8 (two E2M1 codes per byte, element `2j` in
//! the low nibble of byte `j`), `[n, k / 16]` E4M3 block scales, an F32 `weight_scale_2` and an
//! F32 `input_scale`. A row slice keeps whole rows; a column slice keeps bytes
//! `[start / 2, end / 2)` and scales `[start / 16, end / 16)` of every row, so it must start and
//! end on a 16-column block.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - The slices are pure functions of host bytes; only [`build_dense_nvfp4`] touches the device.
//! - A W4A4 projection's K is a multiple of 128 (the mx kernels read whole k128 chunks and drop
//!   a tail) and at most 32768; [`check_w4a4_k`] refuses anything else before a launch exists.
//!   2026-10-10: Except a routed down projection on the `_k64` twins: K a multiple of 64
//!   ([`check_w4a4_k64`]).

use anyhow::{Result, bail};
use metrale_config::TpSlice;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::WeightDtype;

use super::weights::{Glm5NextDenseNvfp4Weights, Glm5NextExpertWeights, Nvfp4Proj, W4a4ActScales};

/// 2026-10-08: One layer-relative tensor as `(dtype, shape, bytes)`.
pub type RawFn<'a> = &'a dyn Fn(&str) -> Result<(WeightDtype, Vec<usize>, Vec<u8>)>;

/// 2026-10-08: The largest K the W4A4 activation scratch holds (`w4a4_gemv_mx.cu`, K <= 32768).
pub const W4A4_MAX_K: usize = 32768;

/// 2026-10-08: Refuse a W4A4 projection whose reduction width the mx kernels would mis-read.
pub fn check_w4a4_k(k: usize, what: &str) -> Result<()> {
    if k == 0 || !k.is_multiple_of(128) || k > W4A4_MAX_K {
        bail!(
            "GLM W4A4 {what}: K = {k} must be a positive multiple of 128 and at most \
             {W4A4_MAX_K} (the mx kernels read whole k128 chunks); serve this group with \
             --activation-quantization ffn:bf16 or moe:bf16 instead"
        );
    }
    Ok(())
}

/// 2026-10-10: Refuse a routed down projection the `_k64` twins (`w4a4_gemv_mx_moe.cu`) would
/// mis-read: K must be a positive multiple of 64 whose 128-padded activation width fits.
pub fn check_w4a4_k64(k: usize, what: &str) -> Result<()> {
    if k == 0 || !k.is_multiple_of(64) || k.next_multiple_of(128) > W4A4_MAX_K {
        bail!(
            "GLM W4A4 {what}: K = {k} must be a positive multiple of 64 whose 128-padded width \
             is at most {W4A4_MAX_K} (the _k64 kernels read a half k128 chunk at most); serve \
             this group with --activation-quantization moe:bf16 instead"
        );
    }
    Ok(())
}

/// 2026-10-08: Rows `[start, start + len)` of an NVFP4 `[n, k]` weight: `(packed, scales)`.
pub fn slice_nvfp4_rows(
    packed: &[u8],
    scales: &[u8],
    n: usize,
    k: usize,
    rows: TpSlice,
) -> Result<(Vec<u8>, Vec<u8>)> {
    check_nvfp4_shape(packed, scales, n, k)?;
    if rows.len == 0 || rows.end() > n {
        bail!("NVFP4 row slice {:?} does not fit {n} rows", rows.range());
    }
    let (pb, sb) = (k / 2, k / 16);
    Ok((
        packed[rows.start * pb..rows.end() * pb].to_vec(),
        scales[rows.start * sb..rows.end() * sb].to_vec(),
    ))
}

/// 2026-10-08: Columns `[start, start + len)` of every row of an NVFP4 `[n, k]` weight:
/// `(packed, scales)`. Both ends must sit on a 16-column block boundary.
pub fn slice_nvfp4_cols(
    packed: &[u8],
    scales: &[u8],
    n: usize,
    k: usize,
    cols: TpSlice,
) -> Result<(Vec<u8>, Vec<u8>)> {
    check_nvfp4_shape(packed, scales, n, k)?;
    if cols.len == 0 || cols.end() > k {
        bail!(
            "NVFP4 column slice {:?} does not fit {k} columns",
            cols.range()
        );
    }
    if !cols.start.is_multiple_of(16) || !cols.len.is_multiple_of(16) {
        bail!(
            "NVFP4 column slice {:?} splits a 16-column scale block; the TP split of this \
             width must fall on multiples of 16",
            cols.range()
        );
    }
    let (pb, sb) = (k / 2, k / 16);
    let mut p = Vec::with_capacity(n * cols.len / 2);
    let mut s = Vec::with_capacity(n * cols.len / 16);
    for r in 0..n {
        p.extend_from_slice(&packed[r * pb + cols.start / 2..r * pb + cols.end() / 2]);
        s.extend_from_slice(&scales[r * sb + cols.start / 16..r * sb + cols.end() / 16]);
    }
    Ok((p, s))
}

fn check_nvfp4_shape(packed: &[u8], scales: &[u8], n: usize, k: usize) -> Result<()> {
    if !k.is_multiple_of(16) || packed.len() != n * k / 2 || scales.len() != n * k / 16 {
        bail!(
            "NVFP4 [{n}, {k}]: {} packed and {} scale bytes, expected {} and {}",
            packed.len(),
            scales.len(),
            n * k / 2,
            n * k / 16
        );
    }
    Ok(())
}

/// 2026-10-08: The activation scales of one W4A4 MLP from its projections' `input_scale`s: gate
/// and up read the same input, so their scales must be equal, and every expert of a routed site
/// must carry the same pair (one quantization of a token serves all its slots). The checkpoint
/// this was written for exports one scale per projection and layer; per-expert scales would
/// need a per-slot quantization, which no kernel here does, so they are refused.
pub fn uniform_act_scales<'a>(
    projs: impl IntoIterator<Item = [&'a Nvfp4Proj; 3]>,
    what: &str,
) -> Result<W4a4ActScales> {
    let mut out: Option<W4a4ActScales> = None;
    for (i, [g, u, d]) in projs.into_iter().enumerate() {
        let here = act_scales_of(
            [g.input_scale, u.input_scale, d.input_scale],
            &format!("{what}, projection set {i}"),
        )?;
        match out {
            None => out = Some(here),
            Some(first) if first != here => bail!(
                "GLM W4A4 {what}: projection set {i} has activation scales {here:?}, the first \
                 has {first:?}; per-expert activation scales are not supported"
            ),
            Some(_) => {}
        }
    }
    out.ok_or_else(|| anyhow::anyhow!("GLM W4A4 {what}: no projections"))
}

/// 2026-10-08: The activation scales of one gate/up/down triple: all present, gate equal to
/// up (they quantize the same activations).
pub fn act_scales_of([g, u, d]: [Option<f32>; 3], what: &str) -> Result<W4a4ActScales> {
    let (Some(gs), Some(us), Some(ds)) = (g, u, d) else {
        bail!("GLM W4A4 {what}: no input_scale");
    };
    if gs != us {
        bail!(
            "GLM W4A4 {what}: gate input_scale {gs} and up {us}; they quantize the same \
             activations, so they must be equal"
        );
    }
    Ok(W4a4ActScales {
        gate_up: gs,
        down: ds,
    })
}

/// 2026-10-08: [`uniform_act_scales`] over a routed site's bound experts.
pub fn expert_act_scales(experts: &[Glm5NextExpertWeights]) -> Result<W4a4ActScales> {
    uniform_act_scales(
        experts
            .iter()
            .map(|e| [&e.gate_proj, &e.up_proj, &e.down_proj]),
        "routed experts",
    )
}

/// 2026-10-08: Whether every projection of `experts` carries a static activation scale.
pub fn experts_have_scales(experts: &[Glm5NextExpertWeights]) -> bool {
    experts.iter().all(|e| {
        [&e.gate_proj, &e.up_proj, &e.down_proj]
            .iter()
            .all(|p| p.input_scale.is_some())
    })
}

fn upload(gpu: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(b, p)?;
    Ok(p)
}

/// 2026-10-08: One packed projection `name`: its shape, U8 codes, E4M3 scales and scalar
/// `weight_scale_2`, read through `raw`.
fn raw_proj(raw: RawFn<'_>, name: &str) -> Result<(Vec<usize>, Vec<u8>, Vec<u8>, f32)> {
    let (wd, shape, packed) = raw(&format!("{name}.weight"))?;
    let (sd, _, scales) = raw(&format!("{name}.weight_scale"))?;
    let (s2d, _, s2) = raw(&format!("{name}.weight_scale_2"))?;
    if wd != WeightDtype::UInt8 || sd != WeightDtype::FP8E4M3 || s2d != WeightDtype::FP32 {
        bail!(
            "GLM W4A4 {name}: expected packed U8 codes, F8_E4M3 scales and an F32 \
             weight_scale_2, got {wd:?} / {sd:?} / {s2d:?}"
        );
    }
    let [a, b, c, d] = s2[..] else {
        bail!("GLM W4A4 {name}.weight_scale_2 is not one F32");
    };
    Ok((shape, packed, scales, f32::from_le_bytes([a, b, c, d])))
}

/// 2026-10-08: This rank's packed dense MLP (`cols` of the intermediate width): `gate_proj` /
/// `up_proj` (`[inter, hidden]`) by row, `down_proj` (`[hidden, inter]`) by column, uploaded
/// as the checkpoint stores them. `scales` are the MLP's activation scales.
pub fn build_dense_nvfp4(
    gpu: &dyn GpuBackend,
    hidden: usize,
    full_inter: usize,
    cols: TpSlice,
    prefix: &str,
    raw: RawFn<'_>,
    scales: W4a4ActScales,
) -> Result<Glm5NextDenseNvfp4Weights> {
    check_w4a4_k(hidden, &format!("{prefix} gate/up"))?;
    check_w4a4_k(cols.len, &format!("{prefix} down (this rank's width)"))?;
    let proj = |leaf: &str, n: usize, k: usize, by_row: bool, act: f32| -> Result<Nvfp4Proj> {
        let name = format!("{prefix}.{leaf}");
        let (shape, packed, sc, s2) = raw_proj(raw, &name)?;
        if shape != [n, k / 2] {
            bail!(
                "GLM W4A4 {name}.weight: shape {shape:?}, expected [{n}, {}]",
                k / 2
            );
        }
        let (p, s) = if by_row {
            slice_nvfp4_rows(&packed, &sc, n, k, cols)?
        } else {
            slice_nvfp4_cols(&packed, &sc, n, k, cols)?
        };
        Ok(Nvfp4Proj {
            packed: upload(gpu, &p)?,
            scale: upload(gpu, &s)?,
            scale_2: s2,
            input_scale: Some(act),
        })
    };
    Ok(Glm5NextDenseNvfp4Weights {
        gate_proj: proj("gate_proj", full_inter, hidden, true, scales.gate_up)?,
        up_proj: proj("up_proj", full_inter, hidden, true, scales.gate_up)?,
        down_proj: proj("down_proj", hidden, full_inter, false, scales.down)?,
    })
}

#[cfg(test)]
#[path = "build_w4a4_tests.rs"]
mod build_w4a4_tests;
