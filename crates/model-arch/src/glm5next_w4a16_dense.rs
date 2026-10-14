// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: `--dense-quantization w4a16` for GLM-5.3: the KDA layers' q/k/v, f_a, b, g_a and o
//! projections and the shared expert's gate/up/down served W4A16, FURTHER below the checkpoint's
//! declared BF16 than `fp8`: NVFP4 weights (E2M1 codes, one E4M3 scale per 16 along K, one F32
//! scale per tensor) quantized at load from the bound BF16 weights by the engine's NVFP4 recipe
//! (`weight_map::quantize_to_nvfp4`, the one the NVFP4 lm_head uses), and 16-bit activations on
//! the NVFP4 row-tile tensor-core kernel (`ops::w4a16_tc_rows`). The rest of `fp8`'s set (the DSA
//! latent and indexer projections, KDA f_b and g_b) stays in `glm5next_fp8_dense`, which serves
//! it as under `fp8`; the FP32-output projections stay BF16.
//!
//! The loader quantizes each W4A16 projection, points the layer's weight field at the NVFP4
//! packed buffer, which is also this registry's key, and frees the BF16 copy. Keying by a live
//! allocation the registry owns means no later allocation can alias a key, which a freed BF16
//! address could. `glm_mm` asks [`proj`] before the FP8 registry and the BF16 kernels.
//!
//! f_b and g_b stay FP8: their K is the 128-wide head dim, half the row-tile kernel's 256-wide K
//! unit (and the reference engine keeps them 8-bit too).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - [`proj`] declines (`Ok(false)`) unless the registry is in use ([`registry_in_use`]: the
//!   `w4a16` dense tier, or 2026-10-11 `--dsa-quantization w4a16`) and the weight is registered;
//!   otherwise nothing is registered and every launch is unchanged.
//! - A registered weight runs only at the shape it was registered with; another shape is an
//!   error, never a BF16 run (its field no longer points at BF16).
//! - From 2 rows up a row's output does not depend on the row count: `w4a16_tc_rows` is
//!   row-invariant across its entry points and launches here are chunked by whole rows
//!   ([`W4A16_LAUNCH_ROWS`]). 2026-10-09: one row takes `w4a16_gemv` instead, whose bits differ.
//!   2026-10-10: Under `METRALE_GLM_W4A16_GEMV_ROWS=<r>` launches of 2..=r rows take the GEMV
//!   tiers, whose rows are the one-row GEMV's bytes, so rows 1..=r are row-invariant.
//!   2026-10-10: Under `METRALE_GLM_W4A16_GEMV_SEG=1` one row takes `w4a16_gemv_seg` (grouped),
//!   whose bytes are `w4a16_gemv`'s, so that invariant holds with it.
//! - 2026-10-09: `METRALE_GLM_W4A16_SEG` (`glm5next_w4a16_seg`) moves the launches of 2 rows
//!   and up to the segmented row tiles; [`proj_group`] runs projections that share their input
//!   in one launch there and declines otherwise. Unset, every launch is as before.
//! - 2026-10-10: `METRALE_GLM_W4A16_DEQUANT_GEMM_MIN_ROWS` (`glm5next_w4a16_wide`) moves launches
//!   of at least that many rows (prefill) to a dequantize + cuBLASLt GEMM, ahead of every other
//!   mode; unset, every launch is as before.
//! - Every registered K is a multiple of [`W4A16_K_UNIT`]; the TP splits are chosen so that it
//!   holds ([`kda_channel_unit`], `glm5next_fp8_dense::shared_split_unit`), and a width that
//!   breaks it is refused at load.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops::{
    W4A16_TC_ROWS_MAX_M, W4A16_TC_ROWS_MODULE, W4a16Seg, w4a16_gemv_seg_lone, w4a16_tc_rows,
    w4a16_tc_rows_entry, w4a16_tc_rows_shape_ok,
};
use metrale_model_layers::weight_map::{DenseWeight, QuantizedWeight, quantize_to_nvfp4};

use crate::glm5next_w4a16_seg::{self as seg, SegMode, SegPlan};

/// 2026-10-09: The row-tile kernel's K unit (`w4a16_tc_rows_shape_ok`: K a multiple of 256).
pub const W4A16_K_UNIT: usize = 256;

/// 2026-10-09: Rows per `w4a16_tc_rows` launch: its widest entry point.
pub const W4A16_LAUNCH_ROWS: usize = W4A16_TC_ROWS_MAX_M as usize;

/// 2026-10-09: Whether `--dense-quantization w4a16` is in force.
pub fn enabled() -> bool {
    metrale_model_layers::layers::dense_quantization()
        == metrale_model_layers::layers::DenseQuantization::W4a16
}

/// 2026-10-09: The channel unit the KDA heads split over TP in (`TpSupport::Uneven`'s
/// `linear_channel_unit`): [`W4A16_K_UNIT`] under the tier, so every rank's o_proj K is whole
/// row-tile units (128-wide heads in pairs: 22/22/20 of 64 at TP=3, the widest rank unchanged),
/// else 1, the head-by-head split.
pub fn kda_channel_unit() -> usize {
    kda_channel_unit_for(enabled())
}

fn kda_channel_unit_for(w4a16: bool) -> usize {
    if w4a16 { W4A16_K_UNIT } else { 1 }
}

/// 2026-10-09: Which registry serves a projection under the `w4a16` tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjTier {
    /// 2026-10-09: NVFP4 weights, 16-bit activations (this module).
    W4a16,
    /// 2026-10-09: FP8 W8A8, as under `fp8` (`glm5next_fp8_dense`).
    Fp8,
}

/// 2026-10-11: Whether this registry is in use: `--dense-quantization w4a16`, or
/// `--dsa-quantization w4a16` (the DSA q_a / absorbed q / absorbed o projections alone).
/// [`enabled`] stays the dense tier itself, which also shapes the KDA head split and the
/// segmented-GEMV levers.
pub fn registry_in_use() -> bool {
    enabled()
        || metrale_model_layers::layers::dsa_quantization()
            == metrale_model_layers::layers::DsaQuantization::W4a16
}

/// 2026-10-11: The DSA projections `--dsa-quantization` decides (`dense_tiers::tier_for`),
/// whatever `--dense-quantization` is.
pub const DSA_SITES: [&str; 3] = ["dsa.q_a_proj", "dsa.q_absorb", "dsa.o_absorb"];

/// 2026-10-09: The tier of the projection the loader names `name` under `--dense-quantization
/// w4a16` (the names of `Glm5NextKdaLayer::dense_projections`, `Glm5NextDsaLayer::dense_projections`
/// and the shared expert's). An unknown name is an error, so a projection added to those lists
/// gets a decision here before it loads. 2026-10-11: the [`DSA_SITES`] answer `Fp8` here; their
/// tier is `--dsa-quantization`'s (`dense_tiers::tier_for` asks that first).
pub fn tier_of(name: &str) -> Result<ProjTier> {
    Ok(match name {
        "kda.q_proj"
        | "kda.k_proj"
        | "kda.v_proj"
        | "kda.f_a_proj"
        | "kda.b_proj"
        | "kda.g_a_proj"
        | "kda.o_proj"
        | "shared_experts.gate_proj"
        | "shared_experts.up_proj"
        | "shared_experts.down_proj" => ProjTier::W4a16,
        "kda.f_b_proj"
        | "kda.g_b_proj"
        | "dsa.q_a_proj"
        | "dsa.q_absorb"
        | "dsa.kv_a_proj"
        | "dsa.o_absorb"
        | "dsa.indexer.wk"
        | "dsa.indexer.compress_gate" => ProjTier::Fp8,
        other => bail!("--dense-quantization w4a16: no tier decided for projection {other:?}"),
    })
}

/// 2026-10-09: The two load-time NVFP4 quantization kernels `quantize_to_nvfp4` takes.
#[derive(Clone, Copy, Debug)]
pub struct Nvfp4QuantKernels {
    pub absmax: KernelHandle,
    pub quantize: KernelHandle,
}

impl Nvfp4QuantKernels {
    /// 2026-10-09: Resolve them, and the row-tile entry points [`proj`] launches (the
    /// segmented ones too under `METRALE_GLM_W4A16_SEG`), failing when any is not in this
    /// target's kernels: the operator asked for the tier.
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        seg::prepare(gpu)?;
        for rows in [16, 32, 64] {
            let entry = w4a16_tc_rows_entry(rows);
            gpu.op_cache()
                .kernel(gpu, W4A16_TC_ROWS_MODULE, entry)
                .with_context(|| {
                    format!("--dense-quantization w4a16: {W4A16_TC_ROWS_MODULE}::{entry}")
                })?;
        }
        Ok(Self {
            absmax: gpu
                .kernel("quantize_nvfp4", "nvfp4_global_absmax")
                .context("--dense-quantization w4a16")?,
            quantize: gpu
                .kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")
                .context("--dense-quantization w4a16")?,
        })
    }
}

/// 2026-10-09: One registered projection: its NVFP4 weight and `[n, k]` shape.
#[derive(Clone, Copy)]
struct Entry {
    w: QuantizedWeight,
    n: usize,
    k: usize,
}

fn state() -> &'static Mutex<HashMap<u64, Entry>> {
    static S: OnceLock<Mutex<HashMap<u64, Entry>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(HashMap::new()))
}

/// 2026-10-09: Empty the registry (`glm5next_fp8_dense::lock_registries_for_test`).
#[cfg(test)]
pub(crate) fn clear_for_test() {
    state().lock().unwrap_or_else(|e| e.into_inner()).clear();
}

/// 2026-10-09: Whether `[n, k]` is a shape the tier serves: positive, K whole row-tile units,
/// and inside the kernel's contract for contiguous rows.
pub fn shape_ok(n: usize, k: usize) -> bool {
    let (Ok(n32), Ok(k32)) = (u32::try_from(n), u32::try_from(k)) else {
        return false;
    };
    k.is_multiple_of(W4A16_K_UNIT) && w4a16_tc_rows_shape_ok(1, n32, k32, k32, n32)
}

/// 2026-10-09: Quantize the BF16 `[n, k]` weight at `bf16` to NVFP4 and register it. Returns the
/// registry key, the packed buffer's address, which the caller writes into the layer's weight
/// field before freeing `bf16`. The quantization has finished on `stream` when this returns
/// (`quantize_to_nvfp4` synchronizes). Errors on a shape [`shape_ok`] refuses.
pub fn register(
    gpu: &dyn GpuBackend,
    kernels: &Nvfp4QuantKernels,
    bf16: DevicePtr,
    n: usize,
    k: usize,
    what: &str,
    stream: u64,
) -> Result<DevicePtr> {
    ensure!(
        shape_ok(n, k),
        "--dense-quantization w4a16: {what} is [{n}, {k}]; K must be a positive multiple of \
         {W4A16_K_UNIT} (the W4A16 row-tile kernel's unit)"
    );
    let w = quantize_to_nvfp4(
        &DenseWeight { weight: bf16 },
        n,
        k,
        gpu,
        kernels.absmax,
        kernels.quantize,
        stream,
    )
    .with_context(|| format!("--dense-quantization w4a16: quantizing {what}"))?;
    let mut g = state().lock().unwrap_or_else(|e| e.into_inner());
    ensure!(
        !g.contains_key(&w.weight.0),
        "--dense-quantization w4a16: {what}'s packed buffer {:#x} is already a key",
        w.weight.0
    );
    g.insert(w.weight.0, Entry { w, n, k });
    drop(g);
    crate::glm5next_w4a16_wide::reserve(gpu, n, k)?;
    Ok(w.weight)
}

/// 2026-10-09: Registered projections, their NVFP4 bytes (packed codes plus E4M3 scales), and
/// the BF16 bytes they replace, for the load log.
pub fn registered() -> (usize, usize, usize) {
    let g = state().lock().unwrap_or_else(|e| e.into_inner());
    g.values().fold((0, 0, 0), |(c, q, b), e| {
        (c + 1, q + e.n * e.k / 2 + e.n * e.k / 16, b + e.n * e.k * 2)
    })
}

/// 2026-10-10: The device bytes the tier's launches read for the projection registered under
/// `b`: its packed E2M1 codes (`n * k / 2`) and its FP8 group scales (`n * k / 16`), the sizes
/// [`registered`] counts. `None` when `b` is not registered. For a cache warm-up only
/// (`glm5next_layer::l2_warm`); nothing writes through these ranges.
pub fn footprint(b: DevicePtr) -> Option<[(DevicePtr, usize); 2]> {
    let g = state().lock().unwrap_or_else(|e| e.into_inner());
    g.get(&b.0).map(|e| {
        [
            (e.w.weight, e.n * e.k / 2),
            (e.w.weight_scale, e.n * e.k / 16),
        ]
    })
}

/// 2026-10-09: Point the one slot holding `from` at `to`. Errors unless exactly one slot holds
/// `from`: none means the loader's projection list and the layer disagree, two that one buffer
/// backs two projections, and freeing it would leave the other dangling.
pub fn retarget(slots: &mut [&mut DevicePtr], from: DevicePtr, to: DevicePtr) -> Result<()> {
    let mut hits = slots.iter_mut().filter(|s| ***s == from);
    let Some(slot) = hits.next() else {
        bail!("--dense-quantization w4a16: no weight field holds {from}");
    };
    **slot = to;
    ensure!(
        hits.next().is_none(),
        "--dense-quantization w4a16: two weight fields hold {from}"
    );
    Ok(())
}

/// 2026-10-09: `c[m, n] = a[m, k] @ w[n, k]^T` (contiguous BF16 rows) on the NVFP4 weight
/// registered under `b`, in launches of at most [`W4A16_LAUNCH_ROWS`] rows. `Ok(false)` when the
/// tier is off or `b` is not registered: the caller tries the FP8 registry, then BF16.
#[allow(clippy::too_many_arguments)]
pub fn proj(
    gpu: &dyn GpuBackend,
    b: DevicePtr,
    a: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<bool> {
    if !registry_in_use() {
        return Ok(false);
    }
    proj_registered(gpu, seg::plan()?, b, a, c, m, n, k, stream)
}

/// 2026-10-09: The registered entry under `b`, checked against the launch shape `[n, k]`.
fn lookup(b: DevicePtr, n: usize, k: usize) -> Result<Option<Entry>> {
    let g = state().lock().unwrap_or_else(|e| e.into_inner());
    let Some(&e) = g.get(&b.0) else {
        return Ok(None);
    };
    ensure!(
        (e.n, e.k) == (n, k),
        "--dense-quantization w4a16: weight {:#x} registered as [{}, {}], launched as [{n}, {k}]",
        b.0,
        e.n,
        e.k
    );
    Ok(Some(e))
}

/// 2026-10-09: [`proj`] whatever the published tier, under `plan`.
#[allow(clippy::too_many_arguments)]
fn proj_registered(
    gpu: &dyn GpuBackend,
    plan: SegPlan,
    b: DevicePtr,
    a: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<bool> {
    let Some(e) = lookup(b, n, k)? else {
        return Ok(false);
    };
    let (n32, k32) = (n as u32, k as u32);
    // 2026-10-11: A lone projection takes the segmented GEMV only below K 2048 (KDA o, K 2816,
    // is faster on `w4a16_gemv`; `ops::w4a16_gemv_seg_lone`).
    if m == 1 && plan.one_row && w4a16_gemv_seg_lone(k32) {
        let segs = [W4a16Seg {
            weight: e.w,
            output: c,
            n: n32,
        }];
        seg::launch_one_row(gpu, a, &segs, k32, plan.sms, stream)?;
        return Ok(true);
    }
    // 2026-10-12: `METRALE_GLM_W4A16_TC_GEMM_MIN_ROWS`: one launch of the row tiles' bits.
    if crate::glm5next_w4a16_tc::takes(crate::glm5next_w4a16_tc::min_rows()?, m) {
        crate::glm5next_w4a16_tc::proj(gpu, &e.w, a, c, m, n, k, stream)?;
        return Ok(true);
    }
    if crate::glm5next_w4a16_wide::takes(crate::glm5next_w4a16_wide::min_rows()?, m) {
        crate::glm5next_w4a16_wide::proj(gpu, &e.w, a, c, m, n, k, stream)?;
        return Ok(true);
    }
    if m == 1 {
        // 2026-10-11: Under `METRALE_GLM_BATCH_INVARIANT`, the row tile's order (as 2+ rows).
        if crate::glm5next_batch_inv::enabled()? {
            return crate::glm5next_batch_inv::one_row(gpu, &e.w, a, c, n32, k32, stream)
                .map(|()| true);
        }
        // 2026-10-09: One row takes the NVFP4 GEMV (faster; not the bits of a 2+ row launch).
        let kernel = gpu
            .op_cache()
            .kernel(gpu, "w4a16_gemv", "w4a16_gemv")
            .context("--dense-quantization w4a16: w4a16_gemv::w4a16_gemv")?;
        metrale_model_layers::layers::ops::w4a16_gemv(gpu, kernel, a, &e.w, c, n32, k32, stream)?;
        return Ok(true);
    }
    if plan.gemv_tier(m) {
        // 2026-10-11: Under `METRALE_GLM_W4A16_GEMV_SEG`, 2..=4 rows of a lone projection below
        // K 2048 take the M-row segmented GEMV, whose rows are these tiers' bytes (each the
        // one-row GEMV's); from K 2048 up the tier is as fast or faster (KDA o, DSA q_a and
        // o_absorb; `ops::w4a16_gemv_seg_lone`).
        if plan.seg_gemv(m) && w4a16_gemv_seg_lone(k32) {
            let segs = [W4a16Seg {
                weight: e.w,
                output: c,
                n: n32,
            }];
            seg::launch_seg_gemv(gpu, a, &segs, m, k32, plan.sms, stream)?;
        } else {
            seg::launch_gemv_rows(gpu, a, &e.w, c, m, n32, k32, stream)?;
        }
        return Ok(true);
    }
    if plan.mode != SegMode::Off {
        let segs = [W4a16Seg {
            weight: e.w,
            output: c,
            n: n32,
        }];
        seg::launch(gpu, a, &segs, m, k32, plan.split(&[n32], k32), stream)?;
        return Ok(true);
    }
    let mut done = 0;
    while done < m {
        let rows = (m - done).min(W4A16_LAUNCH_ROWS);
        w4a16_tc_rows(
            gpu,
            a.offset(done * k * 2),
            &e.w,
            c.offset(done * n * 2),
            rows as u32,
            n32,
            k32,
            k32,
            n32,
            stream,
        )?;
        done += rows;
    }
    Ok(true)
}

/// 2026-10-09: `members[i] = (weight, output, n_i)`: `output_i [m, n_i] = a [m, k] @ W_i^T` for
/// projections that share the input `a`, in one launch per 64 rows (`glm5next_w4a16_seg`);
/// 2026-10-10: at one row, under `METRALE_GLM_W4A16_GEMV_SEG`, one `w4a16_gemv_seg` launch of up
/// to six members (more than three only there).
/// `Ok(false)`, launching nothing, unless the tier is on, `METRALE_GLM_W4A16_SEG` is not `off`,
/// `m` is 2 or more (one row keeps the GEMV) and every member is registered: the caller then
/// runs them one by one. A registered member at another shape is an error. 2026-10-10: Also
/// `Ok(false)` at a row count `METRALE_GLM_W4A16_GEMV_ROWS` covers.
pub fn proj_group(
    gpu: &dyn GpuBackend,
    members: &[(DevicePtr, DevicePtr, usize)],
    a: DevicePtr,
    m: usize,
    k: usize,
    stream: u64,
) -> Result<bool> {
    if !enabled() {
        return Ok(false);
    }
    proj_group_registered(gpu, seg::plan()?, members, a, m, k, stream)
}

/// 2026-10-11: Whether a group of `m` rows runs as one `w4a16_gemv_seg` launch under `plan`
/// (`SegPlan::seg_gemv`, unless the dequantize + cuBLASLt path takes the rows); then up to six
/// members, else at most the row tiles' three.
fn seg_gemv_group(plan: SegPlan, m: usize) -> Result<bool> {
    Ok(plan.seg_gemv(m)
        && (m == 1
            || !(crate::glm5next_w4a16_wide::takes(crate::glm5next_w4a16_wide::min_rows()?, m)
                || crate::glm5next_w4a16_tc::takes(crate::glm5next_w4a16_tc::min_rows()?, m))))
}

/// 2026-10-11: Whether a group of `m` rows would run as one launch of up to six members (the KDA
/// front end then groups q/k/v/f_a/b/g_a): the tier is on and [`seg_gemv_group`] holds under the
/// published plan.
pub fn group_of_six(m: usize) -> Result<bool> {
    Ok(enabled() && seg_gemv_group(seg::plan()?, m)?)
}

/// 2026-10-09: [`proj_group`] whatever the published tier, under `plan`.
fn proj_group_registered(
    gpu: &dyn GpuBackend,
    plan: SegPlan,
    members: &[(DevicePtr, DevicePtr, usize)],
    a: DevicePtr,
    m: usize,
    k: usize,
    stream: u64,
) -> Result<bool> {
    // 2026-10-10: A launch the GEMV tiers take declines too: its members then run one by one
    // on them (`proj`). One row runs as a group only under `METRALE_GLM_W4A16_GEMV_SEG`.
    // 2026-10-10: A launch the dequantize + cuBLASLt path takes declines too: its members then
    // run one by one there (`proj`).
    // 2026-10-11: Under `METRALE_GLM_W4A16_GEMV_SEG` a group of 1..=4 GEMV-tier rows is one
    // `w4a16_gemv_seg` launch ([`seg_gemv_group`]).
    let seg_gemv = seg_gemv_group(plan, m)?;
    if !seg_gemv
        && (plan.mode == SegMode::Off
            || (m < 2 && !crate::glm5next_batch_inv::enabled()?)
            || plan.gemv_tier(m)
            || crate::glm5next_w4a16_wide::takes(crate::glm5next_w4a16_wide::min_rows()?, m)
            || crate::glm5next_w4a16_tc::takes(crate::glm5next_w4a16_tc::min_rows()?, m))
    {
        return Ok(false);
    }
    let mut segs = Vec::with_capacity(members.len());
    for &(b, c, n) in members {
        let Some(e) = lookup(b, n, k)? else {
            return Ok(false);
        };
        segs.push(W4a16Seg {
            weight: e.w,
            output: c,
            n: n as u32,
        });
    }
    let k32 = k as u32;
    if seg_gemv {
        seg::launch_seg_gemv(gpu, a, &segs, m, k32, plan.sms, stream)?;
        return Ok(true);
    }
    let ns: Vec<u32> = segs.iter().map(|g| g.n).collect();
    seg::launch(gpu, a, &segs, m, k32, plan.split(&ns, k32), stream)?;
    Ok(true)
}

#[cfg(test)]
#[path = "glm5next_w4a16_dense_tests.rs"]
mod tests;
