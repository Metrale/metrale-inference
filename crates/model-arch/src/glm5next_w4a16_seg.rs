// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: `METRALE_GLM_W4A16_SEG`: the W4A16 tier's launches of 2 rows and up on the
//! segmented row tiles (`ops::w4a16_tc_rows_seg`): projections that share their input in one
//! launch (`glm5next_w4a16_dense::proj_group`), and under `split` a K-split per shape.
//!
//! At 2..16 rows the 64-column row tile left at most one 4-warp block per SM and narrow
//! projections on a few SMs. GB10 microbench, 2026-10-09 (cold weights, CUDA-graph replay, 4..16
//! rows, best of 3 sweeps): shared gate+up 2 x 768 x 4096 from two launches, 29.7-33.6 us, to one
//! of 19.2-20.0 us (split 2); KDA f_a+g_a+b from three, 37.1-46.7 us, to one of 6.8-7.8 us (split
//! 4); KDA q/k/v 92.4-97.9 us as three, 93.8-99.8 us as one (unsplit: already near bandwidth).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - `off` (unset or `0`) launches exactly what the tier launched before this lever.
//! - `fuse`: one launch per group, unsplit; every output byte equals `off`'s.
//! - `split`: the split of a launch is `ops::w4a16_tc_rows_split_for` of its column tiles and K
//!   on this device, fixed per shape (never per row count), so CUDA graphs replay one choice and
//!   a row's bits do not depend on the row count; a split launch's bits differ from `off`'s
//!   (another FP32 summation order), so it is opt-in and accuracy-gated.
//! - One row keeps the tier's GEMV in every mode.
//! - 2026-10-10: `METRALE_GLM_W4A16_GEMV_ROWS=<r>` (2..=8; unset or `0` is off) runs launches of
//!   2..=r rows on the CUDA-core GEMV tiers (`w4a16_gemv_batch{m}`, [`launch_gemv_rows`]) ahead
//!   of every mode: each row's bytes are the one-row `w4a16_gemv`'s (the kernels' invariant,
//!   `w4a16_gemv.cu`), so a verify or a batched decode of up to r rows computes the dense
//!   projections bit for bit as the single-row decode. Opt-in; wider launches are unchanged.
//! - 2026-10-10: `METRALE_GLM_W4A16_GEMV_SEG=1` runs one-row launches on `w4a16_gemv_seg`
//!   ([`launch_one_row`]) in every mode: a group of projections that share the input row is one
//!   persistent launch, a lone projection below K 2048 a one-segment launch (2026-10-11: from K
//!   2048 up a lone projection keeps `w4a16_gemv`, `ops::w4a16_gemv_seg_lone`). Every output byte is
//!   `w4a16_gemv`'s (the kernel's invariant), so the lever changes no bits; unset or `0` keeps
//!   `w4a16_gemv`.
//! - 2026-10-21: `METRALE_GLM_W4A16_GEMV_PIPE` (on unless `0`), under `METRALE_GLM_W4A16_GEMV_SEG`:
//!   the launches of 1 and 2 rows [`launch_seg_gemv`] makes run on `w4a16_gemv_pipe`
//!   ([`SegPlan::pipe_rows`]), as does a lone 2-row projection from K 2048 up (KDA o), which ran
//!   on `w4a16_gemv_batch2`. Every output byte is `w4a16_gemv`'s per row (the kernel's
//!   invariant), so the lever changes no bits; `0` keeps `w4a16_gemv_seg` and the batch tier.

use std::sync::OnceLock;

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use metrale_model_layers::layers::ops::{
    W4A16_GEMV_PIPE_MODULE, W4A16_GEMV_SEG_MAX_ROWS, W4A16_GEMV_SEG_MODULE, W4A16_GEMV_SEG_TILES, W4A16_TC_ROWS_MAX_M,
    W4A16_TC_ROWS_SEG_MODULE, W4a16GemvSegPoint, W4a16Seg, w4a16_gemv_seg, w4a16_gemv_seg_point,
    w4a16_gemv_pipe, w4a16_gemv_pipe_entry, w4a16_gemv_seg_policy, w4a16_tc_rows_seg, w4a16_tc_rows_seg_entries, w4a16_tc_rows_seg_tiles,
    w4a16_tc_rows_split_for,
};
use metrale_model_layers::weight_map::QuantizedWeight;

use crate::glm5next_dense_levers::DenseLevers;

/// 2026-10-09: What `METRALE_GLM_W4A16_SEG` selects (`DenseLevers::seg`): unset or `0` is
/// [`SegMode::Off`], `fuse` and `split` the others.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegMode {
    /// 2026-10-09: The tier's launches as before the lever.
    Off,
    /// 2026-10-09: Groups in one launch, unsplit (the same bytes as `Off`).
    Fuse,
    /// 2026-10-09: Groups in one launch, every launch K-split per shape.
    Split,
}

/// 2026-10-10: The widest row count `METRALE_GLM_W4A16_GEMV_ROWS` may name: the widest
/// CUDA-core tier `w4a16_gemv_batch8`.
pub const GEMV_ROWS_MAX: u32 = 8;

/// 2026-10-10: The CUDA-core GEMV tier of exactly `m` rows (2..=8): `w4a16_gemv_batch2` and
/// `_batch3` take no row argument, `_batch4`..`_batch8` take `m`. Never the tensor-core route
/// of `ops::w4a16_gemv_batchm`, whose bytes differ from the one-row GEMV's.
const GEMV_ROW_TIERS: [&str; 7] = [
    "w4a16_gemv_batch2",
    "w4a16_gemv_batch3",
    "w4a16_gemv_batch4",
    "w4a16_gemv_batch5",
    "w4a16_gemv_batch6",
    "w4a16_gemv_batch7",
    "w4a16_gemv_batch8",
];

/// 2026-10-10: `c [m, n] = a [m, k] @ W^T` (contiguous rows) on the GEMV tier of `m` rows,
/// 2..=[`GEMV_ROWS_MAX`]; an error outside that range.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_gemv_rows(
    gpu: &dyn GpuBackend,
    a: DevicePtr,
    w: &QuantizedWeight,
    c: DevicePtr,
    m: usize,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let Some(&name) = m.checked_sub(2).and_then(|i| GEMV_ROW_TIERS.get(i)) else {
        anyhow::bail!("W4A16 GEMV rows: {m} rows outside 2..={GEMV_ROWS_MAX}");
    };
    let kernel = gpu
        .op_cache()
        .kernel(gpu, "w4a16_gemv", name)
        .with_context(|| format!("METRALE_GLM_W4A16_GEMV_ROWS: w4a16_gemv::{name}"))?;
    let l = KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w.weight)
        .arg_ptr(w.weight_scale)
        .arg_f32(w.weight_scale_2)
        .arg_ptr(c);
    if m <= 3 {
        l.arg_u32(n).arg_u32(k).launch(stream)
    } else {
        l.arg_u32(m as u32).arg_u32(n).arg_u32(k).launch(stream)
    }
}

/// 2026-10-11: `segs[i].output [m, n_i] = a [m, k] @ W_i^T` (contiguous rows) in one
/// `w4a16_gemv_seg` launch: [`launch_one_row`] at one row, the M-row point at 2..=4 (each row the
/// one-row kernel's bytes for that row alone). 2026-10-21: On `w4a16_gemv_pipe` where
/// [`SegPlan::pipe_rows`] holds (the same bytes).
pub(crate) fn launch_seg_gemv(
    gpu: &dyn GpuBackend,
    a: DevicePtr,
    segs: &[W4a16Seg],
    m: usize,
    k: u32,
    plan: SegPlan,
    stream: u64,
) -> Result<()> {
    if plan.pipe_rows(m) {
        return w4a16_gemv_pipe(gpu, a, segs, m as u32, k, plan.sms, stream);
    }
    if m == 1 {
        return launch_one_row(gpu, a, segs, k, plan.sms, stream);
    }
    w4a16_gemv_seg_point(
        gpu,
        W4a16GemvSegPoint::Rows(m as u32),
        a,
        segs,
        k,
        u32::MAX,
        stream,
    )
}

/// 2026-10-10: `segs[i].output [n_i] = a [k] @ W_i^T` for one row, in one `w4a16_gemv_seg`
/// launch (1..=6 segments) with the tile height and block cap of `w4a16_gemv_seg_policy`.
pub(crate) fn launch_one_row(
    gpu: &dyn GpuBackend,
    a: DevicePtr,
    segs: &[W4a16Seg],
    k: u32,
    sms: u32,
    stream: u64,
) -> Result<()> {
    let (r, max_blocks) = w4a16_gemv_seg_policy(k, sms);
    w4a16_gemv_seg(gpu, a, segs, k, r, max_blocks, stream)
}

/// 2026-10-09: The SM count the split rule reads, recorded by [`prepare`].
static SMS: OnceLock<u32> = OnceLock::new();

/// 2026-10-09: At load, under a mode other than `off`: resolve every entry point (a missing one
/// is refused here, not at the first launch) and record the SM count. Never inside a capture.
/// 2026-10-10: Under `METRALE_GLM_W4A16_GEMV_ROWS`, also the GEMV tiers it reaches.
pub(crate) fn prepare(gpu: &dyn GpuBackend, l: &DenseLevers) -> Result<()> {
    // 2026-10-10: The GEMV tiers `METRALE_GLM_W4A16_GEMV_ROWS` reaches, resolved here so a
    // missing one is refused at load.
    for &name in GEMV_ROW_TIERS
        .iter()
        .take(l.gemv_rows.saturating_sub(1) as usize)
    {
        gpu.op_cache()
            .kernel(gpu, "w4a16_gemv", name)
            .with_context(|| format!("METRALE_GLM_W4A16_GEMV_ROWS: w4a16_gemv::{name}"))?;
    }
    // 2026-10-10: Every one-row entry `METRALE_GLM_W4A16_GEMV_SEG` may reach. 2026-10-11: and the
    // M-row points of the GEMV-tier row counts (2..=min(4, METRALE_GLM_W4A16_GEMV_ROWS)).
    if l.gemv_seg {
        let rows = (2..=l.gemv_rows.min(W4A16_GEMV_SEG_MAX_ROWS)).map(W4a16GemvSegPoint::Rows);
        for pt in W4A16_GEMV_SEG_TILES
            .into_iter()
            .map(W4a16GemvSegPoint::Tile)
            .chain(rows)
        {
            let entry = pt
                .entry()
                .with_context(|| format!("METRALE_GLM_W4A16_GEMV_SEG: {pt:?}"))?;
            gpu.op_cache()
                .kernel(gpu, W4A16_GEMV_SEG_MODULE, entry)
                .with_context(|| {
                    format!("METRALE_GLM_W4A16_GEMV_SEG: {W4A16_GEMV_SEG_MODULE}::{entry}")
                })?;
        }
    }
    // 2026-10-21: The `w4a16_gemv_pipe` entries `METRALE_GLM_W4A16_GEMV_PIPE` may reach.
    if l.gemv_seg && l.gemv_pipe {
        for entry in (1..=l.gemv_rows.max(1)).filter_map(w4a16_gemv_pipe_entry) {
            gpu.op_cache()
                .kernel(gpu, W4A16_GEMV_PIPE_MODULE, entry)
                .with_context(|| {
                    format!("METRALE_GLM_W4A16_GEMV_PIPE: {W4A16_GEMV_PIPE_MODULE}::{entry}")
                })?;
        }
    }
    if SMS.get().is_none() && (l.gemv_seg || l.seg != SegMode::Off) {
        let _ = SMS.set(gpu.sm_count()?);
    }
    if l.seg == SegMode::Off {
        return Ok(());
    }
    for entry in w4a16_tc_rows_seg_entries() {
        gpu.op_cache()
            .kernel(gpu, W4A16_TC_ROWS_SEG_MODULE, entry)
            .with_context(|| {
                format!("METRALE_GLM_W4A16_SEG: {W4A16_TC_ROWS_SEG_MODULE}::{entry}")
            })?;
    }
    Ok(())
}

/// 2026-10-09: The mode and the SM count its split rule reads. 2026-10-10: `gemv_rows`, the
/// widest launch on the GEMV tiers (`DenseLevers::gemv_rows`; 0 = none). 2026-10-10: `one_row`,
/// whether one-row launches take `w4a16_gemv_seg` (`DenseLevers::gemv_seg`). 2026-10-17: the
/// row floors of the wide dequant GEMM and the tensor-core GEMM (`None` off) and whether the
/// batch-invariant mode is on, copied from the same levers. 2026-10-21: `pipe`,
/// `DenseLevers::gemv_pipe`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegPlan {
    pub mode: SegMode,
    pub sms: u32,
    pub gemv_rows: u32,
    pub one_row: bool,
    pub pipe: bool,
    pub tc_min_rows: Option<usize>,
    pub dequant_min_rows: Option<usize>,
    pub batch_invariant: bool,
}

impl SegPlan {
    /// 2026-10-11: Whether a launch of `m` rows runs on `w4a16_gemv_seg` under
    /// `METRALE_GLM_W4A16_GEMV_SEG`: one row, or a GEMV-tier row count
    /// ([`Self::gemv_tier`]) the M-row points have (2..=4). Their rows carry the GEMV tiers' bytes.
    pub(crate) fn seg_gemv(&self, m: usize) -> bool {
        self.one_row && (m == 1 || (self.gemv_tier(m) && m <= W4A16_GEMV_SEG_MAX_ROWS as usize))
    }

    /// 2026-10-21: Whether a launch of `m` rows that [`Self::seg_gemv`] takes runs on
    /// `w4a16_gemv_pipe`: the lever on and an entry for `m` (1 and 2 rows).
    pub(crate) fn pipe_rows(&self, m: usize) -> bool {
        self.pipe
            && self.seg_gemv(m)
            && u32::try_from(m).is_ok_and(|m| w4a16_gemv_pipe_entry(m).is_some())
    }

    /// 2026-10-10: Whether a launch of `m` rows runs on the GEMV tiers.
    pub(crate) fn gemv_tier(&self, m: usize) -> bool {
        m >= 2 && m <= self.gemv_rows as usize
    }

    /// 2026-10-09: The split for a launch over segments `ns` with K `k`.
    pub(crate) fn split(&self, ns: &[u32], k: u32) -> u32 {
        match self.mode {
            SegMode::Off | SegMode::Fuse => 1,
            SegMode::Split => w4a16_tc_rows_split_for(w4a16_tc_rows_seg_tiles(ns), k, self.sms),
        }
    }
}

/// 2026-10-09: The published plan: the mode of `l` and, unless it is `off`, the SM count
/// [`prepare`] recorded (an error when it did not run).
pub(crate) fn plan(l: &DenseLevers) -> Result<SegPlan> {
    let (mode, one_row) = (l.seg, l.gemv_seg);
    let sms = if mode == SegMode::Off && !one_row {
        0
    } else {
        *SMS.get()
            .context("METRALE_GLM_W4A16_SEG / _GEMV_SEG: the SM count was not recorded at load")?
    };
    Ok(SegPlan {
        mode,
        sms,
        gemv_rows: l.gemv_rows,
        one_row,
        pipe: l.gemv_pipe,
        tc_min_rows: l.tc_min_rows,
        dequant_min_rows: l.dequant_min_rows,
        batch_invariant: l.batch_invariant,
    })
}

/// 2026-10-09: `segs[i].output [m, n_i] = a [m, k] @ W_i^T` for contiguous rows, in launches of
/// at most `W4A16_TC_ROWS_MAX_M` rows over consecutive row ranges, each over every segment with
/// split `s`.
pub(crate) fn launch(
    gpu: &dyn GpuBackend,
    a: DevicePtr,
    segs: &[W4a16Seg],
    m: usize,
    k: u32,
    s: u32,
    stream: u64,
) -> Result<()> {
    let max = W4A16_TC_ROWS_MAX_M as usize;
    let mut done = 0;
    while done < m {
        let rows = (m - done).min(max);
        let chunk: Vec<W4a16Seg> = segs
            .iter()
            .map(|g| W4a16Seg {
                output: g.output.offset(done * g.n as usize * 2),
                ..*g
            })
            .collect();
        w4a16_tc_rows_seg(
            gpu,
            a.offset(done * k as usize * 2),
            &chunk,
            rows as u32,
            k,
            k,
            s,
            stream,
        )?;
        done += rows;
    }
    Ok(())
}
