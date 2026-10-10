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

use std::sync::OnceLock;

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops::{
    W4A16_TC_ROWS_MAX_M, W4A16_TC_ROWS_SEG_MODULE, W4a16Seg, w4a16_tc_rows_seg,
    w4a16_tc_rows_seg_entries, w4a16_tc_rows_seg_tiles, w4a16_tc_rows_split_for,
};

/// 2026-10-09: What [`seg_mode`] selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegMode {
    /// 2026-10-09: The tier's launches as before the lever.
    Off,
    /// 2026-10-09: Groups in one launch, unsplit (the same bytes as `Off`).
    Fuse,
    /// 2026-10-09: Groups in one launch, every launch K-split per shape.
    Split,
}

/// 2026-10-09: `METRALE_GLM_W4A16_SEG`, read once: unset or `0` is [`SegMode::Off`], `fuse` and
/// `split` the others; another value, or `fuse`/`split` without `--dense-quantization w4a16`,
/// is an error.
pub fn seg_mode() -> Result<SegMode> {
    static V: OnceLock<Result<SegMode, String>> = OnceLock::new();
    V.get_or_init(|| {
        parse_seg(
            std::env::var("METRALE_GLM_W4A16_SEG").ok().as_deref(),
            crate::glm5next_w4a16_dense::enabled(),
        )
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// 2026-10-09: [`seg_mode`]'s parse, pure: the value and whether the tier is on.
pub(crate) fn parse_seg(v: Option<&str>, tier_on: bool) -> Result<SegMode, String> {
    let mode = match v {
        None | Some("0") => return Ok(SegMode::Off),
        Some("fuse") => SegMode::Fuse,
        Some("split") => SegMode::Split,
        Some(other) => {
            return Err(format!(
                "METRALE_GLM_W4A16_SEG={other:?}: expected 0, fuse or split"
            ));
        }
    };
    if !tier_on {
        return Err(format!(
            "METRALE_GLM_W4A16_SEG={} needs --dense-quantization w4a16",
            v.unwrap_or_default()
        ));
    }
    Ok(mode)
}

/// 2026-10-09: The SM count the split rule reads, recorded by [`prepare`].
static SMS: OnceLock<u32> = OnceLock::new();

/// 2026-10-09: At load, under a mode other than `off`: resolve every entry point (a missing one
/// is refused here, not at the first launch) and record the SM count. Never inside a capture.
pub(crate) fn prepare(gpu: &dyn GpuBackend) -> Result<()> {
    if seg_mode()? == SegMode::Off {
        return Ok(());
    }
    for entry in w4a16_tc_rows_seg_entries() {
        gpu.op_cache()
            .kernel(gpu, W4A16_TC_ROWS_SEG_MODULE, entry)
            .with_context(|| {
                format!("METRALE_GLM_W4A16_SEG: {W4A16_TC_ROWS_SEG_MODULE}::{entry}")
            })?;
    }
    if SMS.get().is_none() {
        let _ = SMS.set(gpu.sm_count()?);
    }
    Ok(())
}

/// 2026-10-09: The mode and the SM count its split rule reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegPlan {
    pub mode: SegMode,
    pub sms: u32,
}

impl SegPlan {
    /// 2026-10-09: The split for a launch over segments `ns` with K `k`.
    pub(crate) fn split(&self, ns: &[u32], k: u32) -> u32 {
        match self.mode {
            SegMode::Off | SegMode::Fuse => 1,
            SegMode::Split => w4a16_tc_rows_split_for(w4a16_tc_rows_seg_tiles(ns), k, self.sms),
        }
    }
}

/// 2026-10-09: The published plan: [`seg_mode`] and, unless it is `off`, the SM count
/// [`prepare`] recorded (an error when it did not run).
pub(crate) fn plan() -> Result<SegPlan> {
    let mode = seg_mode()?;
    let sms = match mode {
        SegMode::Off => 0,
        _ => *SMS
            .get()
            .context("METRALE_GLM_W4A16_SEG: the SM count was not recorded at load")?,
    };
    Ok(SegPlan { mode, sms })
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
