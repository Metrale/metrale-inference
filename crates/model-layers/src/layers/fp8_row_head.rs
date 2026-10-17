// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The row dispatch of a per-row-scaled FP8 E4M3 projection with a wide output,
//! the FP8 LM head (`--lm-head-dtype fp8`), into a column window of a `[rows, ldc]` BF16
//! output. With the window `0..ldc` it is the whole head. A vocab-parallel head passes its
//! shard's window: the weight rows and their scales are offset by the window's start, and
//! each output row lands at that start with a full-vocab pitch.
//!
//! Routes, by rows left to project:
//! - declared W8A8 (`w8a8` given): `ops::w8a8_proj` in `ops::W8A8_MAX_ROWS`-row calls;
//! - otherwise W8A16: `dense_gemv_fp8w` for 1 row, `dense_gemv_fp8w_batch2` for 2, and
//!   `fp8_gemv_rowscale_batch{8,16}_rt2` for 3..=8 and in 16-row chunks above. A kernel the
//!   target lacks (0 handle) drops to the next route, and last to one GEMV per row.
//!
//! Owner: model-layers.
//! Invariants:
//! - The route depends only on the rows left and the kernel handles, never on the window, so
//!   a window gets the kernel and K order the whole head uses for those rows. Per output, no
//!   kernel's arithmetic depends on the output index, its block, or `ldc`: the window's
//!   logits are the whole head's bits there (`tests/fp8_row_head_vocab_shard_cuda.rs`).
//! - Columns outside the window are never written.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::W8a8Ctx;
use super::ops::{self, W8a8Weight};
use crate::weight_map::{Fp8DenseWeight, Fp8Weight, WeightQuantFormat};

/// 2026-10-09: The W8A16 kernels of the FP8 row head; a 0 handle is a kernel the target
/// lacks. `gemv` is required.
#[derive(Clone, Copy, Debug)]
pub struct Fp8RowHeadKernels {
    /// 2026-10-09: `dense_gemv_fp8w` (module `gemv_fp8w`).
    pub gemv: KernelHandle,
    /// 2026-10-09: `dense_gemv_fp8w_batch2` (module `dense_gemv_fp8w_batch2`).
    pub batch2: KernelHandle,
    /// 2026-10-09: `fp8_gemv_rowscale_batch8_rt2` (module `fp8_gemv_rt`).
    pub rt8: KernelHandle,
    /// 2026-10-09: `fp8_gemv_rowscale_batch16_rt2` (module `fp8_gemv_rt`).
    pub rt16: KernelHandle,
}

/// 2026-10-09: Output columns `begin..begin + len` of a `[rows, ldc]` BF16 output, and so
/// weight rows `begin..begin + len` of the head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadWindow {
    pub begin: usize,
    pub len: usize,
    pub ldc: usize,
}

impl HeadWindow {
    /// 2026-10-09: The whole `n`-row head into a dense `[rows, n]` output.
    pub fn whole(n: usize) -> Self {
        Self {
            begin: 0,
            len: n,
            ldc: n,
        }
    }
}

/// 2026-10-09: `out[r, begin..begin + len]` for `r < rows`, from `x` (`[rows, k]` BF16) and
/// the head's rows `begin..begin + len` (`fp8`: `[*, k]` E4M3 with `[*]` f32 row scales).
/// `w8a8`, when given, runs the declared W8A8 route instead of W8A16.
#[allow(clippy::too_many_arguments)]
pub fn fp8_row_head(
    gpu: &dyn GpuBackend,
    kernels: &Fp8RowHeadKernels,
    w8a8: Option<&W8a8Ctx>,
    fp8: &Fp8DenseWeight,
    x: DevicePtr,
    k: usize,
    rows: usize,
    out: DevicePtr,
    win: HeadWindow,
    stream: u64,
) -> Result<()> {
    ensure!(
        win.len > 0 && win.begin + win.len <= win.ldc,
        "fp8_row_head: window {win:?} does not fit the output pitch"
    );
    let shard = Fp8DenseWeight {
        weight: fp8.weight.offset(win.begin * k),
        row_scale: fp8.row_scale.offset(win.begin * 4),
    };
    let out = out.offset(win.begin * 2);
    let (n, ku, ldc) = (win.len as u32, k as u32, win.ldc as u32);
    if let Some(ctx) = w8a8 {
        let w = W8a8Weight::new(&[Fp8Weight {
            weight: shard.weight,
            row_scale: shard.row_scale,
            n,
            k: ku,
            scale_format: WeightQuantFormat::Fp8PerRow,
        }])?;
        // 2026-09-28: `W8A8_MAX_ROWS`-row calls (the scratch's capacity); the quantization is
        // per row, so chunking does not change any row's bits.
        let mut done = 0;
        while done < rows {
            let m = (rows - done).min(ops::W8A8_MAX_ROWS);
            let (xr, o) = (x.offset(done * k * 2), out.offset(done * win.ldc * 2));
            ensure!(
                ctx.proj(gpu, &w, xr, ku, m, o, ldc, stream)?,
                "W8A8 FP8 head: {m} rows not servable"
            );
            done += m;
        }
        return Ok(());
    }
    let mut done = 0;
    while done < rows {
        let (xr, o, left) = (
            x.offset(done * k * 2),
            out.offset(done * win.ldc * 2),
            rows - done,
        );
        let step = match left {
            2 if kernels.batch2.0 != 0 => {
                ops::dense_gemv_fp8w_batch2(
                    gpu,
                    kernels.batch2,
                    xr,
                    &shard,
                    o,
                    n,
                    ku,
                    ldc,
                    stream,
                )?;
                2
            }
            3..=8 if kernels.rt8.0 != 0 => {
                let m = left as u32;
                ops::fp8_gemv_rowscale_batch8_rt2_ldc(
                    gpu,
                    kernels.rt8,
                    xr,
                    &shard,
                    o,
                    m,
                    n,
                    ku,
                    ldc,
                    stream,
                )?;
                left
            }
            _ if left > 2 && kernels.rt16.0 != 0 => {
                let m = left.min(16);
                ops::fp8_gemv_rowscale_batch16_rt2_ldc(
                    gpu,
                    kernels.rt16,
                    xr,
                    &shard,
                    o,
                    m as u32,
                    n,
                    ku,
                    ldc,
                    stream,
                )?;
                m
            }
            // 2026-10-09: One row, or a row count no batched kernel serves: one GEMV per row.
            // A single output row needs no pitch.
            _ => {
                ops::dense_gemv_fp8w(gpu, kernels.gemv, xr, &shard, o, n, ku, stream)?;
                1
            }
        };
        done += step;
    }
    Ok(())
}
