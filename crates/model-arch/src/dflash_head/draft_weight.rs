// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: A quantized copy of one drafter GEMM weight (`--draft-weight-quantization`): its
//! build from the BF16 weight and the row GEMM every copy dispatches through.
//!
//! FP8 copies are E4M3 with one f32 scale per row (`quantize_bf16_to_fp8`). NVFP4 copies are
//! packed E2M1 with one E4M3 scale per 16 weights and one f32 scale per tensor
//! (`quantize_bf16_to_nvfp4`), run as W4A16 on `w4a16_tc_rows`.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - An NVFP4 copy's per-tensor scale is `max |w| / (6 * 448)` over the whole unsharded tensors
//!   it is cut from (or over the weight itself), the scale `quantize_to_nvfp4` sets; its 16-wide
//!   groups and their scales depend only on their own row. A tensor-parallel row slice, or a
//!   column slice starting on a 16 boundary, is therefore byte for byte that part of the
//!   one-rank copy.
//! - The FP8 copy of a column slice quantizes the whole rows and keeps this rank's columns with
//!   the whole rows' scales, so it too is the one-rank copy's part.
//! - Building synchronises `stream`; it runs at load, never in a forward.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::{
    DenseWeight, Fp8DenseWeight, QuantizedWeight, nvfp4_scale2,
};

use super::BlockDiffusionDraftHead;
use super::tp::{ColSlice, FullWeight};

/// 2026-10-10: One drafter GEMM weight's quantized copy.
#[derive(Debug, Clone, Copy)]
pub enum DraftWeight {
    Fp8(Fp8DenseWeight),
    Nvfp4(QuantizedWeight),
}

/// 2026-10-10: `(first row, rows)` of each launch covering `g` rows at most `max` at a time.
pub fn row_chunks(g: u32, max: u32) -> impl Iterator<Item = (u32, u32)> {
    (0..g.div_ceil(max)).map(move |i| (i * max, (g - i * max).min(max)))
}

/// 2026-10-10: Builds [`DraftWeight`] copies in one format with the drafter's quantizer
/// kernels (`quantize_bf16_to_fp8` of `gemv_fp8w`; `nvfp4_global_absmax` and
/// `quantize_bf16_to_nvfp4` of `quantize_bf16_to_nvfp4`).
pub struct DraftQuantizer<'a> {
    pub gpu: &'a dyn GpuBackend,
    /// 2026-10-10: NVFP4 copies when true, else FP8.
    pub nvfp4: bool,
    pub fp8_k: KernelHandle,
    pub absmax_k: KernelHandle,
    pub nvfp4_k: KernelHandle,
    pub stream: u64,
}

impl DraftQuantizer<'_> {
    /// 2026-10-10: The copy of the row-major `[n, k]` BF16 `local`. `whole`: the unsharded
    /// tensors `local` was cut from, whose largest magnitude sets the NVFP4 scale (empty: `local`
    /// itself). `cols`: the whole rows of a column slice, which the FP8 copy quantizes.
    pub fn copy(
        &self,
        local: &DenseWeight,
        n: usize,
        k: usize,
        whole: &[FullWeight],
        cols: Option<ColSlice>,
    ) -> Result<DraftWeight> {
        if !self.nvfp4 {
            return Ok(DraftWeight::Fp8(match cols {
                Some(c) => self.fp8_cols(c, n, k)?,
                None => local.quantize_to_fp8(self.gpu, self.fp8_k, n, k, self.stream)?,
            }));
        }
        ensure!(
            k.is_multiple_of(256),
            "--draft-weight-quantization nvfp4: a [{n}, {k}] drafter weight; the W4A16 row \
             kernel needs K a multiple of 256"
        );
        if let Some(c) = cols {
            ensure!(
                c.start.is_multiple_of(16),
                "DFlash TP NVFP4: column slice at {} splits a 16-wide scale group",
                c.start
            );
        }
        let own = [FullWeight {
            weight: *local,
            elems: n * k,
        }];
        let scale2 = nvfp4_scale2(self.absmax(if whole.is_empty() { &own } else { whole })?);
        let packed = self.gpu.alloc(n * k / 2)?;
        let scales = self.gpu.alloc(n * k / 16)?;
        KernelLaunch::new(self.gpu, self.nvfp4_k)
            .grid([n as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(local.weight)
            .arg_ptr(packed)
            .arg_ptr(scales)
            .arg_f32(scale2)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(self.stream)?;
        self.gpu.synchronize(self.stream)?;
        Ok(DraftWeight::Nvfp4(QuantizedWeight {
            weight: packed,
            weight_scale: scales,
            weight_scale_2: scale2,
            ..QuantizedWeight::null()
        }))
    }

    /// 2026-10-10: The largest `|w|` over every tensor of `of`, folded into one maximum
    /// (`nvfp4_global_absmax` takes the max with what the buffer holds).
    fn absmax(&self, of: &[FullWeight]) -> Result<f32> {
        let buf = self.gpu.alloc(4)?;
        self.gpu.memset(buf, 0, 4)?;
        for t in of {
            ensure!(
                t.elems <= u32::MAX as usize,
                "NVFP4 absmax: {} elements",
                t.elems
            );
            KernelLaunch::new(self.gpu, self.absmax_k)
                .grid([(t.elems / 256).clamp(1, 1024) as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(t.weight.weight)
                .arg_ptr(buf)
                .arg_u32(t.elems as u32)
                .launch(self.stream)?;
        }
        self.gpu.synchronize(self.stream)?;
        let mut b = [0u8; 4];
        self.gpu.copy_d2h(buf, &mut b)?;
        self.gpu.free(buf)?;
        Ok(f32::from_le_bytes(b))
    }

    /// 2026-10-10: The FP8 copy of an `[n, k]` column slice: quantize the whole rows
    /// (`[n, c.full_k]`), keep `k` columns from `c.start` and the whole rows' scales.
    fn fp8_cols(&self, c: ColSlice, n: usize, k: usize) -> Result<Fp8DenseWeight> {
        ensure!(
            c.start + k <= c.full_k,
            "DFlash TP FP8: columns {}..{} past the whole row of {}",
            c.start,
            c.start + k,
            c.full_k
        );
        let full = c
            .full
            .quantize_to_fp8(self.gpu, self.fp8_k, n, c.full_k, self.stream)?;
        let weight = self.gpu.alloc(n * k)?;
        self.gpu.copy_d2d_2d_async(
            full.weight.offset(c.start),
            c.full_k,
            weight,
            k,
            k,
            n,
            self.stream,
        )?;
        self.gpu.synchronize(self.stream)?;
        self.gpu.free(full.weight)?;
        Ok(Fp8DenseWeight {
            weight,
            row_scale: full.row_scale,
        })
    }
}

impl BlockDiffusionDraftHead {
    /// 2026-10-10: `dst [g, n] = src [g, k] @ W^T` on a quantized copy: FP8 through
    /// [`Self::fp8_rows`], NVFP4 on `w4a16_tc_rows` in launches of at most
    /// `W4A16_TC_ROWS_MAX_M` rows (a row's bits do not depend on the launch's row count).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn quant_rows(
        &self,
        gpu: &dyn GpuBackend,
        src: DevicePtr,
        w: &DraftWeight,
        dst: DevicePtr,
        g: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        match w {
            DraftWeight::Fp8(f) => self.fp8_rows(gpu, src, f, dst, g, n, k, stream),
            DraftWeight::Nvfp4(q) => nvfp4_rows(gpu, src, q, dst, g, n, k, stream),
        }
    }
}

/// 2026-10-10: `dst [g, n] = src [g, k] @ W^T` on an NVFP4 copy: `w4a16_tc_rows` in launches of
/// at most `W4A16_TC_ROWS_MAX_M` rows, row-major `src` and `dst` (pitches `k` and `n`).
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_rows(
    gpu: &dyn GpuBackend,
    src: DevicePtr,
    q: &QuantizedWeight,
    dst: DevicePtr,
    g: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    for (r0, m) in row_chunks(g, ops::W4A16_TC_ROWS_MAX_M) {
        ops::w4a16_tc_rows(
            gpu,
            src.offset(r0 as usize * k as usize * 2),
            q,
            dst.offset(r0 as usize * n as usize * 2),
            m,
            n,
            k,
            k,
            n,
            stream,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::row_chunks;

    #[test]
    fn row_chunks_cover_every_row_once_in_order() {
        for (g, max) in [(1, 64), (8, 64), (64, 64), (65, 64), (200, 64), (7, 3)] {
            let chunks: Vec<(u32, u32)> = row_chunks(g, max).collect();
            let mut next = 0;
            for &(r0, m) in &chunks {
                assert_eq!(r0, next, "g={g} max={max}: {chunks:?}");
                assert!((1..=max).contains(&m), "g={g} max={max}: {chunks:?}");
                next += m;
            }
            assert_eq!(next, g, "g={g} max={max}: {chunks:?}");
        }
        assert_eq!(row_chunks(0, 64).count(), 0);
        assert_eq!(
            row_chunks(130, 64).collect::<Vec<_>>(),
            [(0, 64), (64, 64), (128, 2)]
        );
    }
}
