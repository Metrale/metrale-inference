// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The W8A8 per-row projection at prefill widths (`w8a8_gemm.cu`): one tensor-core
//! GEMM over the activation [`super::w8a8_act_quant`] left in the scratch, reading the weight once
//! per token tile instead of once per 128 rows. Every output value has the bits the GEMV entries
//! ([`super::w8a8_gemv`]) write for the same row: the kernel walks K in the engine's class order
//! (chunk c in class c mod 8, classes summed in order), so a row's output depends on neither M nor
//! the tile nor which family served it.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - Per-row (rowscale) weights only; a 128x128 block-scaled weight is refused.
//! - [`w8a8_gemm`] launches nothing unless the tile's entry resolved, the weight's K is a multiple
//!   of 128 and the scratch holds `rows` quantized rows of that K.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::layers::try_kernel;

use super::{W8A8_K_UNIT, W8a8Kernels, W8a8Scale, W8a8Scratch, W8a8Weight, w8a8_act_quant};

const GEMM_MODULE: &str = "w8a8_gemm";
const GEMM_THREADS: u32 = 256;

/// 2026-10-14: One compiled tile of `w8a8_gemm.cu`: weight rows x tokens per CTA and the cp.async
/// ring depth. Same arithmetic in each; the choice changes speed only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum W8a8GemmTile {
    W128T64S3,
    W128T64S2,
    W64T128S3,
    W128T128S2,
}

impl W8a8GemmTile {
    pub const ALL: [Self; 4] = [
        Self::W128T64S3,
        Self::W128T64S2,
        Self::W64T128S3,
        Self::W128T128S2,
    ];

    /// 2026-10-14: `(weight rows, tokens, stages)` of the tile.
    pub fn dims(self) -> (u32, u32, u32) {
        match self {
            Self::W128T64S3 => (128, 64, 3),
            Self::W128T64S2 => (128, 64, 2),
            Self::W64T128S3 => (64, 128, 3),
            Self::W128T128S2 => (128, 128, 2),
        }
    }

    pub fn entry(self) -> &'static str {
        match self {
            Self::W128T64S3 => "w8a8_gemm_rowscale_128x64_s3",
            Self::W128T64S2 => "w8a8_gemm_rowscale_128x64_s2",
            Self::W64T128S3 => "w8a8_gemm_rowscale_64x128_s3",
            Self::W128T128S2 => "w8a8_gemm_rowscale_128x128_s2",
        }
    }

    /// 2026-10-14: Dynamic shared memory of one CTA: a 128-byte K chunk of every staged row per
    /// stage.
    pub fn shared_bytes(self) -> u32 {
        let (bn, bm, stages) = self.dims();
        stages * (bn + bm) * W8A8_K_UNIT
    }

    fn index(self) -> usize {
        match self {
            Self::W128T64S3 => 0,
            Self::W128T64S2 => 1,
            Self::W64T128S3 => 2,
            Self::W128T128S2 => 3,
        }
    }
}

/// 2026-10-14: The GEMM entries, indexed like [`W8a8GemmTile::ALL`]. A handle is 0 when the module
/// is not compiled.
#[derive(Clone, Copy, Debug)]
pub struct W8a8GemmKernels {
    entries: [KernelHandle; 4],
}

impl W8a8GemmKernels {
    pub fn load(gpu: &dyn GpuBackend) -> Self {
        Self {
            entries: W8a8GemmTile::ALL.map(|t| try_kernel(gpu, GEMM_MODULE, t.entry())),
        }
    }

    pub fn none() -> Self {
        Self {
            entries: [KernelHandle(0); 4],
        }
    }

    pub fn resolved(&self, tile: W8a8GemmTile) -> bool {
        self.entries[tile.index()].0 != 0
    }
}

/// 2026-10-14: The GEMM over the activation [`w8a8_act_quant`] left in `scratch`: `out[m, 0..n]`
/// (row pitch `ldc` BF16 elements) for `m < rows`, in one launch.
#[allow(clippy::too_many_arguments)]
pub fn w8a8_gemm(
    gpu: &dyn GpuBackend,
    kernels: &W8a8GemmKernels,
    tile: W8a8GemmTile,
    w: &W8a8Weight,
    scratch: &W8a8Scratch,
    rows: usize,
    out: DevicePtr,
    ldc: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        w.scale == W8a8Scale::PerRow,
        "w8a8_gemm: {:?} weights are not served (per-row only)",
        w.scale
    );
    ensure!(
        kernels.resolved(tile),
        "w8a8_gemm: {} not compiled",
        tile.entry()
    );
    ensure!(
        rows >= 1 && u32::try_from(rows).is_ok() && scratch.fits(rows, w.k, w.scale),
        "w8a8_gemm: {rows} rows of K={} do not fit the scratch",
        w.k
    );
    ensure!(ldc >= w.n, "w8a8_gemm: ldc={ldc} < N={}", w.n);
    let (bn, bm, _) = tile.dims();
    let (n1, n2) = w.boundaries();
    let s = &w.segs;
    KernelLaunch::new(gpu, kernels.entries[tile.index()])
        .grid([w.n.div_ceil(bn), (rows as u32).div_ceil(bm), 1])
        .block([GEMM_THREADS, 1, 1])
        .shared_mem(tile.shared_bytes())
        .arg_ptr(scratch.q)
        .arg_ptr(scratch.scale)
        .arg_ptr(s[0].weight)
        .arg_ptr(s[0].row_scale)
        .arg_ptr(s[1].weight)
        .arg_ptr(s[1].row_scale)
        .arg_ptr(s[2].weight)
        .arg_ptr(s[2].row_scale)
        .arg_ptr(out)
        .arg_u32(rows as u32)
        .arg_u32(w.n)
        .arg_u32(w.k)
        .arg_u32(w.k)
        .arg_u32(ldc)
        .arg_u32(n1)
        .arg_u32(n2)
        .launch(stream)
}

/// 2026-10-14: One per-row W8A8 projection at any width: quantize `x` (`[rows, ldx]` BF16) into
/// `scratch`, then [`w8a8_gemm`].
#[allow(clippy::too_many_arguments)]
pub fn w8a8_proj_wide(
    gpu: &dyn GpuBackend,
    kernels: &W8a8Kernels,
    gemm: &W8a8GemmKernels,
    tile: W8a8GemmTile,
    w: &W8a8Weight,
    x: DevicePtr,
    ldx: u32,
    rows: usize,
    out: DevicePtr,
    ldc: u32,
    scratch: &W8a8Scratch,
    stream: u64,
) -> Result<()> {
    w8a8_act_quant(gpu, kernels, w.scale, x, ldx, rows, w.k, scratch, stream)?;
    w8a8_gemm(gpu, gemm, tile, w, scratch, rows, out, ldc, stream)
}
