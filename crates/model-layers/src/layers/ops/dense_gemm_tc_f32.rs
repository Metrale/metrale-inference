// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: BF16 x BF16 -> FP32 dense projection on the in-tree tensor-core entries of
//! `kernels/gb10/common/dense_gemm_tc_f32out.cu`, the replacement for
//! [`super::cublas_bf16_proj_dense_f32_out`] (cuBLASLt).
//!
//! Owner: model-layers ops.
//! Invariants:
//! - `out[M, N] (f32) = act[M, K] (bf16) @ weight[N, K]^T (bf16)`, FP32 accumulation, no output
//!   rounding: act contiguous (row stride K), weight row-major `[N, K]` (row stride K), out
//!   contiguous (row stride N). The arguments and layouts of the cuBLASLt call it replaces.
//! - [`TcF32Plan::new`] is the one place a path is chosen: the small-M GEMV tier for
//!   `M <= TC32_MAX_M` with `K % K_STEP == 0`, the 16x64 tile otherwise. It refuses, with an
//!   error, a shape the kernels cannot index (a zero dimension, a tile grid past the launch
//!   limits, a tile index past `u32`).
//! - [`launch`] runs exactly what the plan holds; it reads no lever, environment variable or
//!   threshold. It refuses a GEMV plan whose act or weight base is not 16-byte aligned (the GEMV
//!   loads rows as 16-byte vectors).
//! - [`TcF32Kernels::resolve`] fails unless all four entries resolve: they are one module, so a
//!   partial set is a build defect, never a mode.
//! - The two paths sum in different orders, so their outputs for one input differ in the last
//!   bits; a caller that needs batch invariance across the `TC32_MAX_M` edge must not use this.

use anyhow::{Context, Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::dense_gemv_tc::{
    BLOCK as GEMV_BLOCK, DtcKind, K_STEP, ROWS_PER_CTA, TC8_MAX_M, TC16_MAX_M, TC32_MAX_M,
};

/// 2026-10-17: The kernel module (the source stem; KERNEL.toml renames no such stem).
pub const MODULE: &str = "dense_gemm_tc_f32out";
/// 2026-10-17: `TC_TM`, `TC_TN` and `TC_BLOCK` of `dense_gemm_tc_body.cuh`.
pub const TILE_M: u32 = 16;
pub const TILE_N: u32 = 64;
pub const TILE_BLOCK: u32 = 128;
/// 2026-10-17: CUDA's grid-y limit; the tile grid puts `ceil(M / TILE_M)` there.
const MAX_GRID_Y: u32 = 65_535;
/// 2026-10-17: The GEMV entries load act and weight rows as 16-byte vectors.
const GEMV_ALIGN: u64 = 16;

/// 2026-10-17: The four entries, resolved once per backend by the caller.
#[derive(Clone, Copy, Debug)]
pub struct TcF32Kernels {
    tile: KernelHandle,
    tc8: KernelHandle,
    tc16: KernelHandle,
    tc32: KernelHandle,
}

impl TcF32Kernels {
    /// 2026-10-17: Look up every entry of [`MODULE`]; the first missing one is the error (the
    /// backend's message names it). The entry names are literals so the kernel_lookups test of
    /// metrale-kernels checks each against the `kernels/` tree.
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        const WHAT: &str = "FP32-output projection kernels";
        Ok(Self {
            tile: gpu.kernel(MODULE, "dense_gemm_tc_f32out").context(WHAT)?,
            tc8: gpu
                .kernel(MODULE, "dense_gemv_bf16_tc8_f32out")
                .context(WHAT)?,
            tc16: gpu
                .kernel(MODULE, "dense_gemv_bf16_tc16_f32out")
                .context(WHAT)?,
            tc32: gpu
                .kernel(MODULE, "dense_gemv_bf16_tc32_f32out")
                .context(WHAT)?,
        })
    }

    fn gemv(&self, kind: DtcKind) -> KernelHandle {
        match kind {
            DtcKind::M8 => self.tc8,
            DtcKind::M16 => self.tc16,
            DtcKind::M32 => self.tc32,
        }
    }
}

/// 2026-10-17: Which entry runs and on what grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TcF32Path {
    /// 2026-10-17: GEMV tier `kind`: grid `(grid_x, 1, 1)`, block `GEMV_BLOCK`, out_stride N.
    Gemv { kind: DtcKind, grid_x: u32 },
    /// 2026-10-17: The 16x64 tile: grid `(grid_x, grid_y, 1)`, block `TILE_BLOCK`.
    Tile { grid_x: u32, grid_y: u32 },
}

/// 2026-10-17: A validated launch plan for one `(M, N, K)`; only [`TcF32Plan::new`] builds one,
/// so a plan and its shape cannot disagree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TcF32Plan {
    m: u32,
    n: u32,
    k: u32,
    path: TcF32Path,
}

impl TcF32Plan {
    pub fn new(m: u32, n: u32, k: u32) -> Result<Self> {
        ensure!(
            m > 0 && n > 0 && k > 0,
            "FP32-output projection: empty shape M={m} N={n} K={k}"
        );
        let tier = [
            (TC8_MAX_M, DtcKind::M8),
            (TC16_MAX_M, DtcKind::M16),
            (TC32_MAX_M, DtcKind::M32),
        ]
        .into_iter()
        .find(|&(cap, _)| m <= cap)
        .map(|(_, kind)| kind);
        let path = match tier {
            Some(kind) if k.is_multiple_of(K_STEP) => TcF32Path::Gemv {
                kind,
                grid_x: div_ceil(n, ROWS_PER_CTA),
            },
            _ => Self::tile(m, n, k)?,
        };
        Ok(Self { m, n, k, path })
    }

    /// 2026-10-17: The tile indexes act and out with 32-bit `row * K + col` and
    /// `row * N + col`, so `M * K` and `M * N` must fit in `u32`.
    fn tile(m: u32, n: u32, k: u32) -> Result<TcF32Path> {
        let grid_y = div_ceil(m, TILE_M);
        let fits = |x: u32| u64::from(m) * u64::from(x) <= u64::from(u32::MAX);
        if grid_y > MAX_GRID_Y || !fits(k) || !fits(n) {
            bail!(
                "FP32-output projection: tile cannot index M={m} N={n} K={k} \
                 (grid-y {grid_y} of {MAX_GRID_Y}, M*K and M*N must fit in u32)"
            );
        }
        Ok(TcF32Path::Tile {
            grid_x: div_ceil(n, TILE_N),
            grid_y,
        })
    }

    pub fn path(&self) -> TcF32Path {
        self.path
    }

    pub fn shape(&self) -> (u32, u32, u32) {
        (self.m, self.n, self.k)
    }
}

/// 2026-10-17: Launch `plan` on `stream`.
pub fn launch(
    gpu: &dyn GpuBackend,
    kernels: &TcF32Kernels,
    plan: &TcF32Plan,
    act: DevicePtr,
    weight: DevicePtr,
    out: DevicePtr,
    stream: u64,
) -> Result<()> {
    let (m, n, k) = plan.shape();
    match plan.path {
        TcF32Path::Gemv { kind, grid_x } => {
            ensure!(
                act.0.is_multiple_of(GEMV_ALIGN) && weight.0.is_multiple_of(GEMV_ALIGN),
                "FP32-output projection: GEMV needs {GEMV_ALIGN}-byte act and weight bases, \
                 got act {:#x} weight {:#x}",
                act.0,
                weight.0
            );
            KernelLaunch::new(gpu, kernels.gemv(kind))
                .grid([grid_x, 1, 1])
                .block([GEMV_BLOCK, 1, 1])
                .arg_ptr(act)
                .arg_ptr(weight)
                .arg_ptr(out)
                .arg_u32(m)
                .arg_u32(n)
                .arg_u32(k)
                .arg_u32(n)
                .launch(stream)
        }
        TcF32Path::Tile { grid_x, grid_y } => KernelLaunch::new(gpu, kernels.tile)
            .grid([grid_x, grid_y, 1])
            .block([TILE_BLOCK, 1, 1])
            .arg_ptr(act)
            .arg_ptr(weight)
            .arg_ptr(out)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream),
    }
}

/// 2026-10-17: Plan and launch in one call, with the arguments of
/// [`super::cublas_bf16_proj_dense_f32_out`] plus the resolved kernels.
#[allow(clippy::too_many_arguments)]
pub fn dense_proj_f32_out(
    gpu: &dyn GpuBackend,
    kernels: &TcF32Kernels,
    act: DevicePtr,
    weight: DevicePtr,
    out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let plan = TcF32Plan::new(m, n, k)?;
    launch(gpu, kernels, &plan, act, weight, out, stream)
}

#[cfg(test)]
#[path = "dense_gemm_tc_f32_tests.rs"]
mod dense_gemm_tc_f32_tests;

#[cfg(all(test, feature = "cuda"))]
#[path = "dense_gemm_tc_f32_gpu_tests.rs"]
mod dense_gemm_tc_f32_gpu_tests;
