// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The launcher of `glm5next_mla_head_bmm` (module of the same name,
//! `kernels/gb10/glm-5.3-flash/nvfp4/glm5next_mla_head_bmm.cu`).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - [`head_bmm_grid`] refuses, before any launch, a shape the kernel is not compiled for, no
//!   rows, more rows than the scratch holds, a grid past CUDA's y/z bound, or a pointer that is
//!   not 16-byte aligned (the kernel's cp.async and stores are 16-byte).

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::lever::{HEAD_DIM, LATENT};

/// 2026-10-15: The kernel's module.
pub const MODULE: &str = "glm5next_mla_head_bmm";
/// 2026-10-15: Rows and columns of one CTA's output tile, and its thread count.
const TILE: usize = 128;
const THREADS: u32 = 256;
/// 2026-10-15: CUDA's bound on grid y and z.
const GRID_YZ_MAX: usize = 65_535;

/// 2026-10-15: Which half of `kv_b_proj` a launch applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadBmm {
    /// 2026-10-15: `_q`: K `HEAD_DIM`, N `LATENT` (q_b output into latent query).
    Q,
    /// 2026-10-15: `_o`: K `LATENT`, N `HEAD_DIM` (latent attend output into o_proj input).
    O,
}

impl HeadBmm {
    /// 2026-10-15: `(K, N)` per head.
    pub fn dims(self) -> (usize, usize) {
        match self {
            HeadBmm::Q => (HEAD_DIM, LATENT),
            HeadBmm::O => (LATENT, HEAD_DIM),
        }
    }
}

/// 2026-10-15: The two entries.
#[derive(Clone, Copy, Debug)]
pub struct HeadBmmKernels {
    pub q: KernelHandle,
    pub o: KernelHandle,
}

impl HeadBmmKernels {
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            q: gpu.kernel(MODULE, "glm5next_mla_head_bmm_q")?,
            o: gpu.kernel(MODULE, "glm5next_mla_head_bmm_o")?,
        })
    }
}

/// 2026-10-15: The grid of one launch over `m` rows and `heads` heads, or why it cannot run:
/// `m` in `1..=cap_rows`, `heads` in `1..=65535`, and `a`, `b`, `c` 16-byte aligned.
pub fn head_bmm_grid(
    which: HeadBmm,
    m: usize,
    heads: usize,
    cap_rows: usize,
    ptrs: [DevicePtr; 3],
) -> Result<[u32; 3]> {
    let (_, n) = which.dims();
    ensure!(
        (1..=cap_rows).contains(&m),
        "glm5next_mla_head_bmm {which:?}: {m} rows for a scratch of {cap_rows}"
    );
    ensure!(
        (1..=GRID_YZ_MAX).contains(&heads),
        "glm5next_mla_head_bmm {which:?}: {heads} heads"
    );
    let tiles = m.div_ceil(TILE);
    ensure!(
        tiles <= GRID_YZ_MAX,
        "glm5next_mla_head_bmm {which:?}: {m} rows need {tiles} row tiles (max {GRID_YZ_MAX})"
    );
    for (what, p) in ["A", "B", "C"].iter().zip(ptrs) {
        ensure!(
            p.0 != 0 && p.0.is_multiple_of(16),
            "glm5next_mla_head_bmm {which:?}: {what} {:#x} is null or not 16-byte aligned",
            p.0
        );
    }
    Ok([(n / TILE) as u32, tiles as u32, heads as u32])
}

/// 2026-10-15: `C[i, h*N + j] = Σ_k A[i, h*K + k] · B[h][j][k]` for `m` rows (see the kernel).
#[allow(clippy::too_many_arguments)]
pub fn launch_head_bmm(
    gpu: &dyn GpuBackend,
    k: &HeadBmmKernels,
    which: HeadBmm,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: usize,
    heads: usize,
    cap_rows: usize,
    stream: u64,
) -> Result<()> {
    let grid = head_bmm_grid(which, m, heads, cap_rows, [a, b, c])?;
    let handle = match which {
        HeadBmm::Q => k.q,
        HeadBmm::O => k.o,
    };
    KernelLaunch::new(gpu, handle)
        .grid(grid)
        .block([THREADS, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(m as u32)
        .arg_u32(heads as u32)
        .launch(stream)
}
