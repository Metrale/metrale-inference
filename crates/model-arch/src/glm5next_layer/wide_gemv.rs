// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The BF16 projection dispatch every GLM-5.3 block shares (`glm_mm`), with the
//! register-resident batched GEMV for 9..=16 rows (`dense_gemv_bf16_batchm_wide`).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Only `glm_mm` launches the wide entries, only at `BATCHM_WIDE_MIN..=DENSE_GEMV_BATCHM_MAX_M`
//!   rows, and only when the caller passes a wide handle; every other M takes the dispatch the
//!   DSA and MLP `gemm` helpers took before (cuBLASLt above 16 rows, else
//!   `ops::dense_mm_bf16`), which other models share and which is unchanged.
//! - A wide entry's row has the bits of the runtime-M batched GEMV's row
//!   (`kernels/gb10/common/dense_gemv_bf16_batchm.cu`), which has `dense_gemv_bf16`'s.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use metrale_model_layers::layers::ops;

/// 2026-10-09: `#define BATCHM_WIDE_MIN` in `dense_gemv_bf16_batchm.cu`: the narrowest M the
/// wide entries take.
pub(crate) const BATCHM_WIDE_MIN: usize = 9;

/// 2026-10-09: A batched-GEMV pair: `narrow` is the runtime-M entry `ops::dense_mm_bf16`
/// launches at 2..=16 rows; `wide` the register-resident entry for 9..=16 rows, or
/// `KernelHandle(0)` to keep `narrow` there.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Batchm {
    pub(crate) narrow: KernelHandle,
    pub(crate) wide: KernelHandle,
}

impl From<KernelHandle> for Batchm {
    fn from(narrow: KernelHandle) -> Self {
        Self {
            narrow,
            wide: KernelHandle(0),
        }
    }
}

/// 2026-10-09: `METRALE_GLM_BATCHM_WIDE=0` keeps the runtime-M batched GEMV at 9..=16 rows
/// (an A/B of the speed; the bits are the same). On otherwise. Read once.
fn wide_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("METRALE_GLM_BATCHM_WIDE").as_deref() != Ok("0"))
}

/// 2026-10-09: Whether `glm_mm` takes the wide entry for `m` rows with handle `wide`.
pub(crate) fn takes_wide(wide: KernelHandle, m: usize) -> bool {
    wide.0 != 0
        && (BATCHM_WIDE_MIN..=ops::DENSE_GEMV_BATCHM_MAX_M as usize).contains(&m)
        && wide_enabled()
}

/// 2026-10-09: `C[M, N] = A[M, K] @ B[N, K]^T` with BF16 inputs (the output element is the
/// kernels'): cuBLASLt above `DENSE_GEMV_BATCHM_MAX_M` rows when `cublas_wide_proj` is on (BF16
/// out), the wide batched GEMV at 9..=16 rows when `batchm.wide` is set, else
/// `ops::dense_mm_bf16` (GEMV at one row, `batchm.narrow` at 2..=16, the tile GEMM otherwise).
#[allow(clippy::too_many_arguments)]
pub(crate) fn glm_mm(
    gpu: &dyn GpuBackend,
    gemm: KernelHandle,
    gemv: KernelHandle,
    batchm: Batchm,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    kk: usize,
    stream: u64,
) -> Result<()> {
    if m > ops::DENSE_GEMV_BATCHM_MAX_M as usize && crate::glm5next_layer::cublas_wide_proj() {
        return ops::cublas_bf16_proj_dense(a, b, c, m as u32, n as u32, kk as u32, stream);
    }
    if takes_wide(batchm.wide, m) {
        return KernelLaunch::new(gpu, batchm.wide)
            .grid([div_ceil(n as u32, 4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(b)
            .arg_ptr(c)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(kk as u32)
            .arg_u32(n as u32)
            .launch(stream);
    }
    ops::dense_mm_bf16(
        gpu,
        &ops::DenseMmKernels {
            gemm,
            gemv,
            batchm: batchm.narrow,
        },
        a,
        b,
        c,
        m,
        n,
        kk,
        stream,
    )
}
