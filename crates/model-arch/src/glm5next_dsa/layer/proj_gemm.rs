// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `Glm5NextDsaLayer::gemm`, the BF16 projection dispatch every layer method
//! launches its dense projections through.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - M above `DENSE_GEMV_BATCHM_MAX_M` with `GlmLevers::cublas_wide_proj` on writes
//!   BF16, whatever `k` is.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::glm5next_layer::wide_gemv::{Batchm, glm_mm};

impl super::Glm5NextDsaLayer {
    /// 2026-09-25: `C[M, N] = A[M, K] @ B[N, K]^T` with BF16 inputs.
    ///
    /// M above `DENSE_GEMV_BATCHM_MAX_M` goes to cuBLASLt when `levers.cublas_wide_proj` is
    /// on (the default), and that arm writes BF16, so FP32-out callers pass M = 1 (all in this
    /// file do). Otherwise `ops::dense_mm_bf16` picks the GEMV at M = 1, `batchm` at
    /// 2..=`DENSE_GEMV_BATCHM_MAX_M` when it is not `KernelHandle(0)`, and the tile GEMM `k`
    /// otherwise. 2026-10-09: with a wide handle in `batchm`, 9..=16 rows take the
    /// register-resident batched GEMV (`glm5next_layer::wide_gemv`).
    pub(super) fn gemm(
        &self,
        gpu: &dyn GpuBackend,
        k: KernelHandle,
        gemv: KernelHandle,
        batchm: impl Into<Batchm>,
        a: DevicePtr,
        b: DevicePtr,
        c: DevicePtr,
        m: usize,
        n: usize,
        kk: usize,
        stream: u64,
    ) -> Result<()> {
        // 2026-10-09: The dispatch lives in `glm_mm`, shared with the MLP and KDA blocks.
        glm_mm(
            gpu,
            self.levers.cublas_wide_proj,
            k,
            gemv,
            batchm.into(),
            a,
            b,
            c,
            m,
            n,
            kk,
            stream,
        )
    }
}
