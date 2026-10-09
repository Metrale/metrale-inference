// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The GLM-5.3 router logits of a routed MoE site: `[rows, num_experts]` FP32 from
//! the BF16 router weight, before `glm5next_router_topk`.
//!
//! Owner: model-arch (GLM-5.3 MLP).
//! Invariants:
//! - Above `DENSE_GEMV_BATCHM_MAX_M` rows (with `cublas_wide_proj`) one cuBLASLt FP32-out GEMM;
//!   otherwise each row's logits are the bits of the M = 1 `dense_gemv_bf16_fp32out` on that
//!   row, whether the rows run one GEMV each or (`batched`) one batched FP32-out GEMV for the
//!   group (`dense_gemv_bf16_batchm.cu`: each row of the batched entries carries the M = 1
//!   GEMV's arithmetic in the same order).

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::super::{Glm5NextMlpConfig, Glm5NextMlpKernels};
use super::gemm;

/// 2026-10-09: The router logits of `rows` rows of `x` (`[rows, hidden]` BF16) into `logits`
/// (`[rows, num_experts]` FP32). `batched` (`METRALE_GLM_ROUTER_ROWS=1`) sends 2..=16 rows to
/// the batched FP32-out GEMV in one launch, which reads the router weight once per group
/// instead of once per row; without the batched kernels, or with one row, it is one M = 1 GEMV
/// per row as before.
#[allow(clippy::too_many_arguments)]
pub(super) fn router_logits(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    router: DevicePtr,
    x: DevicePtr,
    logits: DevicePtr,
    rows: usize,
    batched: bool,
    stream: u64,
) -> Result<()> {
    let (n, kk) = (cfg.num_experts, cfg.hidden);
    let max_m = metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M as usize;
    if rows > max_m && crate::glm5next_layer::cublas_wide_proj() {
        return metrale_model_layers::layers::ops::cublas_bf16_proj_dense_f32_out(
            x,
            router,
            logits,
            rows as u32,
            n as u32,
            kk as u32,
            stream,
        );
    }
    // 2026-10-09: `glm_mm` runs the M = 1 GEMV at one row, the narrow batched entry at
    // 2..=8 and the wide one at 9..=16. Both batched handles are needed so no row count of
    // the group falls through to the tile GEMM, whose rows are not the GEMV's bits.
    let batch = batched
        && rows >= 2
        && rows <= max_m
        && k.gemv_f32.0 != 0
        && k.gemv_batchm_f32.0 != 0
        && k.gemv_batchm_wide_f32.0 != 0;
    if batch {
        return gemm(
            gpu,
            k.gemm_f32,
            k.gemv_f32,
            k.batchm_f32(),
            x,
            router,
            logits,
            rows,
            n,
            kk,
            stream,
        );
    }
    for r in 0..rows {
        gemm(
            gpu,
            k.gemm_f32,
            k.gemv_f32,
            KernelHandle(0),
            x.offset(r * kk * 2),
            router,
            logits.offset(r * n * 4),
            1,
            n,
            kk,
            stream,
        )?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "router_tests.rs"]
mod tests;
