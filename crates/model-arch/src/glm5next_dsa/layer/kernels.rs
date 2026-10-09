// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: [`Glm5NextDsaLayerKernels`], moved out of `layer.rs` unchanged apart from the
//! FP32-out batched GEMV.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

/// 2026-09-25: The projection, norm and latent-write kernels of a DSA block. The selection
/// kernels are in `Glm5NextDsaKernels`, the decode kernel in `attend`.
#[derive(Clone, Copy)]
pub struct Glm5NextDsaLayerKernels {
    /// 2026-09-25: `dense_gemm_bf16`, `C = A @ B^T`, BF16 out.
    pub gemm: KernelHandle,
    /// 2026-09-25: The same with FP32 out, for the selector's `q_idx` and head weights.
    pub gemm_f32: KernelHandle,
    /// 2026-09-25: M = 1 kernels for `gemm` and `gemm_f32`. `gemv_f32` is `KernelHandle(0)`
    /// on a target without `dense_gemv_bf16_fp32out`, and `dense_mm_bf16` then uses the tile
    /// GEMM.
    pub gemv: KernelHandle,
    pub gemv_f32: KernelHandle,
    /// 2026-09-25: `dense_gemv_bf16_batchm`: 2..=`DENSE_GEMV_BATCHM_MAX_M` rows in one pass
    /// over the weight; `KernelHandle(0)` when the target lacks it.
    pub gemv_batchm: KernelHandle,
    /// 2026-10-09: `dense_gemv_bf16_batchm_fp32out`, the same at FP32 out: each row is
    /// `gemv_f32`'s result for that row. `KernelHandle(0)` when the target lacks it; the
    /// batched decode then computes the indexer query and head weights one row at a time.
    pub gemv_batchm_f32: KernelHandle,
    /// 2026-09-25: `rms_norm_vanilla`; see the module header.
    pub rms_norm: KernelHandle,
    /// 2026-09-25: `glm5next_mla_latent_write_fp8`: RMSNorm, FP8 quantisation and the paged
    /// slot write of the KV latent.
    pub latent_write: KernelHandle,
}

impl Glm5NextDsaLayerKernels {
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            // 2026-09-25: `kernels/gb10/common/KERNEL.toml` `[modules]` maps
            // `dense_gemm_bf16 = "gemm"` and `dense_gemv_bf16 = "gemv"`; an unlisted `.cu`
            // file's module is its file stem.
            gemm: gpu.kernel("gemm", "dense_gemm_bf16")?,
            gemm_f32: gpu.kernel("gemm", "dense_gemm_bf16_f32out")?,
            gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            gemv_f32: metrale_model_layers::layers::try_kernel(
                gpu,
                "gemv",
                "dense_gemv_bf16_fp32out",
            ),
            gemv_batchm: metrale_model_layers::layers::try_kernel(
                gpu,
                "dense_gemv_bf16_batchm",
                "dense_gemv_bf16_batchm",
            ),
            gemv_batchm_f32: metrale_model_layers::layers::try_kernel(
                gpu,
                "dense_gemv_bf16_batchm",
                "dense_gemv_bf16_batchm_fp32out",
            ),
            rms_norm: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            latent_write: gpu
                .kernel("glm5next_mla_latent_write", "glm5next_mla_latent_write_fp8")?,
        })
    }
}
