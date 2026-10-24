// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: Kernel handles of the served embedding model, looked up once at construction.
//! Every kernel is an LKB family the generative models already run (`gemm`, `norm`, `rope`,
//! `residual_add`, `moe_silu_mul`, `embed_from_argmax`), plus the head-dim-64 instantiation of
//! the common prefill attention (`attn_prefill_h64`) and the `mean_pool` family.
//!
//! Owner: model-engine (embedding model).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

pub(super) struct EmbedKernels {
    /// 2026-10-24: Tensor-core BF16 GEMM `C = A B^T`, FP32 accumulation (`dense_bf16`).
    pub gemm: KernelHandle,
    /// 2026-10-24: RMSNorm scaled by `weight` (Llama), not the `1 + weight` form of `norm::rms_norm`.
    pub rms_norm: KernelHandle,
    /// 2026-10-24: RoPE from a precomputed inverse-frequency table (the llama3 table).
    pub rope: KernelHandle,
    /// 2026-10-24: Contiguous prefill flash attention at head_dim 64, launched with causal = 0.
    pub attn: KernelHandle,
    pub residual_add: KernelHandle,
    pub silu_mul: KernelHandle,
    pub embed: KernelHandle,
    pub mean_pool: KernelHandle,
}

impl EmbedKernels {
    pub(super) fn new(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            gemm: gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?,
            rms_norm: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            rope: gpu.kernel("rope", "rope_forward_yarn")?,
            attn: gpu.kernel("attn_prefill_h64", "attn_prefill_h64")?,
            residual_add: gpu.kernel("residual_add", "bf16_residual_add")?,
            silu_mul: gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
            embed: gpu.kernel("embed_from_argmax", "batched_embed")?,
            mean_pool: gpu.kernel("mean_pool", "mean_pool_masked_f32")?,
        })
    }
}
