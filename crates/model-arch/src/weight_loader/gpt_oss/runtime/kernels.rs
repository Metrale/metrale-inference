// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: All composed kernels are required; no silent fallback handle.
use anyhow::Result;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};
pub(super) struct Kernels {
    pub norm: KernelHandle,
    pub gemv: KernelHandle,
    pub bias: KernelHandle,
    pub projection_bias: KernelHandle,
    pub rope: KernelHandle,
    pub cache: KernelHandle,
    pub attention: KernelHandle,
    pub router: KernelHandle,
    pub mxfp4: KernelHandle,
    pub activation: KernelHandle,
    pub reduce: KernelHandle,
    pub residual: KernelHandle,
}
impl Kernels {
    pub fn new(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            norm: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            gemv: gpu.kernel("gemv", "dense_gemv_bf16_fp32out")?,
            projection_bias: gpu.kernel("projection_bias", "projection_bias_bf16")?,
            bias: gpu.kernel("gpt_oss_expert_ops", "gpt_oss_selected_bias_bf16")?,
            rope: gpu.kernel("gpt_oss_rope", "gpt_oss_rope_bf16")?,
            cache: gpu.kernel("reshape_and_cache", "reshape_and_cache_flash")?,
            attention: gpu.kernel("paged_decode", "paged_decode_attn_sink")?,
            router: gpu.kernel("moe_topk", "moe_topk_selected_bf16_rows")?,
            mxfp4: gpu.kernel("gpt_oss_mxfp4_gemv", "gpt_oss_mxfp4_selected_bf16")?,
            activation: gpu.kernel("gpt_oss_expert_ops", "gpt_oss_swiglu_bf16")?,
            reduce: gpu.kernel("gpt_oss_expert_ops", "gpt_oss_expert_reduce_bf16")?,
            residual: gpu.kernel("residual_add", "bf16_residual_add")?,
        })
    }
}
