// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The FP32 router of [`super::NemotronMoeLayer`], for a checkpoint that stores the
//! router FP32 (Nemotron-3-Nano): the logits as HF's `NemotronHTopkRouter` computes them (FP32
//! weight, products, sums and logits; `nemotron_router_f32`), and the sigmoid top-k over the FP32
//! logits (`nemotron_moe_topk_sigmoid_batched_f32`), for every token at once.
//!
//! Owner: model-arch (Nemotron-H).
//! Invariants:
//! - A layer with an FP32 router has both kernels or does not build: the router is never cast to
//!   a narrower format than the checkpoint stores.

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use metrale_model_layers::layer::ForwardContext;

use super::NemotronMoeLayer;

/// 2026-10-02: The FP32 router's kernels.
pub(super) struct RouterF32 {
    gemm: KernelHandle,
    topk: KernelHandle,
}

impl RouterF32 {
    /// 2026-10-02: The kernels when `gate_f32`, `None` otherwise; an error names a missing one.
    pub(super) fn load(gpu: &dyn GpuBackend, gate_f32: bool) -> Result<Option<Self>> {
        if !gate_f32 {
            return Ok(None);
        }
        let k = |f: &str| {
            gpu.kernel("nemotron_moe_prefill", f).with_context(|| {
                format!(
                    "the checkpoint stores the MoE router FP32, which runs on \
                     nemotron_moe_prefill::{f}; this target does not ship it"
                )
            })
        };
        Ok(Some(Self {
            gemm: k("nemotron_router_f32")?,
            topk: k("nemotron_moe_topk_sigmoid_batched_f32")?,
        }))
    }
}

impl NemotronMoeLayer {
    /// 2026-10-02: Route `n` tokens of `normed` [n, hidden] BF16: the top_k expert indices and
    /// weights of each into `indices` [n, top_k] u32 and `weights` [n, top_k] f32. The logits go
    /// through the FP32 router-logit buffer.
    pub(super) fn route_f32(
        &self,
        r: &RouterF32,
        ctx: &ForwardContext,
        normed: DevicePtr,
        n: u32,
        indices: DevicePtr,
        weights: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let logits = ctx.buffers.gate_logits_f32();
        let num_experts = self.weights.experts.len() as u32;
        KernelLaunch::new(ctx.gpu, r.gemm)
            .grid([div_ceil(num_experts, 8), n, 1])
            .block([256, 1, 1])
            .arg_ptr(normed)
            .arg_ptr(self.weights.gate.weight)
            .arg_ptr(logits)
            .arg_u32(n)
            .arg_u32(num_experts)
            .arg_u32(ctx.config.hidden_size as u32)
            .launch(stream)?;
        KernelLaunch::new(ctx.gpu, r.topk)
            .grid([1, n, 1])
            .block([256, 1, 1])
            .arg_ptr(logits)
            .arg_ptr(self.weights.e_score_correction_bias.weight)
            .arg_ptr(indices)
            .arg_ptr(weights)
            .arg_u32(num_experts)
            .arg_u32(self.top_k as u32)
            .arg_u32(u32::from(ctx.config.norm_topk_prob))
            .arg_f32(ctx.config.routed_scaling_factor as f32)
            .arg_u32(n)
            .launch(stream)
    }
}

#[cfg(test)]
#[path = "route_f32_tests.rs"]
mod tests;
