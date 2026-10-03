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
pub struct RouterF32 {
    pub(super) gemm: KernelHandle,
    pub(super) topk: KernelHandle,
}

impl RouterF32 {
    /// 2026-10-02: The kernels when `gate_f32`, `None` otherwise; an error names a missing one.
    pub fn load(gpu: &dyn GpuBackend, gate_f32: bool) -> Result<Option<Self>> {
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

/// 2026-10-02: Where one FP32 routing reads and writes: `normed` [n, hidden] BF16 in, `gate`
/// [num_experts, hidden] FP32 and `bias` [num_experts] FP32, `logits` [n, num_experts] FP32
/// scratch, then the top_k expert ids into `indices` [n, top_k] u32 and their weights into
/// `weights` [n, top_k] f32.
pub struct RouteF32Io {
    pub normed: DevicePtr,
    pub gate: DevicePtr,
    pub bias: DevicePtr,
    pub logits: DevicePtr,
    pub indices: DevicePtr,
    pub weights: DevicePtr,
}

/// 2026-10-02: The routing shape: `n` tokens over `num_experts` experts of `hidden` inputs,
/// `top_k` chosen, their weights normalized when `normalize`, then scaled by `scale`.
#[derive(Clone, Copy)]
pub struct RouteF32Shape {
    pub n: u32,
    pub num_experts: u32,
    pub hidden: u32,
    pub top_k: u32,
    pub normalize: bool,
    pub scale: f32,
}

/// 2026-10-02: Route `shape.n` tokens: the FP32 logits, then the sigmoid top-k over them.
pub fn launch_route_f32(
    gpu: &dyn GpuBackend,
    r: &RouterF32,
    io: &RouteF32Io,
    shape: RouteF32Shape,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, r.gemm)
        .grid([div_ceil(shape.num_experts, 8), shape.n, 1])
        .block([256, 1, 1])
        .arg_ptr(io.normed)
        .arg_ptr(io.gate)
        .arg_ptr(io.logits)
        .arg_u32(shape.n)
        .arg_u32(shape.num_experts)
        .arg_u32(shape.hidden)
        .launch(stream)?;
    KernelLaunch::new(gpu, r.topk)
        .grid([1, shape.n, 1])
        .block([256, 1, 1])
        .arg_ptr(io.logits)
        .arg_ptr(io.bias)
        .arg_ptr(io.indices)
        .arg_ptr(io.weights)
        .arg_u32(shape.num_experts)
        .arg_u32(shape.top_k)
        .arg_u32(u32::from(shape.normalize))
        .arg_f32(shape.scale)
        .arg_u32(shape.n)
        .launch(stream)
}

impl NemotronMoeLayer {
    /// 2026-10-02: Route `n` tokens of `normed` [n, hidden] BF16 into `indices` / `weights`
    /// ([`RouteF32Io`]), through the FP32 router-logit buffer.
    #[allow(clippy::too_many_arguments)]
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
        let io = RouteF32Io {
            normed,
            gate: self.weights.gate.weight,
            bias: self.weights.e_score_correction_bias.weight,
            logits: ctx.buffers.gate_logits_f32(),
            indices,
            weights,
        };
        let shape = RouteF32Shape {
            n,
            num_experts: self.weights.experts.len() as u32,
            hidden: ctx.config.hidden_size as u32,
            top_k: self.top_k as u32,
            normalize: ctx.config.norm_topk_prob,
            scale: ctx.config.routed_scaling_factor as f32,
        };
        launch_route_f32(ctx.gpu, r, &io, shape, stream)
    }
}

#[cfg(test)]
#[path = "route_f32_tests.rs"]
mod tests;
