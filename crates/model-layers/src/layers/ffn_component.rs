// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: `FfnComponent`, a layer's FFN (MoE, dense or none), and its per-pass dispatch.
//! Moved unchanged out of `layers/mod.rs`, which had reached the 500-line cap.
//!
//! Owner: model-layers.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::{DenseFfnLayer, MoeLayer, moe, ops};
use crate::layer::ForwardContext;

/// 2026-09-25: A layer's FFN: MoE, dense, or none.
#[allow(clippy::large_enum_variant)]
pub enum FfnComponent {
    Moe(MoeLayer),
    Dense(DenseFfnLayer),
    /// 2026-09-25: No FFN: `forward` returns its input and the other passes launch nothing.
    None,
}

impl FfnComponent {
    /// 2026-09-28: Add the FFN's weights to a circuit binding; a MoE or absent FFN is reported
    /// as unmodelled (the executor binds dense FFNs so far).
    pub(crate) fn circuit_bind(
        &self,
        levers: &ops::ModelLevers,
        weights: &mut std::collections::BTreeMap<
            crate::circuit_exec::WeightSlot,
            crate::circuit_exec::BoundWeight,
        >,
        unmodelled: &mut Vec<String>,
    ) {
        match self {
            Self::Dense(d) => d.circuit_bind(levers, weights, unmodelled),
            Self::Moe(_) => unmodelled.push("a MoE FFN (not bound yet)".to_string()),
            Self::None => unmodelled.push("no FFN".to_string()),
        }
    }

    /// 2026-09-28: Build what a dense FFN's binding hands out (`DenseFfnLayer::circuit_prepare`);
    /// nothing for a MoE or an absent FFN, which bind as unmodelled.
    pub(crate) fn circuit_prepare(
        &self,
        gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
        config: &metrale_config::ModelConfig,
        levers: &ops::ModelLevers,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Dense(d) => d.circuit_prepare(gpu, config, levers, stream),
            Self::Moe(_) | Self::None => Ok(()),
        }
    }

    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    pub fn is_dense(&self) -> bool {
        matches!(self, Self::Dense(_))
    }

    /// 2026-09-25: Whether this is a MoE whose `MoeLayer::grouped_decode_ok` holds,
    /// so multi-sequence decode may route it through `forward_prefill`
    /// (`qwen3_attention/trait_impl/multi_seq/ffn.rs`). False for dense and none.
    pub fn moe_grouped_decode_ok(&self) -> bool {
        match self {
            Self::Moe(m) => m.grouped_decode_ok(),
            _ => false,
        }
    }

    /// 2026-09-25: `MoeLayer::fp32_routing_active` for a MoE; false otherwise.
    pub fn fp32_routing_active(&self, levers: &ops::ModelLevers) -> bool {
        match self {
            Self::Moe(m) => m.fp32_routing_active(levers),
            _ => false,
        }
    }

    pub fn forward(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        match self {
            Self::Moe(m) => m.forward(input, ctx, stream),
            Self::Dense(d) => d.forward(input, ctx, stream),
            Self::None => Ok(input),
        }
    }

    pub fn forward_k2(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_k2(input, ctx, stream),
            Self::Dense(d) => d.forward_k2(input, ctx, stream),
            Self::None => Ok(()),
        }
    }

    pub fn forward_k3(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_k3(input, ctx, stream),
            Self::Dense(d) => d.forward_k3(input, ctx, stream),
            Self::None => Ok(()),
        }
    }

    /// 2026-09-30: Whether this is a dense FFN that runs `m` rows under a fixed
    /// `--activation-quantization` (`DenseFfnLayer::fixed_ok`). False for MoE and none.
    pub fn dense_fixed_ok(&self, m: usize, ctx: &ForwardContext) -> bool {
        matches!(self, Self::Dense(d) if d.fixed_ok(m, ctx))
    }

    /// 2026-09-30: `DenseFfnLayer::forward_fixed` over `m` rows into `moe_output`; call only
    /// when `dense_fixed_ok(m)` holds.
    pub fn forward_dense_fixed(
        &self,
        input: DevicePtr,
        m: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Dense(d) => d.forward_fixed(input, m, ctx, stream),
            _ => anyhow::bail!("forward_dense_fixed on a non-dense FFN"),
        }
    }

    /// 2026-09-25: Whether this is a dense FFN whose
    /// `DenseFfnLayer::can_forward_km(m)` holds: a batched-GEMV tier serves `m`
    /// rows and NVFP4 or FP8 weights are loaded. False for MoE and none.
    /// Multi-sequence attention decode (`qwen3_attention/trait_impl/multi_seq/ffn.rs`)
    /// asks before computing the pre-FFN norm.
    pub fn can_forward_km(&self, m: u32) -> bool {
        matches!(self, Self::Dense(d) if d.can_forward_km(m))
    }

    /// 2026-09-28: Row edge of a dense FFN's narrow decode arms (`DenseFfnLayer::narrow_rows`:
    /// its W4A4 edge under the weight-quantization tier, capped at 32, else the W4A16 edge).
    /// MoE and none answer the W4A16 edge; `can_forward_km` is false for them anyway.
    pub fn narrow_rows(&self) -> u32 {
        match self {
            Self::Dense(d) => d.narrow_rows(),
            _ => ops::gemv_tc::narrow_gemv_max_rows().min(ops::w4a4_proj::W4A4_MAX_M),
        }
    }

    /// 2026-09-25: `DenseFfnLayer::forward_km` over `m` rows. Returns `Ok(false)`
    /// without launching when `can_forward_km(m)` is false.
    pub fn try_forward_km(
        &self,
        input: DevicePtr,
        m: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        match self {
            Self::Dense(d) if d.can_forward_km(m) => {
                d.forward_km(input, m, ctx, stream)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    pub fn forward_prefill(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_prefill(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_prefill(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }

    pub fn forward_batched(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_batched(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_batched(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }

    pub fn forward_token_major_decode(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_token_major_decode(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_batched(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }

    pub fn forward_atomic_c4_decode(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_atomic_c4_decode(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_batched(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }

    /// 2026-09-25: Whether the target model's cross-row grouped FP8 MoE decode
    /// serves `m` rows: `ModelLevers::moe_fp8_grouped_decode_target`
    /// (`METRALE_FP8_MOE_GROUPED_DECODE`) is on and this is a MoE whose
    /// `MoeLayer::fp8_grouped_decode_ok(m)` holds. The MTP drafter calls
    /// `MoeLayer::fp8_grouped_decode_ok` directly, without the lever.
    pub fn fp8_grouped_decode_ok(&self, m: usize, ctx: &ForwardContext) -> bool {
        ctx.levers.moe_fp8_grouped_decode_target
            && matches!(self, Self::Moe(moe) if moe.fp8_grouped_decode_ok(m, ctx))
    }

    /// 2026-09-26: Whether this is a MoE whose grouped FP8 decode serves `m`
    /// rows with `routing` (`MoeLayer::fp8_grouped_routing_ok`). Unlike
    /// [`Self::fp8_grouped_decode_ok`] it does not read the
    /// `moe_fp8_grouped_decode_target` lever: the exact routings reproduce the
    /// path they replace byte for byte, so they need no opt-in.
    pub fn fp8_grouped_routing_ok(
        &self,
        m: usize,
        routing: moe::GroupedRouting,
        ctx: &ForwardContext,
    ) -> bool {
        matches!(self, Self::Moe(moe) if moe.fp8_grouped_routing_ok(m, routing, ctx))
    }

    /// 2026-09-26: [`Self::forward_fp8_grouped_decode`] with `routing`. Errors
    /// for dense and none; callers gate on `fp8_grouped_routing_ok` first.
    pub fn forward_fp8_grouped_decode_routed(
        &self,
        input: DevicePtr,
        m: usize,
        routing: moe::GroupedRouting,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(moe) => moe.forward_fp8_grouped_decode_routed(input, m, routing, ctx, stream),
            _ => anyhow::bail!("forward_fp8_grouped_decode_routed is MoE-only (m={m})"),
        }
    }

    /// 2026-09-25: Cross-row grouped FP8 MoE decode over `[m, H]` rows into
    /// `moe_output()`. Errors for dense and none; callers gate on
    /// `fp8_grouped_decode_ok` first.
    pub fn forward_fp8_grouped_decode(
        &self,
        input: DevicePtr,
        m: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(moe) => moe.forward_fp8_grouped_decode(input, m, ctx, stream),
            _ => anyhow::bail!("forward_fp8_grouped_decode is MoE-only (m={m})"),
        }
    }
}
