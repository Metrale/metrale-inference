// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Which expert kernels the grouped FP8 MoE decode
//! (`forward_fp8_grouped_decode.rs`) launches: the scalar ones
//! (`moe_shared_expert_fused_fp8_grouped.cu`, FP32 SiLU products) or the tensor-core ones
//! (`moe_fp8_grouped_tc.cu`, the same FP32 SiLU products).
//!
//! The tensor-core kernels are the default (`METRALE_NO_MOE_FP8_TC` keeps the scalar ones).
//! They read the same FP8 weights at the same precision (W8A16) and stream them at the same
//! rate, for 30-45% fewer GPU-rail joules per layer on GB10. They sum in a different order (the
//! FP32 SiLU product enters the down MMAs as two BF16 terms), so the output bits differ from
//! the scalar kernels' by about 2e-4 relative. A row's bits still do not depend on how many
//! rows share the launch, so under the tensor-core kernels the grouped decode also serves one
//! row (`MoeLayer::forward` delegates to it) and a row-invariant tier policy holds.
//!
//! Owner: model-layers (MoE).
//! Invariants: the tensor-core kernels run only when both resolved and the layer's
//! projections pass `ops::fp8_grouped_tc_shape_ok`; the SiLU buffers they fill are read
//! only by the matching down kernel.

use super::*;

/// 2026-09-28: The tensor-core expert kernels, looked up with `try_kernel`; a zero handle
/// keeps the scalar kernels.
/// The three `_w8a8` handles are the opt-in W8A8 twin's (`fp8_grouped_tc_w8a8.rs`).
pub(super) struct Fp8GroupedTcKernels {
    pub gate_up: KernelHandle,
    pub down: KernelHandle,
    pub quant_w8a8: KernelHandle,
    pub gate_up_w8a8: KernelHandle,
    pub down_w8a8: KernelHandle,
}

impl Fp8GroupedTcKernels {
    /// 2026-09-28: One direct `try_kernel` call per kernel (`#[track_caller]` audit lines).
    pub(super) fn resolve(gpu: &dyn GpuBackend) -> Self {
        use super::super::try_kernel;
        const MODULE: &str = "moe_fp8_grouped_tc";
        const W8A8: &str = "moe_fp8_grouped_tc_w8a8";
        Self {
            gate_up: try_kernel(gpu, MODULE, "moe_expert_gate_up_act_fp8_grouped_tc"),
            down: try_kernel(gpu, MODULE, "moe_expert_down_act_fp8_grouped_tc"),
            quant_w8a8: try_kernel(gpu, W8A8, "moe_act_quant_e4m3"),
            gate_up_w8a8: try_kernel(gpu, W8A8, "moe_expert_gate_up_act_fp8_grouped_tc_w8a8"),
            down_w8a8: try_kernel(gpu, W8A8, "moe_expert_down_act_fp8_grouped_tc_w8a8"),
        }
    }
}

/// 2026-09-28: The grouped FP8 MoE decode takes the tensor-core expert kernels unless
/// `METRALE_NO_MOE_FP8_TC` is present (a debugging kill switch: the scalar kernels). Read once
/// per process.
fn fp8_grouped_tc_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METRALE_NO_MOE_FP8_TC").is_none())
}

/// 2026-10-03: The process-wide half of [`MoeLayer::fp8_grouped_tc_on`]: the kill switch is
/// absent or FP8 expert activations are published. The circuit's `moe_fp8_tc` policy setting;
/// the per-layer half (kernels resolved, shapes) is checked where the layer binds.
pub fn fp8_grouped_tc_setting() -> bool {
    fp8_grouped_tc_enabled() || super::moe_expert_fp8_act()
}

/// 2026-09-28: The gate+up and down launches of one grouped decode.
pub(super) struct Fp8GroupedExpertKernels {
    pub gate_up: KernelHandle,
    pub gate_up_geometry: ops::Fp8GroupedGeometry,
    pub down: KernelHandle,
    pub down_geometry: ops::Fp8GroupedGeometry,
}

impl MoeLayer {
    /// 2026-09-28: Whether this layer's grouped FP8 decode runs the tensor-core expert
    /// kernels: not switched off, both kernels resolved, and gate+up (`inter` x `hidden`) and
    /// down (`hidden` x `inter`) fit their tiles.
    ///
    /// 2026-09-28: Published FP8 expert activations (`super::moe_expert_fp8_act`) turn the
    /// tensor-core path on too: the W8A8 step is part of it, and its one-row delegation keeps
    /// a row's bits independent of the row count.
    pub(super) fn fp8_grouped_tc_on(&self, hidden: usize, inter: usize) -> bool {
        let (h, i) = (hidden as u32, inter as u32);
        fp8_grouped_tc_setting()
            && self.fp8_grouped_tc.gate_up.0 != 0
            && self.fp8_grouped_tc.down.0 != 0
            && ops::fp8_grouped_tc_shape_ok(i, h, ops::FP8_GROUPED_GATE_UP_TC)
            && ops::fp8_grouped_tc_shape_ok(h, i, ops::FP8_GROUPED_DOWN_TC)
    }

    /// 2026-09-28: The expert kernels of the grouped FP8 decode on this layer.
    pub(super) fn fp8_grouped_expert_kernels(
        &self,
        hidden: usize,
        inter: usize,
    ) -> Fp8GroupedExpertKernels {
        if self.fp8_grouped_tc_on(hidden, inter) {
            Fp8GroupedExpertKernels {
                gate_up: self.fp8_grouped_tc.gate_up,
                gate_up_geometry: ops::FP8_GROUPED_GATE_UP_TC,
                down: self.fp8_grouped_tc.down,
                down_geometry: ops::FP8_GROUPED_DOWN_TC,
            }
        } else {
            Fp8GroupedExpertKernels {
                gate_up: self.moe_expert_gate_up_act_fp8_grouped_k,
                gate_up_geometry: ops::FP8_GROUPED_GATE_UP_SCALAR,
                down: self.moe_expert_down_act_fp8_grouped_k,
                down_geometry: ops::FP8_GROUPED_DOWN_SCALAR,
            }
        }
    }

    /// 2026-09-28: `MoeLayer::forward` of one row through the grouped decode with this
    /// path's router (`GroupedRouting::PerRow`), so the row gets the bits it gets in a batch;
    /// `None` when the tensor-core kernels are off or the grouped decode declines the row.
    pub(super) fn forward_fp8_grouped_tc_one_row(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Option<Result<DevicePtr>> {
        let per_row = GroupedRouting::PerRow;
        (self.fp8_grouped_tc_on(ctx.config.hidden_size, ctx.config.moe_intermediate_size)
            && self.fp8_grouped_routing_ok(1, per_row, ctx))
        .then(|| {
            self.forward_fp8_grouped_decode_routed(input, 1, per_row, ctx, stream)
                .map(|()| ctx.buffers.moe_output())
        })
    }
}
