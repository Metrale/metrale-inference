// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The opt-in W8A8 expert step of the grouped FP8 MoE decode
//! (`moe_fp8_grouped_tc_w8a8.cu`): the layer input and the SiLU products are quantized to
//! E4M3 per (row, 128 group), the checkpoint's declared `activation_scheme: dynamic`, and the
//! FP8 weights go into E4M3 tensor-core MMAs undecoded.
//!
//! WHEN it runs is the weight-quantization policy's call, not this module's: the serve publishes
//! whether the experts decode with FP8 activations ([`set_moe_expert_fp8_act`], the
//! `--weight-quantization` policy's `fp8_decode_act` for the expert modules); nothing publishes
//! it by default, so the experts stay W8A16. This module says whether it CAN
//! ([`MoeLayer::fp8_grouped_tc_w8a8_on`]: the tensor-core path is on, the three W8A8 kernels
//! resolved and the shapes fit). Its output differs from the W8A16 kernels' by the activation
//! rounding; a row's bits still do not depend on the other rows.
//!
//! Owner: model-layers (MoE).
//! Invariants:
//! - The first publication or read of the cell wins (`OnceLock`); the serve publishes before
//!   the model runs, and anything that reads first fixes it at W8A16.
//! - [`MoeLayer::run_fp8_grouped_w8a8`] launches nothing unless
//!   [`MoeLayer::fp8_grouped_tc_w8a8_on`] holds and the buffers hold the layout.

use super::*;

static FP8_ACT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// 2026-09-28: Publish whether the MoE experts decode with FP8 activations (W8A8), as the
/// weight-quantization policy decides for the expert modules. Returns the value in force; a
/// caller that gets a different one should warn.
pub fn set_moe_expert_fp8_act(fp8: bool) -> bool {
    let _ = FP8_ACT.set(fp8);
    *FP8_ACT.get().expect("just set")
}

/// 2026-09-28: Whether the MoE experts decode with FP8 activations: false (W8A16) unless the
/// serve published otherwise.
pub fn moe_expert_fp8_act() -> bool {
    *FP8_ACT.get_or_init(|| false)
}

/// 2026-09-28: One grouped decode's inputs and outputs for the expert step.
pub(super) struct GroupedExpertIo {
    pub input: DevicePtr,
    pub act: DevicePtr,
    pub shared_act: DevicePtr,
    pub expert_down_out: DevicePtr,
    pub shared_out: DevicePtr,
    pub rows: ops::Fp8GroupedW8a8Rows,
}

impl MoeLayer {
    /// 2026-09-28: Whether this layer's grouped decode runs the W8A8 expert step: FP8 expert
    /// activations published ([`moe_expert_fp8_act`]), the tensor-core W8A16 kernels'
    /// conditions, the three W8A8 kernels, and gate+up and down in whole CTAs of their tiles.
    pub(super) fn fp8_grouped_tc_w8a8_on(&self, hidden: usize, inter: usize) -> bool {
        let (h, i) = (hidden as u32, inter as u32);
        let k = &self.fp8_grouped_tc;
        moe_expert_fp8_act()
            && self.fp8_grouped_tc_on(hidden, inter)
            && k.quant_w8a8.0 != 0
            && k.gate_up_w8a8.0 != 0
            && k.down_w8a8.0 != 0
            && ops::fp8_grouped_tc_shape_ok(i, h, ops::FP8_GROUPED_GATE_UP_TC_W8A8)
            && ops::fp8_grouped_tc_shape_ok(h, i, ops::FP8_GROUPED_DOWN_TC_W8A8)
    }

    /// 2026-09-28: Quantize the layer input, then W8A8 gate+up (quantized SiLU products) and
    /// down, routed and shared, into `io.expert_down_out` / `io.shared_out`. The quantized
    /// activations live in the SiLU buffers (`ops::Fp8GroupedW8a8Layout`).
    pub(super) fn run_fp8_grouped_w8a8(
        &self,
        io: &GroupedExpertIo,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let (Some(gp), Some(up), Some(dp), Some(sh)) = (
            &self.fp8_gate_weight_ptrs,
            &self.fp8_up_weight_ptrs,
            &self.fp8_down_weight_ptrs,
            &self.fp8_shared_expert,
        ) else {
            anyhow::bail!("W8A8 grouped decode: the FP8 expert tables are missing");
        };
        let cfg = ctx.config;
        let (h, inter) = (cfg.hidden_size, cfg.moe_intermediate_size);
        let m = io.rows.num_tokens as usize;
        let lay = ops::Fp8GroupedW8a8Layout::new(
            m,
            cfg.num_experts_per_tok,
            h,
            inter,
            ctx.buffers.expert_gate_out_bytes(),
            ctx.buffers.logits_bytes(),
        )?;
        let k = &self.fp8_grouped_tc;
        let (xq, xs) = (io.act.offset(lay.xq), io.act.offset(lay.xs));
        let act = (io.act, io.act.offset(lay.act_s));
        let sh_act = (io.shared_act, io.shared_act.offset(lay.sh_s));
        let (h32, i32) = (h as u32, inter as u32);
        ops::moe_act_quant_e4m3(
            ctx.gpu,
            k.quant_w8a8,
            io.input,
            xq,
            xs,
            m as u32,
            h32,
            stream,
        )?;
        ops::moe_expert_gate_up_act_fp8_grouped_tc_w8a8(
            ctx.gpu,
            k.gate_up_w8a8,
            xq,
            xs,
            (gp.weight_ptrs, gp.scale_ptrs),
            (up.weight_ptrs, up.scale_ptrs),
            act,
            &io.rows,
            &sh.gate_proj,
            &sh.up_proj,
            sh_act,
            i32,
            h32,
            stream,
        )?;
        ops::moe_expert_down_act_fp8_grouped_tc_w8a8(
            ctx.gpu,
            k.down_w8a8,
            act,
            (dp.weight_ptrs, dp.scale_ptrs),
            io.expert_down_out,
            &io.rows,
            sh_act,
            &sh.down_proj,
            io.shared_out,
            h32,
            i32,
            stream,
        )
    }
}
