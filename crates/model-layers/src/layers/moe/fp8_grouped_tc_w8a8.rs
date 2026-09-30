// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The opt-in W8A8 expert step of the grouped FP8 MoE decode
//! (`moe_fp8_grouped_tc_w8a8.cu`): the layer input and the SiLU products are quantized to
//! E4M3 per (row, 128 group), the checkpoint's declared `activation_scheme: dynamic`, and the
//! FP8 weights go into E4M3 tensor-core MMAs undecoded.
//!
//! 2026-09-29: Or ([`MoeExpertDecode::W8a8GateUp`]) only gate+up takes the E4M3 input: its
//! `_hilo` entry keeps the SiLU product at FP32 precision (BF16 hi|lo, the W8A16 layout) and the
//! W8A16 tensor-core down kernel reads it, so the down projection runs above the declared FP8
//! activations. The model asks for that in its MODEL.toml (`[behavior] expert_down_w8a16`).
//!
//! WHEN it runs is the weight-quantization policy's call, not this module's: the serve publishes
//! the expert decode ([`set_moe_expert_decode`], from the `--weight-quantization` policy's
//! `fp8_decode_act` for the expert gate and down modules, published at serve setup);
//! unpublished, the experts stay W8A16. This module says whether it CAN
//! ([`MoeLayer::fp8_grouped_tc_w8a8_on`]: the tensor-core path is on, the step's W8A8 kernels
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

/// 2026-09-29: The activations the grouped FP8 MoE decode feeds its expert projections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeExpertDecode {
    /// 2026-09-29: BF16 activations into gate+up and down (W8A16).
    W8a16,
    /// 2026-09-28: E4M3 input and E4M3 SiLU product (W8A8).
    W8a8,
    /// 2026-09-29: E4M3 input into gate+up; the SiLU product at FP32 precision into the W8A16
    /// down.
    W8a8GateUp,
}

static DECODE: std::sync::OnceLock<MoeExpertDecode> = std::sync::OnceLock::new();

/// 2026-09-28: Publish the expert decode the weight-quantization policy decides for the expert
/// modules. Returns the value in force; a caller that gets a different one should warn.
pub fn set_moe_expert_decode(decode: MoeExpertDecode) -> MoeExpertDecode {
    let _ = DECODE.set(decode);
    *DECODE.get().expect("just set")
}

/// 2026-09-28: The expert decode in force: W8A16 unless the serve published otherwise.
pub fn moe_expert_decode() -> MoeExpertDecode {
    *DECODE.get_or_init(|| MoeExpertDecode::W8a16)
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
    /// 2026-09-28: Whether this layer's grouped decode runs the W8A8 expert step: a W8A8
    /// decode published ([`moe_expert_decode`]), the tensor-core W8A16 kernels' conditions,
    /// the step's W8A8 kernels, and its W8A8 projections in whole CTAs of their tiles.
    pub(super) fn fp8_grouped_tc_w8a8_on(&self, hidden: usize, inter: usize) -> bool {
        let (h, i) = (hidden as u32, inter as u32);
        let k = &self.fp8_grouped_tc;
        let gate_up = ops::fp8_grouped_tc_shape_ok(i, h, ops::FP8_GROUPED_GATE_UP_TC_W8A8);
        let step = match moe_expert_decode() {
            MoeExpertDecode::W8a16 => false,
            MoeExpertDecode::W8a8 => {
                k.gate_up_w8a8.0 != 0
                    && k.down_w8a8.0 != 0
                    && gate_up
                    && ops::fp8_grouped_tc_shape_ok(h, i, ops::FP8_GROUPED_DOWN_TC_W8A8)
            }
            MoeExpertDecode::W8a8GateUp => k.gate_up_w8a8_hilo.0 != 0 && gate_up,
        };
        step && self.fp8_grouped_tc_on(hidden, inter) && k.quant_w8a8.0 != 0
    }

    /// 2026-09-28: Quantize the layer input, then W8A8 gate+up (quantized SiLU products) and
    /// down, routed and shared, into `io.expert_down_out` / `io.shared_out`. The quantized
    /// activations live in the SiLU buffers (`ops::Fp8GroupedW8a8Layout`).
    /// 2026-09-29: Under [`MoeExpertDecode::W8a8GateUp`], [`Self::run_fp8_grouped_w8a8_gate_up`].
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
        if moe_expert_decode() == MoeExpertDecode::W8a8GateUp {
            return self.run_fp8_grouped_w8a8_gate_up(io, ctx, stream);
        }
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

    /// 2026-09-29: Quantize the layer input into `io.expert_down_out`
    /// (`ops::Fp8GroupedW8a8HiloLayout`), W8A8 gate+up with the SiLU products as BF16 hi|lo in
    /// the SiLU buffers, then the W8A16 tensor-core down over them into `io.expert_down_out` /
    /// `io.shared_out`, which overwrites the quantized input after gate+up has read it.
    fn run_fp8_grouped_w8a8_gate_up(
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
            anyhow::bail!("W8A8 gate+up grouped decode: the FP8 expert tables are missing");
        };
        let cfg = ctx.config;
        let (h, inter) = (cfg.hidden_size, cfg.moe_intermediate_size);
        let m = io.rows.num_tokens as usize;
        let lay = ops::Fp8GroupedW8a8HiloLayout::new(
            m,
            cfg.num_experts_per_tok,
            h,
            inter,
            ctx.buffers.expert_gate_out_bytes(),
            ctx.buffers.logits_bytes(),
            ctx.buffers.expert_down_out_bytes(),
        )?;
        let k = &self.fp8_grouped_tc;
        let (xq, xs) = (io.expert_down_out, io.expert_down_out.offset(lay.xs));
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
            k.gate_up_w8a8_hilo,
            xq,
            xs,
            (gp.weight_ptrs, gp.scale_ptrs),
            (up.weight_ptrs, up.scale_ptrs),
            (io.act, DevicePtr::NULL),
            &io.rows,
            &sh.gate_proj,
            &sh.up_proj,
            (io.shared_act, DevicePtr::NULL),
            i32,
            h32,
            stream,
        )?;
        ops::moe_expert_down_act_fp8_grouped(
            ctx.gpu,
            k.down,
            ops::FP8_GROUPED_DOWN_TC,
            io.act,
            dp.weight_ptrs,
            dp.scale_ptrs,
            io.expert_down_out,
            io.rows.expert_offsets,
            io.rows.active_experts,
            io.rows.active_count,
            io.shared_act,
            &sh.down_proj,
            io.shared_out,
            h32,
            i32,
            io.rows.cap,
            io.rows.num_tokens,
            stream,
        )
    }
}
