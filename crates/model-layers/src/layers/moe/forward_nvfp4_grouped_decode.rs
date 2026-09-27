// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-27: `MoeLayer::forward_nvfp4_grouped_decode`: the NVFP4 MoE of `m` rows in one
//! routed+shared expert dispatch, under `--moe-nvfp4-experts`.
//!
//! The steps are those of the grouped FP8 decode (`forward_fp8_grouped_decode.rs`): the
//! per-row router (`GroupedRouting::PerRow`), `moe_sort_by_expert`, the active-expert
//! compaction, gate+up and SiLU, down, and the grouped blend. The expert kernels
//! (`ops/nvfp4_moe_grouped.rs`) read each active expert's NVFP4 weights once per pass for all
//! the rows routed to it. Every step computes a row independently of the other rows, so a
//! row's output bits do not depend on `m`: the path serves every width from one row up, and a
//! row-invariant tier policy (`row_tiers.rs`) holds with it on.
//! `model-arch/examples/nvfp4_moe_grouped_microtest.rs` checks this.
//!
//! Owner: model-layers (MoE).
//! Invariants: `forward_nvfp4_grouped_decode` launches nothing unless
//! `nvfp4_grouped_decode_ok(m, ctx)` holds.

use super::*;

/// 2026-09-27: Widest row count this path admits, the grouped FP8 decode's.
pub const NVFP4_GROUPED_DECODE_MAX_ROWS: usize =
    super::forward_fp8_grouped_decode::FP8_GROUPED_DECODE_MAX_ROWS;

/// 2026-09-27: The two expert kernels of this path, looked up with `try_kernel`; a zero handle
/// declines it. The sort, compaction, router and blend are the grouped FP8 decode's.
pub(super) struct Nvfp4GroupedKernels {
    pub gate_up: KernelHandle,
    pub down: KernelHandle,
}

impl Nvfp4GroupedKernels {
    /// 2026-09-27: One direct `try_kernel` call per kernel (`#[track_caller]` audit lines).
    pub(super) fn resolve(gpu: &dyn GpuBackend) -> Self {
        use super::super::try_kernel;
        const MODULE: &str = "moe_nvfp4_grouped";
        Self {
            gate_up: try_kernel(gpu, MODULE, "moe_expert_gate_up_act_nvfp4_grouped"),
            down: try_kernel(gpu, MODULE, "moe_expert_down_act_nvfp4_grouped"),
        }
    }
}

/// 2026-09-27: Shape admission without a GPU: `m` in `1..=NVFP4_GROUPED_DECODE_MAX_ROWS`,
/// `hidden % 32 == 0` (gate+up reads 32-element chunks) and `inter % 16 == 0` (down reads
/// 16-element blocks), and a shared expert as wide as a routed one (the shared SiLU product
/// shares the routed layout).
pub fn nvfp4_grouped_decode_shape_ok(
    m: usize,
    hidden: usize,
    inter: usize,
    shared_inter: usize,
) -> bool {
    (1..=NVFP4_GROUPED_DECODE_MAX_ROWS).contains(&m)
        && hidden >= 32
        && hidden.is_multiple_of(32)
        && inter >= 16
        && inter.is_multiple_of(16)
        && shared_inter == inter
}

impl MoeLayer {
    /// 2026-09-27: Whether `forward_nvfp4_grouped_decode` serves `m` rows on this layer:
    /// `--moe-nvfp4-experts` is on, the routed and shared experts are NVFP4 in the row-major
    /// decode layout, the per-row router applies (BF16 softmax gate, no correction bias,
    /// pre-router norm, FP32 gate or FP32 routing, no DFlash reroute), the kernels resolved,
    /// the shape is admitted, the arena is wide enough, and there is no LoRA, pre-expert norm,
    /// hash routing or expert parallelism.
    pub fn nvfp4_grouped_decode_ok(&self, m: usize, ctx: &ForwardContext) -> bool {
        let cfg = ctx.config;
        let (h, inter) = (cfg.hidden_size, cfg.moe_intermediate_size);
        let need = super::forward_fp8_grouped_decode::grouped_decode_buffer_need(
            m,
            h,
            inter,
            cfg.num_experts,
            cfg.num_experts_per_tok,
        );
        let b = ctx.buffers;
        crate::layers::moe_nvfp4_experts_enabled()
            && nvfp4_grouped_decode_shape_ok(m, h, inter, cfg.shared_expert_intermediate_size)
            && self.nvfp4_grouped.gate_up.0 != 0
            && self.nvfp4_grouped.down.0 != 0
            && self.moe_weighted_sum_blend_fp8_grouped_k.0 != 0
            && self.moe_fp8_grouped_compact_k.0 != 0
            && self.moe_sort_by_expert.0 != 0
            && self.moe_topk_softmax_rows_k.0 != 0
            && self.router_gemv_batchm_k.0 != 0
            && self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && self.shared_experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && !self.gate_ptrs.packed_ptrs.is_null()
            && !self.up_ptrs.packed_ptrs.is_null()
            && !self.down_ptrs.packed_ptrs.is_null()
            && !self.weights.shared_expert.gate_proj.weight.is_null()
            && !self.weights.shared_expert.up_proj.weight.is_null()
            && !self.weights.shared_expert.down_proj.weight.is_null()
            && self.fp8_gate_weight_ptrs.is_none()
            && self.bf16_gate_weight_ptrs.is_none()
            && self.bf16_shared_expert.is_none()
            && !self.use_t_layout_for_decode()
            && self.lora.is_none()
            && self.pre_expert_norm.is_none()
            && self.tid2eid_dev.is_none()
            && self.router_logits_n as usize == cfg.num_experts
            && self.gate_nvfp4.is_none()
            && self.correction_bias_dev.is_none()
            && self.weights.router_pre_norm.is_none()
            && !ctx.levers.fp32_gate
            && !self.fp32_routing_active(ctx.levers)
            && !(self.is_dflash_capture_layer && ctx.levers.frankenstein_decode_via_prefill)
            && h.is_multiple_of(8)
            && !(ctx.comm.is_some() && cfg.ep_world_size > 1)
            && b.scratch_bytes() >= need.scratch
            && b.gate_logits_bytes() >= need.gate_logits
            && b.expert_gate_out_bytes() >= need.expert_gate_out
            && b.expert_down_out_bytes() >= need.expert_down_out
            && b.logits_bytes() >= need.shared_act
            && b.attn_output_bytes() >= need.row_hidden
            && b.moe_output_bytes() >= need.row_hidden
    }

    /// 2026-09-27: The NVFP4 MoE of `m` rows: `input` is `[m, H]` BF16, the output lands in
    /// rows `0..m` of `moe_output()`. Returns an error, launching nothing, when
    /// `nvfp4_grouped_decode_ok(m, ctx)` is false.
    pub fn forward_nvfp4_grouped_decode(
        &self,
        input: DevicePtr,
        m: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            self.nvfp4_grouped_decode_ok(m, ctx),
            "forward_nvfp4_grouped_decode: predicate false for m={m} (caller must gate on it)"
        );
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let n = m as u32;
        let te = m * top_k as usize;
        if ctx.stats.once("log:moe_nvfp4_grouped_decode") {
            tracing::info!(
                "MoE NVFP4 grouped decode active (first use M={m}, top_k={top_k}, \
                 experts={num_experts}; one-time log)"
            );
        }

        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(te * 4);
        self.grouped_route(
            input,
            m,
            GroupedRouting::PerRow,
            indices_dev,
            weights_dev,
            ctx,
            stream,
        )?;

        // 2026-09-27: The sort scratch reuses `gate_logits`, which top-k has read earlier on
        // this stream (the layout of `grouped_decode_buffer_need`).
        let ne = num_experts as usize;
        let gate_logits = ctx.buffers.gate_logits();
        let sorted_token_ids = gate_logits;
        let sorted_expert_ids = gate_logits.offset(te * 4);
        let expert_offsets = gate_logits.offset(te * 4 * 2);
        let token_to_perm = gate_logits.offset(te * 4 * 2 + (ne + 1) * 4);
        ops::moe_sort_by_expert(
            ctx.gpu,
            self.moe_sort_by_expert,
            indices_dev,
            sorted_token_ids,
            sorted_expert_ids,
            expert_offsets,
            token_to_perm,
            te as u32,
            num_experts,
            top_k,
            stream,
        )?;
        let cap = ops::fp8_grouped_active_cap(n, top_k, num_experts);
        let active_experts = token_to_perm.offset(te * 4);
        let active_count = active_experts.offset(cap as usize * 4);
        ops::moe_fp8_grouped_compact(
            ctx.gpu,
            self.moe_fp8_grouped_compact_k,
            expert_offsets,
            active_experts,
            active_count,
            num_experts,
            stream,
        )?;

        let act = ctx.buffers.expert_gate_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        let shared_act = ctx.buffers.logits();
        let shared_out = ctx.buffers.attn_output();
        let tables = |t: &ExpertPtrTable| ops::Nvfp4ExpertTables {
            packed_ptrs: t.packed_ptrs,
            scale_ptrs: t.scale_ptrs,
            scale2_vals: t.scale2_vals,
        };
        let sh = &self.weights.shared_expert;
        ops::moe_expert_gate_up_act_nvfp4_grouped(
            ctx.gpu,
            self.nvfp4_grouped.gate_up,
            input,
            tables(&self.gate_ptrs),
            tables(&self.up_ptrs),
            act,
            expert_offsets,
            sorted_token_ids,
            active_experts,
            active_count,
            &sh.gate_proj,
            &sh.up_proj,
            shared_act,
            inter,
            h,
            cap,
            n,
            stream,
        )?;
        ops::moe_expert_down_act_nvfp4_grouped(
            ctx.gpu,
            self.nvfp4_grouped.down,
            act,
            tables(&self.down_ptrs),
            expert_down_out,
            expert_offsets,
            active_experts,
            active_count,
            shared_act,
            &sh.down_proj,
            shared_out,
            h,
            inter,
            cap,
            n,
            stream,
        )?;
        ops::moe_weighted_sum_blend_fp8_grouped(
            ctx.gpu,
            self.moe_weighted_sum_blend_fp8_grouped_k,
            ctx.buffers.moe_output(),
            expert_down_out,
            weights_dev,
            token_to_perm,
            shared_out,
            input,
            self.weights.shared_expert_gate.weight,
            h,
            top_k,
            h,
            n,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-27: The admitted widths and the shape terms, each refused alone.
    #[test]
    fn shape_admission() {
        assert!(nvfp4_grouped_decode_shape_ok(1, 2048, 512, 512));
        assert!(nvfp4_grouped_decode_shape_ok(
            NVFP4_GROUPED_DECODE_MAX_ROWS,
            2048,
            512,
            512
        ));
        assert!(!nvfp4_grouped_decode_shape_ok(0, 2048, 512, 512));
        assert!(!nvfp4_grouped_decode_shape_ok(
            NVFP4_GROUPED_DECODE_MAX_ROWS + 1,
            2048,
            512,
            512
        ));
        assert!(!nvfp4_grouped_decode_shape_ok(4, 2048 + 16, 512, 512));
        assert!(!nvfp4_grouped_decode_shape_ok(4, 2048, 512 + 8, 512 + 8));
        assert!(!nvfp4_grouped_decode_shape_ok(4, 2048, 512, 1024));
    }
}
