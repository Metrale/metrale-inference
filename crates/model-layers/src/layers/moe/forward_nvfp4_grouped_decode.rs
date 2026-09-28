// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-27: `MoeLayer::forward_nvfp4_grouped_decode`: the NVFP4 MoE of `m` rows in one
//! routed+shared expert dispatch, under an NVFP4 `--expert-quantization` tier.
//!
//! The steps are those of the grouped FP8 decode (`forward_fp8_grouped_decode.rs`): the
//! per-row router (`GroupedRouting::PerRow`), `moe_fp8_grouped_sort` (the slot sort and the
//! active-expert list), gate+up and SiLU, down, and the grouped blend. The expert kernels
//! (`ops/nvfp4_moe_grouped.rs`) read each active expert's NVFP4 weights once per pass for all
//! the rows routed to it. When the layer keeps its FP8 experts too (the qwen35 loader does,
//! for a native-FP8 checkpoint), the shared expert runs in FP8 through the grouped FP8 kernels
//! (with no active experts), the NVFP4 kernels take only the routed experts, and prefill keeps
//! the FP8 experts; under `nvfp4-gate-up` every down projection, routed and shared, runs
//! through the grouped FP8 down kernel instead. Every step computes a row independently of
//! the other rows, so a row's output bits do not depend on `m`: the path serves every width
//! from one row up, and a row-invariant tier policy (`row_tiers.rs`) holds with it on.
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
/// declines it. The sort, router and blend are the grouped FP8 decode's.
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
    /// 2026-09-27: Whether `forward_nvfp4_grouped_decode` serves `m` rows on this layer: an
    /// NVFP4 `--expert-quantization` tier is in force, the routed projections that tier decodes
    /// as NVFP4 are present in the row-major decode layout, the shared expert (and, under
    /// `nvfp4-gate-up`, the routed down projections) are there in FP8 or NVFP4 as the tier
    /// reads them, the per-row router applies (BF16 softmax gate, no correction bias,
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
        let tier = crate::layers::expert_quantization();
        let fp8_shared_ok = self.nvfp4_decode_fp8_shared().is_some()
            && self.moe_expert_gate_up_act_fp8_grouped_k.0 != 0
            && self.moe_expert_down_act_fp8_grouped_k.0 != 0;
        let nvfp4_shared_ok = !self.weights.shared_expert.gate_proj.weight.is_null()
            && !self.weights.shared_expert.up_proj.weight.is_null()
            && !self.weights.shared_expert.down_proj.weight.is_null();
        // 2026-09-27: The routed NVFP4 projections must have been built: a layer loaded without
        // them (the MTP head's native-FP8 MoE) has pointer tables of NULL entries, which the
        // kernels would read as zero experts. Expert parallelism is refused below, so every
        // expert is local and the first stands for all.
        let routed = self.weights.experts.first();
        let gate_up_ok =
            routed.is_some_and(|e| !e.gate_proj.weight.is_null() && !e.up_proj.weight.is_null());
        let down_ok = if tier.nvfp4_down() {
            self.nvfp4_grouped.down.0 != 0
                && !self.down_ptrs.packed_ptrs.is_null()
                && routed.is_some_and(|e| !e.down_proj.weight.is_null())
        } else {
            fp8_shared_ok && self.fp8_down_weight_ptrs.is_some()
        };
        tier.nvfp4_decode()
            && nvfp4_grouped_decode_shape_ok(m, h, inter, cfg.shared_expert_intermediate_size)
            && self.nvfp4_grouped.gate_up.0 != 0
            && gate_up_ok
            && down_ok
            && (fp8_shared_ok || nvfp4_shared_ok)
            && self.moe_weighted_sum_blend_fp8_grouped_k.0 != 0
            && self.moe_fp8_grouped_sort_k.0 != 0
            && cfg.num_experts <= ops::FP8_GROUPED_SORT_MAX_EXPERTS as usize
            && self.moe_topk_softmax_rows_k.0 != 0
            && self.router_gemv_batchm_k.0 != 0
            && self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && self.shared_experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && !self.gate_ptrs.packed_ptrs.is_null()
            && !self.up_ptrs.packed_ptrs.is_null()
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

    /// 2026-09-27: The FP8 shared expert this path runs instead of the NVFP4 one, when the
    /// layer keeps its FP8 experts (`set_fp8_experts`).
    fn nvfp4_decode_fp8_shared(&self) -> Option<&Fp8ExpertWeight> {
        self.fp8_shared_expert.as_ref()
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
        let cap = ops::fp8_grouped_active_cap(n, top_k, num_experts);
        let active_experts = token_to_perm.offset(te * 4);
        let active_count = active_experts.offset(cap as usize * 4);
        ops::moe_fp8_grouped_sort(
            ctx.gpu,
            self.moe_fp8_grouped_sort_k,
            ops::Fp8GroupedSortOut {
                sorted_token_ids,
                sorted_expert_ids,
                expert_offsets,
                token_to_perm,
                active_experts,
                active_count,
            },
            indices_dev,
            te as u32,
            num_experts,
            top_k,
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
        // 2026-09-27: With the layer's FP8 experts kept, the grouped FP8 gate+up kernel runs the
        // FP8 shared expert alone (no active experts) and the NVFP4 one only the routed experts
        // (no shared rows). The down projections: under `nvfp4` the grouped FP8 down kernel runs
        // the shared expert alone and the NVFP4 one the routed experts; under `nvfp4-gate-up`
        // the grouped FP8 down kernel runs all of them from the FP8 experts.
        let fp8_shared = self.nvfp4_decode_fp8_shared();
        let fp8_down = if crate::layers::expert_quantization().nvfp4_down() {
            None
        } else {
            self.fp8_down_weight_ptrs.as_ref().zip(fp8_shared)
        };
        if let Some(fsh) = fp8_shared {
            ops::moe_expert_gate_up_act_fp8_grouped(
                ctx.gpu,
                self.moe_expert_gate_up_act_fp8_grouped_k,
                input,
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
                act,
                expert_offsets,
                sorted_token_ids,
                active_experts,
                active_count,
                &fsh.gate_proj,
                &fsh.up_proj,
                shared_act,
                inter,
                h,
                0,
                n,
                stream,
            )?;
        }
        if let (Some(fsh), None) = (fp8_shared, fp8_down) {
            ops::moe_expert_down_act_fp8_grouped(
                ctx.gpu,
                self.moe_expert_down_act_fp8_grouped_k,
                act,
                DevicePtr::NULL,
                DevicePtr::NULL,
                expert_down_out,
                expert_offsets,
                active_experts,
                active_count,
                shared_act,
                &fsh.down_proj,
                shared_out,
                h,
                inter,
                0,
                n,
                stream,
            )?;
        }
        let nvfp4_shared_rows = if fp8_shared.is_some() { 0 } else { n };
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
            nvfp4_shared_rows,
            stream,
        )?;
        if let Some((dp, fsh)) = fp8_down {
            ops::moe_expert_down_act_fp8_grouped(
                ctx.gpu,
                self.moe_expert_down_act_fp8_grouped_k,
                act,
                dp.weight_ptrs,
                dp.scale_ptrs,
                expert_down_out,
                expert_offsets,
                active_experts,
                active_count,
                shared_act,
                &fsh.down_proj,
                shared_out,
                h,
                inter,
                cap,
                n,
                stream,
            )?;
        } else {
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
                nvfp4_shared_rows,
                stream,
            )?;
        }
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
