// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: Admission of the grouped NVFP4 MoE decode (`forward_nvfp4_grouped_decode.rs`):
//! `nvfp4_grouped_decode_ok`, which shared expert it runs, and the declared-experts marks.
//! Split out of `forward_nvfp4_grouped_decode.rs` (500-line cap) when the BF16 shared expert
//! was admitted.
//!
//! Owner: model-layers (MoE).
//! Invariants: `forward_nvfp4_grouped_decode` runs the shared expert
//! [`MoeLayer::nvfp4_grouped_shared`] names, and launches nothing when it names none.

use super::forward_nvfp4_grouped_decode::nvfp4_grouped_decode_shape_ok;
use super::*;

/// 2026-10-13: The shared expert one grouped NVFP4 decode runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Nvfp4GroupedShared {
    /// 2026-10-14: The layer's FP8 shared expert, through the grouped FP8 kernels with no active experts.
    Fp8,
    /// 2026-10-14: The NVFP4 shared expert, as the first block rows of the NVFP4 expert kernels.
    Nvfp4,
    /// 2026-10-14: The checkpoint's BF16 shared expert, through `run_bf16_shared_expert` (batched GEMV
    /// rows, each the one-row `dense_gemv`'s bits); the NVFP4 kernels launch no shared rows.
    Bf16,
}

/// 2026-10-13: The shared expert from what the layer holds: FP8 when its FP8 experts are kept
/// (and their kernels resolved), else NVFP4 when all three NVFP4 projections are there, else
/// BF16 when only a BF16 shared expert is installed. A BF16 shared expert beside an FP8 or
/// NVFP4 one is ambiguous and refused, as is a layer with none.
pub(super) fn grouped_shared_form(
    fp8_ok: bool,
    nvfp4_ok: bool,
    bf16_installed: bool,
) -> Option<Nvfp4GroupedShared> {
    match (fp8_ok, nvfp4_ok, bf16_installed) {
        (_, _, true) if fp8_ok || nvfp4_ok => None,
        (true, _, false) => Some(Nvfp4GroupedShared::Fp8),
        (false, true, false) => Some(Nvfp4GroupedShared::Nvfp4),
        (false, false, true) => Some(Nvfp4GroupedShared::Bf16),
        _ => None,
    }
}

impl MoeLayer {
    /// 2026-10-13: The shared expert `forward_nvfp4_grouped_decode` runs on this layer, or
    /// `None` when it has no admissible one ([`grouped_shared_form`]).
    pub(super) fn nvfp4_grouped_shared(&self) -> Option<Nvfp4GroupedShared> {
        let fp8_ok = self.nvfp4_decode_fp8_shared().is_some()
            && self.moe_expert_gate_up_act_fp8_grouped_k.0 != 0
            && self.moe_expert_down_act_fp8_grouped_k.0 != 0;
        let sh = &self.weights.shared_expert;
        let nvfp4_ok = !sh.gate_proj.weight.is_null()
            && !sh.up_proj.weight.is_null()
            && !sh.down_proj.weight.is_null();
        grouped_shared_form(fp8_ok, nvfp4_ok, self.bf16_shared_expert.is_some())
    }

    /// 2026-09-27: Whether `forward_nvfp4_grouped_decode` serves `m` rows on this layer: an
    /// NVFP4 `--expert-quantization` tier is in force or (2026-10-02) the experts are the
    /// checkpoint's declared NVFP4 (`set_declared_nvfp4_experts`), the routed projections that tier decodes
    /// as NVFP4 are present in the row-major decode layout, the shared expert (and, under
    /// `nvfp4-gate-up`, the routed down projections) are there in FP8 or NVFP4 as the tier
    /// reads them (2026-10-13: or the shared expert is the checkpoint's BF16 one), the per-row
    /// router applies (BF16 softmax gate, no correction bias,
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
        let shared = self.nvfp4_grouped_shared();
        // 2026-09-27: The routed NVFP4 projections must have been built: a layer loaded without
        // them (the MTP head's native-FP8 MoE) has pointer tables of NULL entries, which the
        // kernels would read as zero experts. Expert parallelism is refused below, so every
        // expert is local and the first stands for all.
        let routed = self.weights.experts.first();
        let gate_up_ok =
            routed.is_some_and(|e| !e.gate_proj.weight.is_null() && !e.up_proj.weight.is_null());
        let native = self.nvfp4_grouped.declared_experts;
        let launch = self.nvfp4_grouped.select(h as u32, inter as u32);
        let down_ok = if tier.nvfp4_down() || native {
            !self.down_ptrs.packed_ptrs.is_null()
                && routed.is_some_and(|e| !e.down_proj.weight.is_null())
        } else {
            shared == Some(Nvfp4GroupedShared::Fp8) && self.fp8_down_weight_ptrs.is_some()
        };
        (tier.nvfp4_decode() || native)
            && nvfp4_grouped_decode_shape_ok(
                m,
                launch.max_rows,
                h,
                inter,
                cfg.shared_expert_intermediate_size,
            )
            && launch.gate_up.0 != 0
            && launch.down.0 != 0
            && gate_up_ok
            && down_ok
            && shared.is_some()
            && self.moe_weighted_sum_blend_fp8_grouped_k.0 != 0
            && self.moe_fp8_grouped_sort_k.0 != 0
            && cfg.num_experts <= ops::FP8_GROUPED_SORT_MAX_EXPERTS as usize
            && self.moe_topk_softmax_rows_k.0 != 0
            && self.dense_gemv_batchm_k.0 != 0
            && self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && self.shared_experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && !self.gate_ptrs.packed_ptrs.is_null()
            && !self.up_ptrs.packed_ptrs.is_null()
            && self.bf16_gate_weight_ptrs.is_none()
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

    /// 2026-10-02: Mark this layer's routed and shared experts as the checkpoint's declared
    /// NVFP4, so `nvfp4_grouped_decode_ok` admits it under every `--expert-quantization` tier.
    /// Refused for a layer that also holds FP8 experts: the tier decides between those.
    pub fn set_declared_nvfp4_experts(&mut self) -> Result<()> {
        self.set_draft_nvfp4_experts()
    }

    /// 2026-10-02: The same mark for a drafter whose experts were requantized to NVFP4 for drafting
    /// (`MtpHead::new_nvfp4_draft_moe`): the grouped NVFP4 decode serves them at every width.
    pub fn set_draft_nvfp4_experts(&mut self) -> Result<()> {
        anyhow::ensure!(
            self.fp8_shared_expert.is_none() && self.fp8_down_weight_ptrs.is_none(),
            "set_declared_nvfp4_experts: the layer holds FP8 experts"
        );
        self.nvfp4_grouped.declared_experts = true;
        Ok(())
    }

    /// 2026-09-27: The FP8 shared expert this path runs instead of the NVFP4 one, when the
    /// layer keeps its FP8 experts (`set_fp8_experts`).
    pub(super) fn nvfp4_decode_fp8_shared(&self) -> Option<&Fp8ExpertWeight> {
        self.fp8_shared_expert.as_ref()
    }
}

#[cfg(test)]
#[path = "nvfp4_grouped_admit_tests.rs"]
mod tests;
