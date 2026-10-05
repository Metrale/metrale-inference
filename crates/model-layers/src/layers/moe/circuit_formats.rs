// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The format arms of `MoeLayer::circuit_bind`: FP8 experts (the grouped FP8
//! decode), the checkpoint's declared NVFP4 experts (the grouped NVFP4 decode on the lean or the
//! row-major tensor-core pair, `forward_nvfp4_grouped_decode.rs`) and BF16 experts (an MTP drafter
//! excluded from quantization: one row through `MoeLayer::forward`'s fused BF16 kernels, more
//! through the grouped BF16 point, `forward_bf16_grouped_decode.rs`). Each returns the tables,
//! the experts' kind and the expert pair the layer's own dispatch launches, or `None` with what it
//! does not model.
//!
//! Owner: model-layers (MoE).
//! Invariants:
//! - Each arm restates the weight, kernel and shape terms of the legacy predicate it mirrors
//!   (`fp8_grouped_decode_ok`, `nvfp4_grouped_decode_ok`, `bf16_grouped_decode_arena_ok`); the
//!   per-width arena terms are `MoeFacts::check_rows`.
//! - The expert pair is the one the legacy dispatch selects (`fp8_grouped_expert_kernels`,
//!   `Nvfp4GroupedKernels::select`), never chosen here.

use super::MoeLayer;
use super::circuit::{ExpertKind, Fp8Tables, MoeExperts};
use super::forward_nvfp4_grouped_decode::{
    nvfp4_grouped_decode_shape_ok, nvfp4_grouped_tc_enabled,
};
use super::fp8_grouped_tc::Fp8GroupedExpertKernels as ExpertPair;
use crate::layers::ops;

/// 2026-10-05: Push each present reason; whether any was.
fn refuse(arms: &[(bool, &str)], unmodelled: &mut Vec<String>) -> bool {
    let mut refused = false;
    for (present, what) in arms {
        if *present {
            unmodelled.push((*what).to_string());
            refused = true;
        }
    }
    refused
}

impl MoeLayer {
    /// 2026-10-03: The FP8 arm: the grouped FP8 decode's tables and expert pair, or `None`
    /// with what it does not model.
    pub(super) fn bind_fp8(
        &self,
        config: &metrale_config::ModelConfig,
        unmodelled: &mut Vec<String>,
    ) -> Option<(MoeExperts, ExpertKind, ExpertPair)> {
        let (h, inter) = (config.hidden_size, config.moe_intermediate_size);
        let arms = [
            (
                self.bf16_gate_weight_ptrs.is_some() || self.bf16_shared_expert.is_some(),
                "BF16 experts or a BF16 shared expert beside FP8 ones",
            ),
            (
                self.fp8_up_weight_ptrs.is_none()
                    || self.fp8_down_weight_ptrs.is_none()
                    || self.fp8_shared_expert.is_none(),
                "FP8 experts without their up, down or shared tables",
            ),
            (
                !super::forward_fp8_grouped_decode::fp8_grouped_decode_enabled(),
                "the grouped MoE decode off (METRALE_NO_FP8_MOE_GROUPED_DECODE)",
            ),
            (
                !self.fp8_grouped_tc_on(h, inter),
                "the grouped MoE decode off its tensor-core expert kernels (METRALE_NO_MOE_FP8_TC, \
                 or the kernels or shapes refused): the scalar arms are not bound",
            ),
            (
                !super::forward_fp8_grouped_decode::fp8_grouped_decode_dims_ok(
                    h as u32,
                    inter as u32,
                ),
                "a hidden or expert width the grouped decode refuses",
            ),
            (
                self.moe_expert_gate_up_act_fp8_grouped_k.0 == 0
                    || self.moe_expert_down_act_fp8_grouped_k.0 == 0,
                "the grouped MoE decode without its FP8 expert kernels",
            ),
        ];
        if refuse(&arms, unmodelled) {
            return None;
        }
        let (gp, up, dp, sh) = (
            self.fp8_gate_weight_ptrs.as_ref()?,
            self.fp8_up_weight_ptrs.as_ref()?,
            self.fp8_down_weight_ptrs.as_ref()?,
            self.fp8_shared_expert?,
        );
        let tables = |t: &super::Fp8ExpertPtrTable| Fp8Tables {
            weights: t.weight_ptrs,
            scales: t.scale_ptrs,
        };
        let experts = MoeExperts::Fp8 {
            gate: tables(gp),
            up: tables(up),
            down: tables(dp),
            shared: sh,
        };
        Some((
            experts,
            ExpertKind::Fp8,
            self.fp8_grouped_expert_kernels(h, inter),
        ))
    }

    /// 2026-10-05: The declared-NVFP4 arm (`set_declared_nvfp4_experts`): the routed and shared
    /// experts' NVFP4 tables and the pair `Nvfp4GroupedKernels::select` launches: the lean pair
    /// once the loader repacked them, else the row-major tensor-core pair. The CUDA-core pair
    /// (`METRALE_NO_MOE_NVFP4_TC`) is not bound.
    pub(super) fn bind_nvfp4(
        &self,
        config: &metrale_config::ModelConfig,
        unmodelled: &mut Vec<String>,
    ) -> Option<(MoeExperts, ExpertKind, ExpertPair)> {
        let (h, inter) = (config.hidden_size, config.moe_intermediate_size);
        let k = &self.nvfp4_grouped;
        let launch = k.select(h as u32, inter as u32);
        let tc = k.lean || nvfp4_grouped_tc_enabled() && launch.gate_up.0 == k.gate_up_tc.0;
        let sh = &self.weights.shared_expert;
        let routed = self.weights.experts.first();
        let nvfp4 = crate::weight_map::WeightQuantFormat::Nvfp4;
        let arms = [
            (
                !tc,
                "the grouped NVFP4 decode on its CUDA-core expert kernels \
                 (METRALE_NO_MOE_NVFP4_TC, or the tensor-core kernels or shapes refused)",
            ),
            (
                !nvfp4_grouped_decode_shape_ok(
                    1,
                    launch.max_rows,
                    h,
                    inter,
                    config.shared_expert_intermediate_size,
                ),
                "a hidden, expert or shared width the grouped NVFP4 decode refuses",
            ),
            (
                launch.gate_up.0 == 0 || launch.down.0 == 0,
                "the grouped NVFP4 decode without its expert kernels",
            ),
            (
                !routed.is_some_and(|e| {
                    !e.gate_proj.weight.is_null()
                        && !e.up_proj.weight.is_null()
                        && !e.down_proj.weight.is_null()
                }) || self.gate_ptrs.packed_ptrs.is_null()
                    || self.up_ptrs.packed_ptrs.is_null()
                    || self.down_ptrs.packed_ptrs.is_null(),
                "NVFP4 experts without their routed tables",
            ),
            (
                sh.gate_proj.weight.is_null()
                    || sh.up_proj.weight.is_null()
                    || sh.down_proj.weight.is_null(),
                "NVFP4 experts without an NVFP4 shared expert",
            ),
            (
                self.experts_scale_kind != nvfp4 || self.shared_experts_scale_kind != nvfp4,
                "NVFP4 experts whose scales are not the NVFP4 format",
            ),
            (
                self.bf16_gate_weight_ptrs.is_some() || self.bf16_shared_expert.is_some(),
                "BF16 experts or a BF16 shared expert beside NVFP4 ones",
            ),
            (
                self.use_t_layout_for_decode(),
                "the transposed decode layout",
            ),
        ];
        if refuse(&arms, unmodelled) {
            return None;
        }
        let tables = |t: &super::ExpertPtrTable| ops::Nvfp4ExpertTables {
            packed_ptrs: t.packed_ptrs,
            scale_ptrs: t.scale_ptrs,
            scale2_vals: t.scale2_vals,
        };
        let experts = MoeExperts::Nvfp4 {
            gate: tables(&self.gate_ptrs),
            up: tables(&self.up_ptrs),
            down: tables(&self.down_ptrs),
            shared: [sh.gate_proj, sh.up_proj, sh.down_proj],
        };
        let kind = if k.lean {
            ExpertKind::Nvfp4Lean
        } else {
            ExpertKind::Nvfp4TensorCore
        };
        let pair = ExpertPair {
            gate_up: launch.gate_up,
            gate_up_geometry: launch.gate_up_geometry,
            down: launch.down,
            down_geometry: launch.down_geometry,
        };
        Some((experts, kind, pair))
    }

    /// 2026-10-05: The BF16 arm (`set_bf16_experts`): one row runs `MoeLayer::forward`'s fused
    /// BF16 kernels, two or more the grouped BF16 tensor-core pair (`bf16_grouped_decode_arena_ok`).
    pub(super) fn bind_bf16(
        &self,
        config: &metrale_config::ModelConfig,
        unmodelled: &mut Vec<String>,
    ) -> Option<(MoeExperts, ExpertKind, ExpertPair)> {
        let (h, inter) = (
            config.hidden_size as u32,
            config.moe_intermediate_size as u32,
        );
        let k = &self.nvfp4_grouped;
        let arms = [
            (
                self.bf16_up_weight_ptrs.is_none()
                    || self.bf16_down_weight_ptrs.is_none()
                    || self.bf16_shared_expert.is_none(),
                "BF16 experts without their up, down or shared tables",
            ),
            (
                !ops::bf16_grouped_tc_shape_ok(inter, h, ops::BF16_GROUPED_GATE_UP_TC)
                    || !ops::bf16_grouped_tc_shape_ok(h, inter, ops::BF16_GROUPED_DOWN_TC),
                "a hidden or expert width the grouped BF16 decode refuses",
            ),
            (
                k.bf16_gate_up_tc.0 == 0 || k.bf16_down_tc.0 == 0,
                "the grouped BF16 decode without its expert kernels",
            ),
            (
                self.dense_gemv.0 == 0
                    || self.moe_topk.0 == 0
                    || self.moe_expert_gate_up_shared_bf16_k.0 == 0
                    || self.moe_expert_silu_down_shared_bf16_k.0 == 0
                    || self.moe_weighted_sum_blend.0 == 0,
                "the one-row BF16 MoE without its kernels",
            ),
        ];
        if refuse(&arms, unmodelled) {
            return None;
        }
        let sh = self.bf16_shared_expert.as_ref()?;
        let experts = MoeExperts::Bf16 {
            gate: self.bf16_gate_weight_ptrs?,
            up: self.bf16_up_weight_ptrs?,
            down: self.bf16_down_weight_ptrs?,
            shared: [sh.gate_proj, sh.up_proj, sh.down_proj],
        };
        let pair = ExpertPair {
            gate_up: k.bf16_gate_up_tc,
            gate_up_geometry: ops::BF16_GROUPED_GATE_UP_TC,
            down: k.bf16_down_tc,
            down_geometry: ops::BF16_GROUPED_DOWN_TC,
        };
        Some((experts, ExpertKind::Bf16, pair))
    }
}
