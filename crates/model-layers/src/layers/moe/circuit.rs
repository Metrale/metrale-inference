// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The MoE FFN as the circuit executor binds it: the router, the FP8 routed and
//! shared experts' tables, the shared-expert gate, the kernels this layer's own dispatch would
//! launch (the emitters refuse a plan whose kernel differs), and the arena scratch the grouped
//! decode sorts its slots into, which is no circuit edge.
//!
//! Bound arm: the grouped FP8 decode with the per-row router (`GroupedRouting::PerRow`,
//! `forward_fp8_grouped_router.rs`), which the target model takes at every row count under the
//! tensor-core expert kernels (`fp8_grouped_tc.rs`, one row through `MoeLayer::forward`) and
//! under a row-invariant tier policy, with its W8A8 expert step when the serve published FP8
//! expert activations (`fp8_grouped_tc_w8a8.rs`); and the batched router
//! (`GroupedRouting::Batched`) the MTP drafter's n-row propose runs.
//!
//! Owner: model-layers (MoE).
//! Invariants:
//! - Every feature of the layer that changes which arm `MoeLayer::forward` or the grouped
//!   decode takes, other than the ones above, is reported in `unmodelled`; the executor refuses
//!   a model with any.
//! - The per-row-count conditions of the grouped decode (its row envelope and the arena
//!   capacities it checks) are re-checked at compile time for each plan's rows
//!   ([`MoeFacts::check_rows`]), so a plan never runs a width legacy would decline.

use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::MoeLayer;
use super::forward_fp8_grouped_decode::{
    fp8_grouped_decode_rows_ok, grouped_decode_buffer_need, grouped_sort_out,
};
use crate::layers::ops;
use crate::weight_map::{DenseWeight, Fp8ExpertWeight};

/// 2026-10-03: One FP8 routed projection's per-expert tables: a u64 device pointer per expert
/// to its E4M3 weight and to its block scales.
#[derive(Debug, Clone, Copy)]
pub struct Fp8Tables {
    pub weights: DevicePtr,
    pub scales: DevicePtr,
}

/// 2026-10-03: The kernels this layer's own dispatch launches on the bound arms (zero where the
/// target lacks one). An emitter compares the plan's kernel with the one here.
#[derive(Debug, Clone, Copy)]
pub struct MoeKernels {
    /// 2026-10-03: `dense_gemv_bf16_batchm`, the per-row router.
    pub router_rows: KernelHandle,
    /// 2026-10-03: The batched router GEMM `router_gemm_bf16` launches: `moe_router_gemm_bf16`
    /// where the target has it and `hidden % 16 == 0`, else `dense_gemm_bf16`.
    pub router_gemm: KernelHandle,
    /// 2026-10-03: `moe_topk_softmax_rows` (the per-row router's top-k, lower expert on a tie).
    pub topk_rows: KernelHandle,
    /// 2026-10-03: `moe_topk_softmax_batched` (the batched router's top-k).
    pub topk_batched: KernelHandle,
    /// 2026-10-03: `moe_fp8_grouped_sort`.
    pub sort: KernelHandle,
    /// 2026-10-03: The W8A16 grouped gate+up and down `fp8_grouped_expert_kernels` picks
    /// (tensor-core or scalar) and their launch shapes.
    pub gate_up: KernelHandle,
    pub gate_up_geometry: ops::Fp8GroupedGeometry,
    pub down: KernelHandle,
    pub down_geometry: ops::Fp8GroupedGeometry,
    /// 2026-10-03: The W8A8 expert step's three kernels.
    pub quant_w8a8: KernelHandle,
    pub gate_up_w8a8: KernelHandle,
    pub down_w8a8: KernelHandle,
    /// 2026-10-03: `moe_weighted_sum_blend_fp8_grouped`.
    pub blend: KernelHandle,
}

/// 2026-10-03: The shape and routing facts the bound kernels read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeFacts {
    pub num_experts: u32,
    pub top_k: u32,
    pub hidden: u32,
    pub inter: u32,
    /// 2026-10-03: `config.norm_topk_prob`, the top-k kernels' renormalisation.
    pub norm_topk_prob: bool,
    /// 2026-10-03: The grouped decode runs the tensor-core expert kernels
    /// (`MoeLayer::fp8_grouped_tc_on`): every row count 1..=256, one row included.
    pub tensor_core: bool,
    /// 2026-10-03: The grouped decode runs the W8A8 expert step
    /// (`MoeLayer::fp8_grouped_tc_w8a8_on`).
    pub w8a8: bool,
}

/// 2026-10-03: One layer's MoE FFN, bound.
#[derive(Debug, Clone)]
pub struct MoeBinding {
    /// 2026-10-03: The BF16 router `[experts, hidden]`.
    pub router: DenseWeight,
    /// 2026-10-03: The BF16 shared-expert gate `[1, hidden]`.
    pub shared_gate: DenseWeight,
    pub gate: Fp8Tables,
    pub up: Fp8Tables,
    pub down: Fp8Tables,
    pub shared: Fp8ExpertWeight,
    pub facts: MoeFacts,
    pub kernels: MoeKernels,
}

/// 2026-10-03: The arena buffers the grouped decode borrows, as the circuit sees them: the sort
/// outputs live in `gate_logits` (legacy reuses it after top-k), and the byte capacities of
/// every buffer `MoeLayer::fp8_grouped_decode_arena_ok` checks, so a plan is refused at a
/// width legacy would decline.
#[derive(Debug, Clone, Copy)]
pub struct MoeScratch {
    /// 2026-10-03: Where the sort writes (`BufferArena::gate_logits`).
    pub sort: DevicePtr,
    pub scratch_bytes: usize,
    pub gate_logits_bytes: usize,
    pub expert_gate_out_bytes: usize,
    pub expert_down_out_bytes: usize,
    pub logits_bytes: usize,
    pub attn_output_bytes: usize,
    pub moe_output_bytes: usize,
}

impl MoeScratch {
    /// 2026-10-03: The scratch of the model's arena.
    pub fn from_arena(b: &BufferArena) -> Self {
        MoeScratch {
            sort: b.gate_logits(),
            scratch_bytes: b.scratch_bytes(),
            gate_logits_bytes: b.gate_logits_bytes(),
            expert_gate_out_bytes: b.expert_gate_out_bytes(),
            expert_down_out_bytes: b.expert_down_out_bytes(),
            logits_bytes: b.logits_bytes(),
            attn_output_bytes: b.attn_output_bytes(),
            moe_output_bytes: b.moe_output_bytes(),
        }
    }
}

impl MoeBinding {
    /// 2026-10-03: Where the sort writes for `m` rows in `scratch`, and the active-expert
    /// capacity (`grouped_sort_out`, legacy's layout).
    pub fn sort_out(&self, scratch: &MoeScratch, m: u32) -> (ops::Fp8GroupedSortOut, u32) {
        grouped_sort_out(scratch.sort, m, self.facts.top_k, self.facts.num_experts)
    }
}

impl MoeFacts {
    /// 2026-10-03: Refuse `m` rows where legacy's grouped decode would decline them: outside
    /// its row envelope, or past an arena buffer it checks. Under the W8A8 step, also where
    /// `Fp8GroupedW8a8Layout::new` would fail.
    pub fn check_rows(&self, m: u64, scratch: &MoeScratch) -> anyhow::Result<()> {
        let f = self;
        let m = usize::try_from(m)?;
        anyhow::ensure!(
            fp8_grouped_decode_rows_ok(m, f.tensor_core),
            "the grouped MoE decode does not take {m} rows with the {} expert kernels",
            if f.tensor_core {
                "tensor-core"
            } else {
                "scalar"
            }
        );
        let need = grouped_decode_buffer_need(
            m,
            f.hidden as usize,
            f.inter as usize,
            f.num_experts as usize,
            f.top_k as usize,
        );
        let s = scratch;
        for (what, need, have) in [
            ("scratch", need.scratch, s.scratch_bytes),
            ("gate_logits", need.gate_logits, s.gate_logits_bytes),
            (
                "expert_gate_out",
                need.expert_gate_out,
                s.expert_gate_out_bytes,
            ),
            (
                "expert_down_out",
                need.expert_down_out,
                s.expert_down_out_bytes,
            ),
            ("logits", need.shared_act, s.logits_bytes),
            ("attn_output", need.row_hidden, s.attn_output_bytes),
            ("moe_output", need.row_hidden, s.moe_output_bytes),
        ] {
            anyhow::ensure!(
                need <= have,
                "at {m} rows the grouped MoE decode needs {need} bytes of `{what}`, the arena \
                 holds {have}; legacy declines this width"
            );
        }
        if f.w8a8 {
            ops::Fp8GroupedW8a8Layout::new(
                m,
                f.top_k as usize,
                f.hidden as usize,
                f.inter as usize,
                s.expert_gate_out_bytes,
                s.logits_bytes,
            )?;
        }
        Ok(())
    }
}

impl MoeLayer {
    /// 2026-10-03: This layer for the circuit executor, or `None` with the reasons in
    /// `unmodelled`. `levers` and `config` are the model's, as its decode reads them.
    pub fn circuit_bind(
        &self,
        config: &metrale_config::ModelConfig,
        levers: &ops::ModelLevers,
        unmodelled: &mut Vec<String>,
    ) -> Option<MoeBinding> {
        let (h, inter) = (config.hidden_size, config.moe_intermediate_size);
        let arms = [
            (self.lora.is_some(), "a MoE LoRA adapter"),
            (self.pre_expert_norm.is_some(), "a pre-expert norm"),
            (self.weights.router_pre_norm.is_some(), "a router pre-norm"),
            (self.tid2eid_dev.is_some(), "hash routing"),
            (
                self.correction_bias_dev.is_some(),
                "a router correction bias",
            ),
            (self.gate_nvfp4.is_some(), "an NVFP4 router"),
            (
                self.router_logits_n as usize != config.num_experts,
                "zero-computation experts",
            ),
            (
                self.bf16_gate_weight_ptrs.is_some() || self.bf16_shared_expert.is_some(),
                "BF16 experts or a BF16 shared expert",
            ),
            (
                self.fp8_gate_weight_ptrs.is_none()
                    || self.fp8_up_weight_ptrs.is_none()
                    || self.fp8_down_weight_ptrs.is_none()
                    || self.fp8_shared_expert.is_none(),
                "experts that are not FP8 (NVFP4 experts are not bound yet)",
            ),
            (
                crate::layers::expert_quantization().nvfp4_decode(),
                "an NVFP4 --expert-quantization tier",
            ),
            (
                self.is_dflash_capture_layer && levers.frankenstein_decode_via_prefill,
                "a DFlash capture layer routed through prefill \
                 (METRALE_FRANKENSTEIN_DECODE_VIA_PREFILL)",
            ),
            (
                self.fp32_routing_active(levers),
                "FP32 routing (METRALE_FP32_ROUTING)",
            ),
            (levers.fp32_gate, "an FP32 router GEMM (METRALE_FP32_GATE)"),
            (config.ep_world_size > 1, "expert parallelism"),
            (
                !super::forward_fp8_grouped_decode::fp8_grouped_decode_enabled(),
                "the grouped MoE decode off (METRALE_NO_FP8_MOE_GROUPED_DECODE)",
            ),
            (
                super::dump::enabled(),
                "expert-id dumps (METRALE_DUMP_EXPERT_IDS)",
            ),
            (
                !crate::layers::row_invariant(),
                "the by-rows tier policy (its multi-row MoE arms are not bound)",
            ),
            (
                self.weights.shared_expert_gate.weight.is_null(),
                "an ungated shared expert",
            ),
            (
                !self.fp8_grouped_tc_on(h, inter),
                "the grouped MoE decode off its tensor-core expert kernels (METRALE_NO_MOE_FP8_TC, \
                 or the kernels or shapes refused): the scalar arms are not bound",
            ),
            (
                config.num_experts > ops::FP8_GROUPED_SORT_MAX_EXPERTS as usize,
                "more experts than the grouped sort takes",
            ),
            (
                !super::forward_fp8_grouped_decode::fp8_grouped_decode_dims_ok(
                    h as u32,
                    inter as u32,
                ) || !h.is_multiple_of(8),
                "a hidden or expert width the grouped decode refuses",
            ),
        ];
        let before = unmodelled.len();
        for (present, what) in arms {
            if present {
                unmodelled.push(what.to_string());
            }
        }
        let kernels = [
            (self.router_gemv_batchm_k, "dense_gemv_bf16_batchm"),
            (self.moe_topk_softmax_rows_k, "moe_topk_softmax_rows"),
            (self.moe_fp8_grouped_sort_k, "moe_fp8_grouped_sort"),
            (
                self.moe_weighted_sum_blend_fp8_grouped_k,
                "moe_weighted_sum_blend_fp8_grouped",
            ),
            (
                self.moe_expert_gate_up_act_fp8_grouped_k,
                "moe_expert_gate_up_act_fp8_grouped",
            ),
            (
                self.moe_expert_down_act_fp8_grouped_k,
                "moe_expert_down_act_fp8_grouped",
            ),
        ];
        for (k, name) in kernels {
            if k.0 == 0 {
                unmodelled.push(format!("the grouped MoE decode without `{name}`"));
            }
        }
        if unmodelled.len() > before {
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
        let experts = self.fp8_grouped_expert_kernels(h, inter);
        let router_gemm = if self.moe_router_gemm_k.0 != 0 && (h as u32).is_multiple_of(16) {
            self.moe_router_gemm_k
        } else {
            self.dense_gemm
        };
        let tc = &self.fp8_grouped_tc;
        Some(MoeBinding {
            router: self.weights.gate,
            shared_gate: self.weights.shared_expert_gate,
            gate: tables(gp),
            up: tables(up),
            down: tables(dp),
            shared: sh,
            facts: MoeFacts {
                num_experts: config.num_experts as u32,
                top_k: config.num_experts_per_tok as u32,
                hidden: h as u32,
                inter: inter as u32,
                norm_topk_prob: config.norm_topk_prob,
                tensor_core: self.fp8_grouped_tc_on(h, inter),
                w8a8: self.fp8_grouped_tc_w8a8_on(h, inter),
            },
            kernels: MoeKernels {
                router_rows: self.router_gemv_batchm_k,
                router_gemm,
                topk_rows: self.moe_topk_softmax_rows_k,
                topk_batched: self.moe_topk_batched,
                sort: self.moe_fp8_grouped_sort_k,
                gate_up: experts.gate_up,
                gate_up_geometry: experts.gate_up_geometry,
                down: experts.down,
                down_geometry: experts.down_geometry,
                quant_w8a8: tc.quant_w8a8,
                gate_up_w8a8: tc.gate_up_w8a8,
                down_w8a8: tc.down_w8a8,
                blend: self.moe_weighted_sum_blend_fp8_grouped_k,
            },
        })
    }
}

#[cfg(test)]
#[path = "circuit_tests.rs"]
mod tests;
