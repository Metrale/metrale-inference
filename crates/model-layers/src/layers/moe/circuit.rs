// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The MoE FFN as the circuit executor binds it: the router, the routed and shared
//! experts' tables, the shared-expert gate, the kernels this layer's own dispatch would launch
//! (the emitters refuse a plan whose kernel differs), and the arena scratch the grouped decode
//! sorts its slots into, which is no circuit edge.
//!
//! Bound arms: the grouped FP8 decode with the per-row router (`GroupedRouting::PerRow`,
//! `forward_fp8_grouped_router.rs`), which the target model takes at every row count under the
//! tensor-core expert kernels (`fp8_grouped_tc.rs`, one row through `MoeLayer::forward`) and
//! under a row-invariant tier policy, with its W8A8 expert step when the serve published FP8
//! expert activations (`fp8_grouped_tc_w8a8.rs`); and the batched router
//! (`GroupedRouting::Batched`) the MTP drafter's n-row propose runs. 2026-10-05: the grouped
//! NVFP4 decode of a checkpoint's declared NVFP4 experts on the lean or row-major tensor-core
//! pair, and BF16 experts (an MTP drafter excluded from quantization): one row through
//! `MoeLayer::forward`'s fused BF16 kernels, more through the grouped BF16 point
//! (`circuit_formats.rs`).
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
use crate::weight_map::{DenseWeight, Fp8ExpertWeight, QuantizedWeight};

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
    /// (tensor-core or scalar) and their launch shapes. 2026-10-05: For NVFP4 experts the pair
    /// `Nvfp4GroupedKernels::select` picks (lean or row-major tensor-core), for BF16 experts the
    /// grouped BF16 tensor-core pair.
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
    /// 2026-10-05: `MoeLayer::forward`'s one-row path over BF16 experts: the router GEMV
    /// (`dense_gemv_bf16`), top-k (`moe_topk_softmax`), the two fused expert kernels and the
    /// blend (`moe_weighted_sum_blend`).
    pub router_gemv: KernelHandle,
    pub topk_one_row: KernelHandle,
    pub fused_gate_up_bf16: KernelHandle,
    pub fused_down_bf16: KernelHandle,
    pub blend_one_row: KernelHandle,
}

/// 2026-10-05: The weight format of a layer's routed and shared experts, and for NVFP4 which
/// layout its tables are in, which selects the expert kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertKind {
    /// 2026-10-05: FP8 block-scaled (`set_fp8_experts`).
    Fp8,
    /// 2026-10-05: The checkpoint's declared NVFP4, repacked in place for the lean pair
    /// (`nvfp4_lean.rs`).
    Nvfp4Lean,
    /// 2026-10-05: The checkpoint's declared NVFP4 in the row-major layout.
    Nvfp4TensorCore,
    /// 2026-10-05: BF16 (`set_bf16_experts`).
    Bf16,
}

impl ExpertKind {
    /// 2026-10-05: The `moe_nvfp4_kernels` policy value this layer states.
    pub fn nvfp4_kernels(self) -> &'static str {
        match self {
            ExpertKind::Nvfp4Lean => "lean",
            ExpertKind::Nvfp4TensorCore => "tensor_core",
            ExpertKind::Fp8 | ExpertKind::Bf16 => "none",
        }
    }

    /// 2026-10-05: The `draft_moe_experts` policy value a drafter with these experts states.
    pub fn draft_name(self) -> &'static str {
        match self {
            ExpertKind::Fp8 => "fp8",
            ExpertKind::Nvfp4Lean | ExpertKind::Nvfp4TensorCore => "nvfp4",
            ExpertKind::Bf16 => "bf16",
        }
    }

    /// 2026-10-05: A drafter's propose of two or more rows routes batched
    /// (`GroupedRouting::Batched`) only through the grouped FP8 decode; the grouped NVFP4 and BF16
    /// decodes route per row.
    pub fn draft_routes_batched(self) -> bool {
        self == ExpertKind::Fp8
    }
}

/// 2026-10-05: The routed experts' per-expert pointer tables and the shared expert, by format.
#[derive(Debug, Clone, Copy)]
pub enum MoeExperts {
    Fp8 {
        gate: Fp8Tables,
        up: Fp8Tables,
        down: Fp8Tables,
        shared: Fp8ExpertWeight,
    },
    Nvfp4 {
        gate: ops::Nvfp4ExpertTables,
        up: ops::Nvfp4ExpertTables,
        down: ops::Nvfp4ExpertTables,
        shared: [QuantizedWeight; 3],
    },
    /// 2026-10-05: Device tables of each expert's BF16 weight pointer; the shared expert's
    /// gate, up and down.
    Bf16 {
        gate: DevicePtr,
        up: DevicePtr,
        down: DevicePtr,
        shared: [DenseWeight; 3],
    },
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
    /// 2026-10-05: The experts' format and layout.
    pub kind: ExpertKind,
}

/// 2026-10-03: One layer's MoE FFN, bound.
#[derive(Debug, Clone)]
pub struct MoeBinding {
    /// 2026-10-03: The BF16 router `[experts, hidden]`.
    pub router: DenseWeight,
    /// 2026-10-03: The BF16 shared-expert gate `[1, hidden]`.
    pub shared_gate: DenseWeight,
    pub experts: MoeExperts,
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
    /// 2026-10-05: Where the grouped NVFP4 and BF16 gate+up write the routed and the shared SiLU
    /// products for the down kernel (`expert_gate_out`, `logits`), as legacy lays them out.
    pub routed_act: DevicePtr,
    pub shared_act: DevicePtr,
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
            routed_act: b.expert_gate_out(),
            shared_act: b.logits(),
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
        if f.kind != ExpertKind::Fp8 {
            anyhow::ensure!(
                (1..=super::NVFP4_GROUPED_DECODE_TC_MAX_ROWS).contains(&m),
                "the grouped {:?} decode does not take {m} rows",
                f.kind
            );
            return f.check_arena(m, scratch);
        }
        anyhow::ensure!(
            fp8_grouped_decode_rows_ok(m, f.tensor_core),
            "the grouped MoE decode does not take {m} rows with the {} expert kernels",
            if f.tensor_core {
                "tensor-core"
            } else {
                "scalar"
            }
        );
        f.check_arena(m, scratch)?;
        if f.w8a8 {
            ops::Fp8GroupedW8a8Layout::new(
                m,
                f.top_k as usize,
                f.hidden as usize,
                f.inter as usize,
                scratch.expert_gate_out_bytes,
                scratch.logits_bytes,
            )?;
        }
        Ok(())
    }

    /// 2026-10-05: The arena capacities every grouped decode checks for `m` rows
    /// (`grouped_decode_buffer_need`).
    fn check_arena(&self, m: usize, scratch: &MoeScratch) -> anyhow::Result<()> {
        let f = self;
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
        Ok(())
    }
}

impl MoeLayer {
    /// 2026-10-03: This layer for the circuit executor, or `None` with the reasons in
    /// `unmodelled`. `levers` and `config` are the model's, as its decode reads them.
    /// 2026-10-05: The format arms are `bind_fp8`, `bind_nvfp4` and `bind_bf16`; this checks the
    /// features every arm shares.
    pub fn circuit_bind(
        &self,
        config: &metrale_config::ModelConfig,
        levers: &ops::ModelLevers,
        unmodelled: &mut Vec<String>,
    ) -> Option<MoeBinding> {
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
                config.num_experts > ops::FP8_GROUPED_SORT_MAX_EXPERTS as usize,
                "more experts than the grouped sort takes",
            ),
            (
                !config.hidden_size.is_multiple_of(8),
                "a hidden width the grouped decode refuses",
            ),
        ];
        let before = unmodelled.len();
        for (present, what) in arms {
            if present {
                unmodelled.push(what.to_string());
            }
        }
        let common = [
            (self.router_gemv_batchm_k, "dense_gemv_bf16_batchm"),
            (self.moe_topk_softmax_rows_k, "moe_topk_softmax_rows"),
            (self.moe_fp8_grouped_sort_k, "moe_fp8_grouped_sort"),
            (
                self.moe_weighted_sum_blend_fp8_grouped_k,
                "moe_weighted_sum_blend_fp8_grouped",
            ),
        ];
        for (k, name) in common {
            if k.0 == 0 {
                unmodelled.push(format!("the grouped MoE decode without `{name}`"));
            }
        }
        let bound = if self.nvfp4_grouped.declared_experts {
            self.bind_nvfp4(config, unmodelled)
        } else if self.fp8_gate_weight_ptrs.is_some() {
            self.bind_fp8(config, unmodelled)
        } else if self.bf16_gate_weight_ptrs.is_some() {
            self.bind_bf16(config, unmodelled)
        } else {
            unmodelled.push(
                "experts in no bound format (the per-expert NVFP4 decode of a checkpoint not \
                 served at its declared formats)"
                    .to_string(),
            );
            None
        };
        if unmodelled.len() > before {
            return None;
        }
        let (experts, kind, pair) = bound?;
        let router_gemm =
            if self.moe_router_gemm_k.0 != 0 && (config.hidden_size as u32).is_multiple_of(16) {
                self.moe_router_gemm_k
            } else {
                self.dense_gemm
            };
        let (h, inter) = (config.hidden_size, config.moe_intermediate_size);
        let tc = &self.fp8_grouped_tc;
        let fp8 = kind == ExpertKind::Fp8;
        Some(MoeBinding {
            router: self.weights.gate,
            shared_gate: self.weights.shared_expert_gate,
            experts,
            facts: MoeFacts {
                num_experts: config.num_experts as u32,
                top_k: config.num_experts_per_tok as u32,
                hidden: h as u32,
                inter: inter as u32,
                norm_topk_prob: config.norm_topk_prob,
                tensor_core: !fp8 || self.fp8_grouped_tc_on(h, inter),
                w8a8: fp8 && self.fp8_grouped_tc_w8a8_on(h, inter),
                kind,
            },
            kernels: MoeKernels {
                router_rows: self.router_gemv_batchm_k,
                router_gemm,
                topk_rows: self.moe_topk_softmax_rows_k,
                topk_batched: self.moe_topk_batched,
                sort: self.moe_fp8_grouped_sort_k,
                gate_up: pair.gate_up,
                gate_up_geometry: pair.gate_up_geometry,
                down: pair.down,
                down_geometry: pair.down_geometry,
                quant_w8a8: tc.quant_w8a8,
                gate_up_w8a8: tc.gate_up_w8a8,
                down_w8a8: tc.down_w8a8,
                blend: self.moe_weighted_sum_blend_fp8_grouped_k,
                router_gemv: self.dense_gemv,
                topk_one_row: self.moe_topk,
                fused_gate_up_bf16: self.moe_expert_gate_up_shared_bf16_k,
                fused_down_bf16: self.moe_expert_silu_down_shared_bf16_k,
                blend_one_row: self.moe_weighted_sum_blend,
            },
        })
    }
}

#[cfg(test)]
#[path = "circuit_tests.rs"]
mod tests;
