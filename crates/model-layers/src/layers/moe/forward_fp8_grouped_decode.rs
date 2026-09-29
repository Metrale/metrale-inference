// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `MoeLayer::forward_fp8_grouped_decode`: the FP8 MoE of `m` rows in
//! one routed+shared expert dispatch.
//!
//! Rows are grouped by expert with `moe_sort_by_expert`, and the grouped
//! kernels (`ops/fp8_moe_grouped.rs`) read each active expert's weights once
//! for all the rows routed to it. Callers: multi-sequence decode
//! (`qwen3_attention/trait_impl/multi_seq/ffn.rs`), multi-sequence and
//! batched SSM decode (`qwen3_ssm/trait_decode_multi_seq.rs`,
//! `qwen3_ssm/trait_decode_batched.rs`) and the MTP drafter
//! (`mtp_head/forward_batch_ffn.rs`).
//!
//! Given the same routing, the output equals a per-row loop of `forward`'s FP8
//! kernels bit for bit; `model-arch/examples/fp8_moe_grouped_decode_microtest.rs`
//! checks this for M = 2, 3, 4, 8, 16 and 32 with the routing fed to both.
//! The routing here is the batched gate GEMM and `moe_topk_*_batched`, which
//! may choose differently from the per-token GEMV and top-k on a near-tie.
//!
//! Owner: model-layers (MoE).
//! Invariants: `forward_fp8_grouped_decode` launches nothing unless
//! `fp8_grouped_decode_ok(m, ctx)` holds.

use super::*;

/// 2026-09-25: Widest row count the scalar expert kernels admit. Above it `forward_prefill`
/// takes its FP8 grouped GEMM when that kernel resolved.
pub const FP8_GROUPED_DECODE_MAX_ROWS: usize = 64;

/// 2026-09-29: Widest row count the tensor-core expert kernels admit: one verify of 128
/// sequences at one draft (`--max-batch-size 128`, k = 1). A row's bits do not depend on the
/// row count there (`fp8_grouped_tc.rs`), so every decode and verify width up to it keeps the
/// arithmetic of one row, where `forward_prefill` (W8A8) and the per-row verify loop took the
/// rows above 64 before.
pub const FP8_GROUPED_DECODE_TC_MAX_ROWS: usize = 256;

/// 2026-09-25: The grouped decode's kernels, looked up with `try_kernel`. A zero
/// handle among the first four declines the path in
/// [`MoeLayer::fp8_grouped_decode_arena_ok`]; a zero `topk_rows` or `router_gemv`
/// (2026-09-26) declines only the exact routings in
/// [`MoeLayer::fp8_grouped_routing_ok`].
pub(super) struct GroupedKernels {
    pub gate_up: KernelHandle,
    pub silu_down: KernelHandle,
    pub blend: KernelHandle,
    pub sort: KernelHandle,
    pub topk_rows: KernelHandle,
    pub router_gemv: KernelHandle,
}

impl GroupedKernels {
    /// 2026-09-25: One direct `try_kernel` call per kernel: `try_kernel` is
    /// `#[track_caller]`, so each lookup keeps its own line in the audit.
    pub(super) fn resolve(gpu: &dyn GpuBackend) -> Self {
        use super::super::try_kernel;
        const FUSED: &str = "moe_shared_expert_fused_fp8_grouped";
        Self {
            gate_up: try_kernel(gpu, FUSED, "moe_expert_gate_up_act_fp8_grouped"),
            silu_down: try_kernel(gpu, FUSED, "moe_expert_down_act_fp8_grouped"),
            blend: try_kernel(
                gpu,
                "moe_fp8_grouped_blend",
                "moe_weighted_sum_blend_fp8_grouped",
            ),
            sort: try_kernel(gpu, "moe_fp8_grouped_sort", "moe_fp8_grouped_sort"),
            topk_rows: try_kernel(gpu, "moe_topk", "moe_topk_softmax_rows"),
            router_gemv: try_kernel(gpu, "dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm"),
        }
    }
}

/// 2026-09-25: `METRALE_NO_FP8_MOE_GROUPED_DECODE`, set to any value, turns this
/// path off. Read once per process. It covers the MTP drafter, which is on by
/// default, and the target model's multi-row decode, which also needs the
/// `ModelLevers::moe_fp8_grouped_decode_target` lever
/// (`FfnComponent::fp8_grouped_decode_ok`). 2026-09-26: It also turns off the
/// exact routings (`GroupedRouting::PerRow` / `PerToken`), which are on by default.
fn fp8_grouped_decode_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METRALE_NO_FP8_MOE_GROUPED_DECODE").is_none())
}

/// 2026-09-25: Shape admission for the scalar grouped kernels, without a GPU:
/// `fp8_grouped_decode_rows_ok(m, false)` and `fp8_grouped_decode_dims_ok`.
pub fn fp8_grouped_decode_shape_ok(m: usize, hidden: u32, inter: u32) -> bool {
    fp8_grouped_decode_rows_ok(m, false) && fp8_grouped_decode_dims_ok(hidden, inter)
}

/// 2026-09-29: The row envelope of the grouped decode: `2..=FP8_GROUPED_DECODE_MAX_ROWS` for
/// the scalar expert kernels, `1..=FP8_GROUPED_DECODE_TC_MAX_ROWS` for the tensor-core ones
/// (`tensor_core`), whose one-row bits equal the batched bits.
pub fn fp8_grouped_decode_rows_ok(m: usize, tensor_core: bool) -> bool {
    if tensor_core {
        (1..=FP8_GROUPED_DECODE_TC_MAX_ROWS).contains(&m)
    } else {
        (2..=FP8_GROUPED_DECODE_MAX_ROWS).contains(&m)
    }
}

/// 2026-09-29: The width terms of the grouped kernels:
/// * `hidden % 16 == 0`: the gate/up kernel reads weights in 16-element chunks.
/// * `inter % 8 == 0`: the down kernel reads weights in 8-element chunks and the
///   SiLU product rows as `float4`.
pub fn fp8_grouped_decode_dims_ok(hidden: u32, inter: u32) -> bool {
    hidden >= 16 && hidden.is_multiple_of(16) && inter >= 8 && inter.is_multiple_of(8)
}

/// 2026-09-25: Bytes this path needs in each arena buffer it borrows, so a batch
/// wider than the arena is refused rather than written past a buffer's end.
pub(crate) struct GroupedDecodeBufferNeed {
    pub scratch: usize,
    pub gate_logits: usize,
    pub expert_gate_out: usize,
    pub expert_down_out: usize,
    pub shared_act: usize,
    pub row_hidden: usize,
}

pub(crate) fn grouped_decode_buffer_need(
    m: usize,
    hidden: usize,
    inter: usize,
    num_experts: usize,
    top_k: usize,
) -> GroupedDecodeBufferNeed {
    let te = m * top_k;
    GroupedDecodeBufferNeed {
        // 2026-09-25: indices `[te]` u32 + weights `[te]` f32.
        scratch: 2 * te * 4,
        // 2026-09-25: max(gate logits `[m, E]` BF16, sort scratch:
        // sorted_token_ids `[te]` + sorted_expert_ids `[te]` + expert_offsets
        // `[E+1]` + token_to_perm `[te]` + active_experts `[cap]` +
        // active_count `[1]`), with `cap = min(te, E)`.
        gate_logits: (m * num_experts * 2)
            .max(te * 4 * 3 + (num_experts + 1) * 4 + (te.min(num_experts) + 1) * 4),
        // 2026-09-26: The routed SiLU product `[te, inter]` FP32.
        expert_gate_out: te * inter * 4,
        expert_down_out: te * hidden * 2,
        // 2026-09-26: The shared-expert SiLU product `[m, inter]` FP32, in `logits()`.
        shared_act: m * inter * 4,
        // 2026-09-25: Shared-expert down output and `moe_output`, `[m, hidden]` BF16.
        row_hidden: m * hidden * 2,
    }
}

impl MoeLayer {
    /// 2026-09-25: The weight, kernel, shape and arena terms of
    /// [`Self::fp8_grouped_decode_ok`], decidable without a `ForwardContext`,
    /// so the MTP batch cap (`mtp_head/batch_caps.rs`) can size a batch. Not
    /// sufficient alone: the kill switch, the FP32-routing lever and expert
    /// parallelism are checked only by the full predicate.
    pub fn fp8_grouped_decode_arena_ok(
        &self,
        m: usize,
        cfg: &metrale_config::ModelConfig,
        b: &metrale_gpu_runtime::buffers::BufferArena,
    ) -> bool {
        let h = cfg.hidden_size;
        let inter = cfg.moe_intermediate_size;
        let need =
            grouped_decode_buffer_need(m, h, inter, cfg.num_experts, cfg.num_experts_per_tok);
        // 2026-09-28: One row too under the tensor-core expert kernels, whose row bits do not
        // depend on the row count (`fp8_grouped_tc.rs`). 2026-09-29: And up to
        // `FP8_GROUPED_DECODE_TC_MAX_ROWS` rows.
        fp8_grouped_decode_rows_ok(m, self.fp8_grouped_tc_on(h, inter))
            && fp8_grouped_decode_dims_ok(h as u32, inter as u32)
            && self.fp8_gate_weight_ptrs.is_some()
            && self.fp8_up_weight_ptrs.is_some()
            && self.fp8_down_weight_ptrs.is_some()
            && self.fp8_shared_expert.is_some()
            && self.moe_expert_gate_up_act_fp8_grouped_k.0 != 0
            && self.moe_expert_down_act_fp8_grouped_k.0 != 0
            && self.moe_weighted_sum_blend_fp8_grouped_k.0 != 0
            && self.moe_fp8_grouped_sort_k.0 != 0
            && cfg.num_experts <= ops::FP8_GROUPED_SORT_MAX_EXPERTS as usize
            // 2026-09-25: This path has no LoRA fold hooks.
            && self.lora.is_none()
            && self.pre_expert_norm.is_none()
            && self.tid2eid_dev.is_none()
            && !self.has_mixed_bf16_shared_expert()
            // 2026-09-25: Zero-expert routers are not served here.
            && self.router_logits_n as usize == cfg.num_experts
            // 2026-09-25: The bias arm below runs `moe_topk_sigmoid_batched`, so a
            // sqrtsoftplus router is refused.
            && !(self.correction_bias_dev.is_some() && cfg.scoring_func == "sqrtsoftplus")
            && b.scratch_bytes() >= need.scratch
            && b.gate_logits_bytes() >= need.gate_logits
            && b.expert_gate_out_bytes() >= need.expert_gate_out
            && b.expert_down_out_bytes() >= need.expert_down_out
            && b.logits_bytes() >= need.shared_act
            && b.attn_output_bytes() >= need.row_hidden
            && b.moe_output_bytes() >= need.row_hidden
    }

    /// 2026-09-25: Whether `forward_fp8_grouped_decode` serves `m` rows on this
    /// layer under `ctx`: [`Self::fp8_grouped_decode_arena_ok`], the kill
    /// switch, FP32 routing off and no expert parallelism.
    pub fn fp8_grouped_decode_ok(&self, m: usize, ctx: &ForwardContext) -> bool {
        let cfg = ctx.config;
        fp8_grouped_decode_enabled()
            && self.fp8_grouped_decode_arena_ok(m, cfg, ctx.buffers)
            && !self.fp32_routing_active(ctx.levers)
            && !(ctx.comm.is_some() && cfg.ep_world_size > 1)
    }

    /// 2026-09-25: The FP8 MoE of `m` rows: `input` is `[m, H]` BF16, the output
    /// lands in rows `0..m` of `moe_output()`. Returns an error, launching
    /// nothing, when `fp8_grouped_decode_ok(m, ctx)` is false. Routes with
    /// [`GroupedRouting::Batched`].
    pub fn forward_fp8_grouped_decode(
        &self,
        input: DevicePtr,
        m: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_fp8_grouped_decode_routed(input, m, GroupedRouting::Batched, ctx, stream)
    }

    /// 2026-09-26: [`Self::forward_fp8_grouped_decode`] with the router arithmetic
    /// `routing` names. Returns an error, launching nothing, when
    /// `fp8_grouped_routing_ok(m, routing, ctx)` is false.
    pub fn forward_fp8_grouped_decode_routed(
        &self,
        input: DevicePtr,
        m: usize,
        routing: GroupedRouting,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            self.fp8_grouped_routing_ok(m, routing, ctx),
            "forward_fp8_grouped_decode: predicate false for m={m}, {routing:?} (caller must gate on it)"
        );
        let (Some(gp), Some(up), Some(dp), Some(sh)) = (
            &self.fp8_gate_weight_ptrs,
            &self.fp8_up_weight_ptrs,
            &self.fp8_down_weight_ptrs,
            &self.fp8_shared_expert,
        ) else {
            unreachable!("fp8_grouped_decode_ok checked the FP8 tables");
        };
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let n = m as u32;
        let te = m * top_k as usize;

        let log_key = match routing {
            GroupedRouting::Batched => "log:moe_fp8_grouped_decode",
            GroupedRouting::PerRow => "log:moe_fp8_grouped_decode_per_row",
            GroupedRouting::PerToken => "log:moe_fp8_grouped_decode_per_token",
        };
        if ctx.stats.once(log_key) {
            let family = if self.fp8_grouped_tc_on(h as usize, inter as usize) {
                "tensor-core"
            } else {
                "scalar"
            };
            tracing::info!(
                "MoE FP8 grouped decode: cross-row batched routed+shared expert dispatch \
                 active (first use M={m}, top_k={top_k}, experts={num_experts}, \
                 routing={routing:?}, expert kernels {family}; one-time log)"
            );
        }

        // 2026-09-25: 1-2. Router and top-k: indices `[m*top_k]` u32, weights
        // `[m*top_k]` f32, with the arithmetic `routing` names.
        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(te * 4);
        self.grouped_route(input, m, routing, indices_dev, weights_dev, ctx, stream)?;
        let gate_logits = ctx.buffers.gate_logits();

        // 2026-09-25: 3. Group slots by expert. Top-k has read `gate_logits`
        //    earlier on the same stream, so the sort scratch reuses it.
        let ne = num_experts as usize;
        let sorted_token_ids = gate_logits;
        let sorted_expert_ids = gate_logits.offset(te * 4);
        let expert_offsets = gate_logits.offset(te * 4 * 2);
        let token_to_perm = gate_logits.offset(te * 4 * 2 + (ne + 1) * 4);
        // 2026-09-25: 4. The active experts go to a list of fixed capacity `cap`
        //    (a function of `m`, so the grids are the same for a captured graph),
        //    then gate+up, silu+down and blend. 2026-09-27: One launch sorts the
        //    slots and builds the list.
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
        super::dump::dump_grouped_active(ctx.gpu, stream, active_count, m, ctx.graph_capture)?;
        // 2026-09-26: The SiLU products are FP32: routed `[te, inter]` in
        // `expert_gate_out()`, shared `[m, inter]` in `logits()`
        // (`grouped_decode_buffer_need`); the W8A8 step keeps its E4M3 products there too.
        let experts = self.fp8_grouped_expert_kernels(h as usize, inter as usize);
        let act = ctx.buffers.expert_gate_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        let shared_act = ctx.buffers.logits();
        let shared_out = ctx.buffers.attn_output();
        let output = ctx.buffers.moe_output();

        // 2026-09-28: The opt-in W8A8 expert step (`fp8_grouped_tc_w8a8.rs`).
        if self.fp8_grouped_tc_w8a8_on(h as usize, inter as usize) {
            let io = super::fp8_grouped_tc_w8a8::GroupedExpertIo {
                input,
                act,
                shared_act,
                expert_down_out,
                shared_out,
                rows: ops::Fp8GroupedW8a8Rows {
                    expert_offsets,
                    sorted_token_ids,
                    active_experts,
                    active_count,
                    cap,
                    num_tokens: n,
                },
            };
            self.run_fp8_grouped_w8a8(&io, ctx, stream)?;
        } else {
            ops::moe_expert_gate_up_act_fp8_grouped(
                ctx.gpu,
                experts.gate_up,
                experts.gate_up_geometry,
                input,
                gp.weight_ptrs,
                gp.scale_ptrs,
                up.weight_ptrs,
                up.scale_ptrs,
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
            ops::moe_expert_down_act_fp8_grouped(
                ctx.gpu,
                experts.down,
                experts.down_geometry,
                act,
                dp.weight_ptrs,
                dp.scale_ptrs,
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
        }
        ops::moe_weighted_sum_blend_fp8_grouped(
            ctx.gpu,
            self.moe_weighted_sum_blend_fp8_grouped_k,
            output,
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
#[path = "forward_fp8_grouped_decode_tests.rs"]
mod tests;
