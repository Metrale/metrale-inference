// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Token-major N-row MoE decode (`forward_token_major_decode`).
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::*;

impl MoeLayer {
    /// 2026-09-25: MoE for `num_tokens` decode rows with the token-major kernels of
    /// the `moe_prefill` module, without sorting by expert: batched gate GEMM and
    /// top-K, then gate+up, silu+down and weighted-sum+blend over every
    /// (row, expert-slot) pair.
    ///
    /// Serves untransposed NVFP4 experts with an NVFP4 shared expert, or (2026-10-13)
    /// with a BF16 shared expert run batched beside them. LoRA, BF16 or FP8 experts,
    /// the unified layout, or no shared expert go to `forward_batched`.
    pub fn forward_token_major_decode(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: These kernels do not handle zero-computation experts (router
        // width above num_experts), so refuse instead of mis-routing.
        anyhow::ensure!(
            self.router_logits_n as usize == ctx.config.num_experts,
            "zero-expert MoE routing is not wired on this dispatch variant yet (forward_token_major)"
        );
        self.refuse_lean_layout("MoeLayer::forward_token_major_decode")?;

        // 2026-09-25: The token-major kernels have no LoRA fold. With a MoE
        // adapter installed, `forward_batched` folds the router and expert deltas
        // per row. The test is on adapter presence, not on the per-step route,
        // so a captured graph holds no host branch on the route.
        if self.lora.is_some() {
            return self.forward_batched(input, num_tokens, ctx, stream);
        }
        // 2026-10-13: A BF16 shared expert (the checkpoint's declared format) takes the
        // token-major kernels' routed rows only and runs as one batched BF16 pass, where
        // it used to send every row through `forward_batched`'s per-token loop. Any
        // NVFP4 copy beside it is not launched, as in every other arm.
        let bf16_shared = self.has_mixed_bf16_shared_expert();
        let sh = &self.weights.shared_expert;
        let (sh_gate, sh_up, sh_down) = if bf16_shared {
            (
                QuantizedWeight::null(),
                QuantizedWeight::null(),
                QuantizedWeight::null(),
            )
        } else {
            (sh.gate_proj, sh.up_proj, sh.down_proj)
        };
        let has_shared = bf16_shared
            || (!sh.gate_proj.is_null() && !sh.up_proj.is_null() && !sh.down_proj.is_null());
        let nvfp4_supported = self.bf16_gate_weight_ptrs.is_none()
            && self.fp8_gate_weight_ptrs.is_none()
            && !self.use_t_layout_for_decode()
            && has_shared;
        if !nvfp4_supported {
            return self.forward_batched(input, num_tokens, ctx, stream);
        }

        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let n = num_tokens as u32;

        let router_in = self.router_input(input, n, h, ctx, stream)?;
        let gate_logits = ctx.buffers.gate_logits();
        if let Some(ref nvfp4) = self.gate_nvfp4 {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm,
                router_in,
                nvfp4,
                gate_logits,
                n,
                num_experts,
                h,
                stream,
            )?;
        } else {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm,
                router_in,
                &self.weights.gate,
                gate_logits,
                n,
                num_experts,
                h,
                stream,
            )?;
        }

        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(num_tokens * top_k as usize * 4);
        if let Some(bias) = self.correction_bias_dev {
            ops::moe_topk_sigmoid_batched(
                ctx.gpu,
                self.moe_topk_sigmoid_batched_k,
                gate_logits,
                bias,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                ctx.config.routed_scaling_factor as f32,
                n,
                stream,
            )?;
        } else {
            ops::moe_topk_softmax_batched(
                ctx.gpu,
                self.moe_topk_batched,
                gate_logits,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                n,
                stream,
            )?;
        }

        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        let shared_gate_scratch = ctx.buffers.logits();
        let shared_up_scratch = ctx.buffers.ssm_qkvz();
        let shared_down_out = ctx.buffers.attn_output();
        let output = ctx.buffers.moe_output();

        ops::moe_expert_gate_up_shared_prefill(
            ctx.gpu,
            self.moe_expert_gate_up_shared_token_major,
            input,
            self.gate_ptrs.packed_ptrs,
            self.gate_ptrs.scale_ptrs,
            self.gate_ptrs.scale2_vals,
            expert_gate_out,
            self.up_ptrs.packed_ptrs,
            self.up_ptrs.scale_ptrs,
            self.up_ptrs.scale2_vals,
            expert_up_out,
            indices_dev,
            &sh_gate,
            shared_gate_scratch,
            &sh_up,
            shared_up_scratch,
            inter,
            h,
            top_k,
            n,
            stream,
        )?;
        ops::moe_expert_silu_down_shared_prefill(
            ctx.gpu,
            self.moe_expert_silu_down_shared_token_major,
            expert_gate_out,
            expert_up_out,
            self.down_ptrs.packed_ptrs,
            self.down_ptrs.scale_ptrs,
            self.down_ptrs.scale2_vals,
            expert_down_out,
            indices_dev,
            shared_gate_scratch,
            shared_up_scratch,
            &sh_down,
            shared_down_out,
            h,
            inter,
            top_k,
            n,
            stream,
        )?;
        if bf16_shared {
            self.run_bf16_shared_expert(
                input,
                n,
                h,
                ctx.config.shared_expert_intermediate_size as u32,
                shared_gate_scratch,
                shared_up_scratch,
                shared_down_out,
                ctx,
                stream,
            )?;
        }

        let is_ep = ctx.comm.is_some() && ctx.config.ep_world_size > 1;
        let shared_for_blend = if is_ep {
            ctx.gpu
                .memset_async(expert_gate_out, 0, num_tokens * h as usize * 2, stream)?;
            expert_gate_out
        } else {
            shared_down_out
        };
        ops::moe_weighted_sum_blend_prefill(
            ctx.gpu,
            self.moe_weighted_sum_blend_token_major,
            output,
            expert_down_out,
            weights_dev,
            shared_for_blend,
            input,
            self.weights.shared_expert_gate.weight,
            h,
            top_k,
            h,
            n,
            stream,
        )?;

        if let Some(comm) = ctx.comm
            && ctx.config.ep_world_size > 1
        {
            if ctx.graph_capture {
                comm.all_reduce(output.0, num_tokens * h as usize * 2)?;
            } else {
                comm.all_reduce_async(output.0, num_tokens * h as usize * 2, stream)?;
            }
            if self.weights.shared_expert_gate.weight.0 == 0 {
                ops::residual_add(
                    ctx.gpu,
                    self.residual_add,
                    output,
                    shared_down_out,
                    n * h,
                    stream,
                )?;
            } else {
                ops::moe_batched_blend(
                    ctx.gpu,
                    self.moe_batched_blend,
                    output,
                    shared_down_out,
                    input,
                    self.weights.shared_expert_gate.weight,
                    h,
                    n,
                    stream,
                )?;
            }
        }

        Ok(())
    }
}
