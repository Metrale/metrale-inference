// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The end of `forward_prefill_fp8`: the routed rows' weighted reduce back to token
//! order, the EP all-reduce, and the shared expert's gated blend, into `ctx.buffers.moe_output()`.
//!
//! Owner: model-layers (MoE).
//! Invariants: the output bits do not depend on whether the fused `moe_unpermute_blend` or the
//! two kernels it replaces ran.

use super::*;

impl MoeLayer {
    /// 2026-09-28: `output = unpermute(expert_down_out)`, then with a shared expert
    /// `output += sigmoid(input . shared_gate) * shared_out` (shared_out in `attn_output`).
    /// One fused launch (`moe_unpermute_blend`) when it resolved, there is a shared expert, no
    /// EP all-reduce sits between the two steps and the expert-id dumps are off; else
    /// `moe_unpermute_reduce_indexed` and `moe_batched_blend`, as before.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn fp8_prefill_combine(
        &self,
        input: DevicePtr,
        expert_down_out: DevicePtr,
        output: DevicePtr,
        token_to_perm: DevicePtr,
        weights_dev: DevicePtr,
        has_shared: bool,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
        mt: &mut Option<std::time::Instant>,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let n = num_tokens as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        macro_rules! mprof {
            ($label:expr) => {
                mprof_step!(*mt, ctx, stream, n, $label)
            };
        }
        let ep = ctx.comm.is_some() && ctx.config.ep_world_size > 1;
        let shared_down_out = ctx.buffers.attn_output();
        if has_shared
            && !ep
            && self.moe_unpermute_blend_k.0 != 0
            && h.is_multiple_of(8)
            && !super::super::dump::enabled()
        {
            ops::moe_unpermute_blend(
                ctx.gpu,
                self.moe_unpermute_blend_k,
                expert_down_out,
                output,
                token_to_perm,
                weights_dev,
                shared_down_out,
                input,
                self.weights.shared_expert_gate.weight,
                h,
                n,
                top_k,
                stream,
            )?;
            mprof!("unpermute_blend");
            return Ok(());
        }
        ops::moe_unpermute_reduce_indexed(
            ctx.gpu,
            self.moe_unpermute_reduce,
            expert_down_out,
            output,
            token_to_perm,
            weights_dev,
            h,
            n,
            top_k,
            stream,
        )?;
        mprof!("unpermute_reduce");

        // 2026-09-25: With EP, the all-reduce covers only the routed output; the
        // shared blend below runs after it.
        if let Some(comm) = ctx.comm
            && ep
        {
            comm.all_reduce_async(output.0, num_tokens * h as usize * 2, stream)?;
        }

        if has_shared {
            super::super::dump::dump_routed_only(ctx.gpu, stream, output, n, h)?;
            super::super::dump::dump_shared_out(ctx.gpu, stream, shared_down_out, n, h)?;
            super::super::dump::dump_shared_gate(
                ctx.gpu,
                stream,
                input,
                self.weights.shared_expert_gate.weight,
                n,
                h,
            )?;
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
            mprof!("blend");
        }
        Ok(())
    }
}
