// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Candidate token-grid MoE, retaining all scalar arithmetic stages.
use super::*;
impl GptOssLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn chunk_experts(
        &self,
        hidden: DevicePtr,
        tokens: u32,
        tc_rows: Option<u32>,
        s: &PrefillScratch,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let gate = self.weights.gate_up;
        Self::chunk_expert_projection(
            gpu, gate, s.norm, s.gate_up, tokens, tc_rows, false, s, stream,
        )?;
        ops::gpt_oss_token_expert_bias(
            gpu,
            s.expert_bias,
            s.gate_up,
            self.weights.gate_up_bias,
            s.ids,
            tokens,
            gate.rows,
            stream,
        )?;
        ops::gpt_oss_swiglu_bf16(
            gpu,
            self.kernels.activation,
            s.gate_up,
            s.activation,
            4 * tokens * 2880,
            stream,
        )?;
        let down = self.weights.down;
        Self::chunk_expert_projection(
            gpu,
            down,
            s.activation,
            s.selected,
            tokens,
            tc_rows,
            true,
            s,
            stream,
        )?;
        ops::gpt_oss_token_expert_bias(
            gpu,
            s.expert_bias,
            s.selected,
            self.weights.down_bias,
            s.ids,
            tokens,
            down.rows,
            stream,
        )?;
        ops::gpt_oss_expert_reduce_bf16(
            gpu,
            self.kernels.reduce,
            s.selected,
            s.scores,
            s.ids,
            s.moe,
            tokens,
            2880,
            stream,
        )?;
        ops::residual_add(
            gpu,
            self.kernels.residual,
            hidden,
            s.moe,
            tokens * 2880,
            stream,
        )
    }
    // 2026-10-07: Wider explicit chunks use cooperative validation; short tails retain token grids.
    pub(super) fn uses_expert_reuse(tokens: u32) -> bool {
        tokens >= 16
    }
    #[allow(clippy::too_many_arguments)]
    fn chunk_expert_projection(
        gpu: &dyn GpuBackend,
        weight: weights::Packed,
        input: DevicePtr,
        output: DevicePtr,
        tokens: u32,
        tc_rows: Option<u32>,
        per_slot_input: bool,
        s: &PrefillScratch,
        stream: u64,
    ) -> Result<()> {
        if let Some(max_rows) = tc_rows {
            ensure!(tokens == 128, "TC diagnostic requires full128 rows");
            return s
                .tc
                .as_ref()
                .context("TC diagnostic scratch missing")?
                .projection(gpu, weight, input, output, max_rows, per_slot_input, stream);
        }
        let g = ops::GptOssTokenExperts {
            tokens,
            rows: weight.rows,
            cols: weight.cols,
            per_slot_input,
        };
        if tokens > 16 {
            ops::gpt_oss_mxfp4_reuse_wide_experts(
                gpu,
                s.expert_reuse_wide,
                weight.blocks,
                weight.scales,
                input,
                s.ids,
                s.expert_plan,
                output,
                &g,
                stream,
            )
        } else if Self::uses_expert_reuse(tokens) {
            ops::gpt_oss_mxfp4_reuse_experts(
                gpu,
                s.expert_reuse,
                weight.blocks,
                weight.scales,
                input,
                s.ids,
                s.expert_plan,
                output,
                &g,
                stream,
            )
        } else {
            ops::gpt_oss_mxfp4_token_experts(
                gpu,
                s.expert_gemm,
                weight.blocks,
                weight.scales,
                input,
                s.ids,
                output,
                &g,
                stream,
            )
        }
    }
}
