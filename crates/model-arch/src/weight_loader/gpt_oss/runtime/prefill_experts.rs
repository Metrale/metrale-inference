// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Candidate token-grid MoE, retaining all scalar arithmetic stages.
use super::*;
impl GptOssLayer {
    pub(super) fn chunk_experts(
        &self,
        hidden: DevicePtr,
        tokens: u32,
        s: &PrefillScratch,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let gate = self.weights.gate_up;
        ops::gpt_oss_mxfp4_token_experts(
            gpu,
            s.expert_gemm,
            gate.blocks,
            gate.scales,
            s.norm,
            s.ids,
            s.gate_up,
            &ops::GptOssTokenExperts {
                tokens,
                rows: gate.rows,
                cols: gate.cols,
                per_slot_input: false,
            },
            stream,
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
        ops::gpt_oss_mxfp4_token_experts(
            gpu,
            s.expert_gemm,
            down.blocks,
            down.scales,
            s.activation,
            s.ids,
            s.selected,
            &ops::GptOssTokenExperts {
                tokens,
                rows: down.rows,
                cols: down.cols,
                per_slot_input: true,
            },
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
}
