// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: [`BufferSizes::total_bytes`], moved from `sizes.rs` unchanged to keep that
//! file under the 500-line cap.
//!
//! Owner: gpu-runtime.
//! Invariants: every size field of [`BufferSizes`] except `o_latent` and `norm_unit_w` is in
//! the sum.

use super::BufferSizes;

impl BufferSizes {
    /// 2026-09-25: The sum of the sizes, which preflight reserves for the arena.
    /// `o_latent` and `norm_unit_w` are not in it.
    pub fn total_bytes(&self) -> usize {
        self.hidden_states
            + self.residual
            + self.norm_output
            + self.qkv_output
            + self.attn_output
            + self.gate_logits
            + self.gate_logits_f32
            + self.moe_router_in_f32
            + self.moe_output
            + self.logits
            + self.ssm_qkvz
            + self.ssm_ba
            + self.ssm_deinterleaved
            + self.ssm_gates
            + self.ssm_conv_out_f32
            + self.scratch
            + self.expert_gate_out
            + self.expert_up_out
            + self.hc_lowrank_scratch
            + self.qsa_select_scratch
            + self.expert_down_out
            + self.splitk_workspace
            + self.gdn_fla_scratch
            + self.ssd_scratch
            + self.hc_streams
            + self.hc_post
            + self.hc_comb
            + self.token_ids
            + self.ffn_act_q8
            + self.ffn_act_a
            + self.ffn_gate_up_fused
            + self.ffn_act_scale
            + self.ffn_act_scale_kmajor
            + self.fp8_act
            + self.moe_fp8_scratch
            + self.fp8_act_scale
            + self.fp8_act_scale_kmajor
            + self.lora_xa
            + self.lora_delta
            + self.lora_hact
            + self.lora_seq_slot
            + self.q2_dequant_scratch
            + self.q2_act_q8
            + self.ssm_rowwise_w_bf16
    }
}
