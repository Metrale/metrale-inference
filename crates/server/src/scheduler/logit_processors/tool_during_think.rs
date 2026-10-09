// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Masks `<tool_call>` during thinking and biases it down when the tool-loop flag is set.
//!
//! - Inside thinking: the `<tool_call>` logit is set to `-inf`, unless the opener
//!   also closes the thinking block (`LogitsContext::thinking_closed_by`, GLM-4.7's
//!   format, 2026-10-09), where a call written while thinking is the model's way
//!   to end its reasoning.
//! - Outside thinking, when `seq.suppress_tool_call` is set (the API's tool-loop
//!   detector, `api/chat/loop_detect.rs`): the logit is lowered by 12.0. The
//!   bias is finite, so a strong enough logit can still win.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;

pub struct ToolCallDuringThinkingMask;

impl LogitsProcessor for ToolCallDuringThinkingMask {
    fn apply(
        &self,
        logits: &mut [f32],
        a: &mut ActiveSeq,
        ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        if a.inside_thinking {
            if let Some(tc_start) = ctx.tool_call_start_token
                && ctx.thinking_closed_by != Some(tc_start)
            {
                let idx = tc_start as usize;
                if idx < logits.len() {
                    logits[idx] = f32::NEG_INFINITY;
                }
            }
        } else if a.suppress_tool_call
            && let Some(tc_start) = ctx.tool_call_start_token
        {
            let idx = tc_start as usize;
            if idx < logits.len() {
                logits[idx] -= 12.0;
            }
        }
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "tool_call_during_thinking_mask"
    }
}
