// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: A tool-call opener that also closes the thinking block
//! (`SchedLimits::thinking_closed_by`). Under a reasoning format whose tool
//! call ends the reasoning (`ReasoningParser::tool_call_closes_reasoning`,
//! GLM-4.7's), a `<tool_call>` written while thinking ends the thinking there,
//! and the opener is the first content token. Without this the sequence stays
//! in thinking through the call: the EOS the model writes after it (GLM's
//! `<|observation|>`) is held back as a thinking EOS, the call is never marked
//! complete and the grammar never sees it, so the model runs on past its call.
//!
//! Both token paths call [`close_thinking_at_tool_call`] before they read
//! `inside_thinking` for the token: `decode_logits_step` and `emit_token`.
//!
//! Owner: scheduler.
//! Invariants:
//! - With `closed_by == None` nothing changes, so every other model's state
//!   transitions are untouched.

use super::types::ActiveSeq;

/// 2026-10-09: When `tok` is `closed_by` and the sequence is thinking, leave
/// thinking with the state changes of a `</think>` the model wrote, minus the
/// one-shot `think_just_ended` (the opener itself is the first content token).
/// Returns whether it closed.
pub(super) fn close_thinking_at_tool_call(
    a: &mut ActiveSeq,
    tok: u32,
    closed_by: Option<u32>,
) -> bool {
    if !a.inside_thinking || closed_by != Some(tok) {
        return false;
    }
    a.inside_thinking = false;
    a.force_end_thinking = false;
    a.sentence_defer_count = 0;
    a.consecutive_confident = 0;
    a.in_code_fence = false;
    a.think_ended = true;
    a.think_just_ended = false;
    tracing::debug!(
        thinking_tokens = a.thinking_tokens,
        "tool-call opener closed the thinking block"
    );
    true
}

#[cfg(test)]
#[path = "think_close_tests.rs"]
mod tests;
