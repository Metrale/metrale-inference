// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The end of the blocking chat path, split from `chat_blocking.rs`: usage
//! assembly, request accounting, the rate-limit refund, and the stop-sequence finish
//! reason correction.
//!
//! Owner: server chat API.
//! Invariants: none beyond the types.

use std::sync::Arc;

use crate::AppState;
use crate::ir;

/// 2026-09-26: Assemble usage, update the metrics, refund the unused rate-limit
/// reservation, and return the response IR; the caller encodes it for the wire.
#[allow(clippy::too_many_arguments)]
pub(super) fn finalize_response(
    state: Arc<AppState>,
    req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    pending: &crate::request_accounting::Pending,
    first_prefill: Option<std::time::Instant>,
    outcome: metrale_qci_dump::contract::usage::Outcome,
    all_choices: Vec<ir::Choice>,
    total_completion_tokens: usize,
    first_ttft: f64,
    last_decode_time_ms: f64,
    total_reasoning_tokens: u32,
    total_cached_prompt_tokens: u32,
    total_accepted_prediction_tokens: usize,
    prompt_len: usize,
) -> super::chat::ChatOutcome {
    let usage = ir::Usage {
        prompt_tokens: prompt_len,
        completion_tokens: total_completion_tokens,
        cached_prompt_tokens: total_cached_prompt_tokens as usize,
        reasoning_tokens: total_reasoning_tokens as usize,
        accepted_prediction_tokens: total_accepted_prediction_tokens,
        time_to_first_token_ms: first_ttft,
        // 2026-09-26: The last choice's decode window, the same one the rate below uses.
        decode_time_ms: last_decode_time_ms,
        response_tokens_per_second: ir::Usage::decode_rate_tok_s(
            total_completion_tokens,
            last_decode_time_ms,
        ),
    };

    // 2026-10-11: Token and latency counters and the usage record, in one place.
    // `first_prefill` is set by the first choice, which every finalized response has.
    let inference_usage = first_prefill.and_then(|prefill_started_at| {
        pending.complete(crate::request_accounting::Finish {
            outcome,
            completion: total_completion_tokens,
            cached_prompt: total_cached_prompt_tokens as usize,
            reasoning: total_reasoning_tokens as usize,
            prefill_started_at,
            ttft_ms: first_ttft,
            decode_ms: last_decode_time_ms,
        })
    });

    // 2026-09-26: The rate-limit middleware reserved `max_seq_len` tokens; refund what
    // prompt and completion did not use.
    if let Some(axum::extract::Extension(ref ctx)) = req_ctx {
        let actual = (prompt_len + total_completion_tokens) as u64;
        let refund = ctx.reserved_tokens.saturating_sub(actual);
        if refund > 0 {
            state.rate_limiter.refund_tokens(&ctx.identity, refund);
        }
    }

    super::chat::ChatOutcome::Blocking(Box::new(ir::ChatResponse {
        id: crate::ids::uuid_v4(),
        model: state.model_name.clone(),
        created: crate::ids::unix_timestamp(),
        choices: all_choices,
        usage,
        inference_usage,
    }))
}

/// 2026-09-26: A response ended by a client stop sequence reports `"stop"`, not
/// `"length"`. The blocking path finds stop strings afterwards (the caller's suffix
/// strip), so only `Length` is rewritten; `"timeout"` and `"tool_calls"` outrank a stop
/// match, as in `chat_stream::handle_done::resolve_wire_finish_reason`.
pub(super) fn stop_match_corrected(fr: ir::FinishReason, stop_matched: bool) -> ir::FinishReason {
    if stop_matched && fr == ir::FinishReason::Length {
        ir::FinishReason::Stop
    } else {
        fr
    }
}

#[cfg(test)]
mod stop_match_corrected_tests {
    use super::stop_match_corrected;
    use crate::ir::{FINISH_REASON_TIMEOUT, FinishReason};

    #[test]
    fn matched_stop_corrects_length_to_stop() {
        assert_eq!(
            stop_match_corrected(FinishReason::Length, true),
            FinishReason::Stop
        );
    }

    #[test]
    fn everything_else_passes_through() {
        assert_eq!(
            stop_match_corrected(FinishReason::Length, false),
            FinishReason::Length
        );
        assert_eq!(
            stop_match_corrected(FinishReason::ToolCalls, true),
            FinishReason::ToolCalls
        );
        assert_eq!(
            stop_match_corrected(FinishReason::Other(FINISH_REASON_TIMEOUT.into()), true),
            FinishReason::Other(FINISH_REASON_TIMEOUT.into())
        );
    }
}
