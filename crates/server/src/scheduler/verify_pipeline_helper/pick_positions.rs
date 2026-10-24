// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: the host-side K-position pick loop of
//! `verify_pick_all_with_pipeline`; it takes host logits rather than a
//! model, so tests drive it directly.
//!
//! Owner: scheduler.
//! Invariants:
//! - On return the grammar matcher is at the history depth it had on entry,
//!   and the thinking state ([`ThinkState`]) holds its entry values, except
//!   the counters only the pipeline stages write (`force_end_thinking` as
//!   `F2ConfidenceEarlyStop` arms it, `sentence_defer_count`,
//!   `consecutive_confident`), which hold their values after position 0: that
//!   pick is always emitted, and `emit_token` re-derives the rest per emitted
//!   token. Within the loop, each later position is masked against the
//!   matcher state the earlier picks leave behind, and a `</think>` pick
//!   switches the thinking flags for the remaining positions, so the first
//!   position after it is picked outside thinking. 2026-10-14: A `<think>`
//!   pick outside thinking switches them back, as `emit_token` does.
//!   2026-10-21: A tool-call opener that closes thinking
//!   (`LogitsContext::thinking_closed_by`) closes it in the window too, and
//!   the matcher takes it, as on the decode path; a thinking pick counts
//!   toward `thinking_budget`, so a budget reached inside the window forces
//!   `</think>` at the next position, as the next serial step would.

use super::verify_pick_with_pipeline;
use crate::scheduler::logit_processors::LogitsContext;
use crate::scheduler::types::ActiveSeq;

/// 2026-09-25: run `verify_pick_with_pipeline` over `k` host-resident BF16
/// rows of `vocab` logits and return the pick per position. Stops early,
/// returning fewer than `k` picks, when the matcher refuses a pick.
pub(super) fn pick_positions_from_host(
    buf: &[u8],
    vocab: usize,
    elem_bytes: usize,
    k: usize,
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
) -> Vec<u32> {
    let mut picks: Vec<u32> = Vec::with_capacity(k);
    // 2026-09-25: the history depth, for the rollback at the end; see there
    // for why it is a depth and not a count of `accept_token` calls.
    let grammar_steps_before = a.grammar_state.as_ref().map(|gs| gs.num_history_steps());
    let think_before = ThinkState::take(a);
    let mut stage_counters_after_first = None;

    for i in 0..k {
        let slice = &buf[i * vocab * elem_bytes..(i + 1) * vocab * elem_bytes];
        let pick = verify_pick_with_pipeline(slice, false, vocab, a, ctx, i);
        picks.push(pick);
        if i == 0 {
            stage_counters_after_first = Some(StageCounters::take(a));
        }

        // 2026-09-25: `</think>` closes thinking for the later positions. It
        // is not fed to the matcher.
        if a.inside_thinking && ctx.think_end_token == Some(pick) {
            a.inside_thinking = false;
            a.think_ended = true;
            a.think_just_ended = true;
            continue;
        }
        // 2026-10-14: `<think>` outside thinking re-opens it for the later
        // positions (`emit_token`). It is not fed to the matcher.
        if !a.inside_thinking && ctx.think_start_token == Some(pick) {
            a.inside_thinking = true;
            a.think_ended = false;
            a.think_just_ended = false;
            continue;
        }
        // 2026-10-21: as `process_decoded_token`: an opener that closes
        // thinking leaves it here and is fed to the matcher below.
        crate::scheduler::think_close::close_thinking_at_tool_call(
            a,
            pick,
            ctx.thinking_closed_by,
        );
        // 2026-10-21: a thinking token counts toward the budget; one that
        // reaches it arms the forced `</think>` for the next position.
        if a.inside_thinking {
            crate::scheduler::think_close::count_thinking_token(a);
            continue;
        }

        // 2026-09-25: advance the matcher with the pick so the next position
        // is masked against the state after it. Not after the last position,
        // and not inside thinking.
        if i + 1 < k
            && let Some(ref mut gs) = a.grammar_state
            && !a.inside_thinking
        {
            // 2026-09-25: a refused advance ends the loop; the picks so far
            // are returned.
            if !gs.accept_token(pick) {
                tracing::debug!(
                    pick,
                    i,
                    "verify_pick: grammar speculative advance refused — pipeline picked a token outside the current bitmask. \
                     This indicates a stale bitmask in the pipeline or a forced-token fastpath that terminated grammar. \
                     Stopping speculation here; emit_token keeps the token and leaves the grammar unadvanced."
                );
                break;
            }
        }
    }

    // 2026-09-25: roll back by the history delta. Counting `accept_token`
    // calls would over-rewind: `GrammarState::accept_token` returns true for
    // stop tokens and in the terminated state without adding a history step.
    // `emit_token` later advances the matcher for the tokens it emits.
    if let (Some(before), Some(gs)) = (grammar_steps_before, a.grammar_state.as_mut()) {
        let advanced = gs.num_history_steps().saturating_sub(before);
        if advanced > 0 {
            gs.rollback(advanced);
        }
    }
    // 2026-09-25: restore the thinking state; `emit_token` makes the real
    // transitions for the tokens it emits.
    think_before.restore(a);
    if let Some(c) = stage_counters_after_first {
        c.restore(a);
    }

    picks
}

/// 2026-10-21: The thinking state the window advances and `emit_token`
/// re-derives per emitted token.
struct ThinkState {
    inside_thinking: bool,
    think_ended: bool,
    think_just_ended: bool,
    thinking_tokens: u32,
    in_code_fence: bool,
    stage: StageCounters,
}

impl ThinkState {
    fn take(a: &ActiveSeq) -> Self {
        Self {
            inside_thinking: a.inside_thinking,
            think_ended: a.think_ended,
            think_just_ended: a.think_just_ended,
            thinking_tokens: a.thinking_tokens,
            in_code_fence: a.in_code_fence,
            stage: StageCounters::take(a),
        }
    }

    fn restore(self, a: &mut ActiveSeq) {
        a.inside_thinking = self.inside_thinking;
        a.think_ended = self.think_ended;
        a.think_just_ended = self.think_just_ended;
        a.thinking_tokens = self.thinking_tokens;
        a.in_code_fence = self.in_code_fence;
        self.stage.restore(a);
    }
}

/// 2026-10-21: The thinking counters the pipeline stages write
/// (`F2ConfidenceEarlyStop`, `ForcedThinkEndInjector`) and the window's
/// budget count and thinking closes reset.
struct StageCounters {
    force_end_thinking: bool,
    sentence_defer_count: u32,
    consecutive_confident: u32,
}

impl StageCounters {
    fn take(a: &ActiveSeq) -> Self {
        Self {
            force_end_thinking: a.force_end_thinking,
            sentence_defer_count: a.sentence_defer_count,
            consecutive_confident: a.consecutive_confident,
        }
    }

    fn restore(self, a: &mut ActiveSeq) {
        a.force_end_thinking = self.force_end_thinking;
        a.sentence_defer_count = self.sentence_defer_count;
        a.consecutive_confident = self.consecutive_confident;
    }
}
