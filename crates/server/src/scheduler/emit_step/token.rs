// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `emit_token`: stream and bookkeeping for one token emitted by
//! the MTP, speculative and DFlash steps.
//!
//! Owner: scheduler.
//! Invariants:
//! - The `<tool_response>` and stray-`</think>` hard stops, and the
//!   post-think cap, content-loop and prose-budget stops, set `a.guard_stop`
//!   where they set `finished`. `finish_guard_tests.rs` pins the count.
//! - 2026-10-14: An EOS is decided right after the hard stops, with the same
//!   `eos_hold` as `decode_logits_step`. A held one leaves no trace (not
//!   pushed, not charged, the grammar not advanced); one that ends the turn
//!   is pushed and charged, as vLLM counts its EOS.
//! - 2026-10-14: A token the grammar rejects is kept and the grammar stays
//!   attached, unadvanced, as vLLM's structured-output manager does.

use super::*;

/// 2026-09-25: Emit a token for an active sequence (stream + bookkeeping).
///
/// EOS and the control tokens that end a turn are not streamed. An EOS that
/// ends the turn is still recorded in `output_tokens` and charged, so it
/// counts as a generated token.
///
/// When `logprobs` is `Some`, it is appended to `a.logprobs_data` and the
/// token is streamed as `StreamEvent::TokenWithLogprobs`.
pub fn emit_token(
    a: &mut ActiveSeq,
    tok: u32,
    logprobs: Option<crate::api::TokenLogprobs>,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) {
    // 2026-09-25: per-token debug line (`slot`, `out_idx`, `tok`), for diffing
    // one stream's tokens between two runs.
    tracing::debug!(
        "TOK slot={} out_idx={} tok={}",
        a.seq.slot_idx,
        a.output_tokens.len(),
        tok,
    );
    // 2026-09-25: the streaming side set the request's cancel flag: finish
    // now, without naming a guard.
    if sched.io.req.is_cancelled(a.cancel_flag.as_ref()) {
        a.finished = true;
        return;
    }

    // 2026-09-25: ChatML role-boundary hard stop (`<|im_start|>`). It runs
    // before the EOS hold below (grammar, `require_tool_call`,
    // `min_tokens`), which would otherwise hold the token back and
    // let generation continue.
    if let Some(ims) = sched.limits.im_start_hard_stop
        && tok == ims
    {
        // 2026-09-25: pushed, not streamed. `tokenizer_runtime.rs` adds
        // `<|im_start|>` to `eos_tokens`, so with it as the last token
        // `derive_finish_reason` takes its EOS rung and reports "stop".
        a.output_tokens.push(tok);
        a.finished = true;
        tracing::debug!(
            "<|im_start|> hard-stop fired (id={ims}); ending turn before grammar/suppress_eos"
        );
        return;
    }

    // 2026-09-25: `<tool_response>` hard stop, armed by
    // `SchedLevers::tool_response_stop`: emitting this control token ends
    // the turn.
    if sched.levers.tool_response_stop
        && let Some(trs) = sched.limits.tool_response_hard_stop
        && tok == trs
    {
        a.output_tokens.push(tok);
        a.finished = true;
        // 2026-09-25: named, as in `decode_logits_step`: this token is not in
        // `eos_tokens`, so unnamed the finish would report "stop" while
        // budget remains.
        a.guard_stop = Some(GUARD_STOP_TOOL_RESPONSE);
        tracing::debug!("<tool_response> hard-stop fired (id={trs}); ending turn");
        return;
    }

    // 2026-10-13: EOS: the same `eos_hold` decision as `decode_logits_step`.
    if a.eos_tokens.contains(&tok) {
        let generated_before = a.output_tokens.len();
        let hold =
            crate::scheduler::eos_hold::eos_hold(a, generated_before, sched.levers.tool_eos_escape);
        if hold.any() {
            // 2026-10-14: The verify pickers mask a held EOS before they pick
            // (`EosHoldMask`, `eos_masked_in_window`), as vLLM's sampler
            // does for the draft and bonus positions. One that still arrives
            // is dropped: not pushed, not charged, not streamed.
            tracing::debug!(
                target: "metrale::eos",
                tok,
                inside_thinking = a.inside_thinking,
                by_grammar = hold.grammar,
                by_required_tool_call = hold.required_tool_call,
                by_min_tokens = hold.min_tokens,
                "EOS suppressed in emit_token; model forced to continue"
            );
            return;
        }
        crate::scheduler::eos_hold::end_turn_at_eos(a, tok, logprobs);
        return;
    }

    // 2026-09-25: a `<think>` outside thinking enters thinking mode with the
    // spontaneous-think budget; the token itself is not emitted.
    // 2026-10-14: Without thinking steering (no spontaneous budget) the
    // request's budget stays and the token is kept as reasoning text, as in
    // vLLM and `decode_logits_step`.
    let reopened_think = !a.inside_thinking && a.think_start_token == Some(tok);
    if reopened_think {
        a.inside_thinking = true;
        // 2026-09-25: restart the post-`</think>` count that
        // `mtp_gate::spec_dispatch_eligible` reads.
        a.post_think_emitted = 0;
        a.think_ended = false;
        a.think_skip_count = 0;
        if let Some(spontaneous) = a.spontaneous_think_budget {
            a.thinking_budget = Some(spontaneous);
        }
        tracing::debug!("Spontaneous <think> detected in emit_token, entering thinking mode");
        if sched.watchdog.thinking_steering {
            return;
        }
    }

    // 2026-09-25: a `</think>` outside thinking is skipped, as in
    // `process_decode_logits`; the 50th in a row ends the turn.
    // 2026-10-14: only with thinking steering; without it the token is
    // content, as in vLLM.
    if sched.watchdog.thinking_steering && !a.inside_thinking && a.think_end_token == Some(tok) {
        a.think_skip_count += 1;
        if a.think_skip_count >= 50 {
            a.finished = true;
            // 2026-09-25: named: the skipped `</think>` is not pushed, so
            // unnamed the finish would report "stop" while budget remains.
            a.guard_stop = Some(GUARD_STOP_THINK_SKIP);
            tracing::debug!(
                "</think> think-skip watchdog hard-stop fired (50 consecutive strays); \
                 ending turn"
            );
        }
        return;
    }
    // 2026-09-25: any other token once thinking has ended resets the stray
    // count, as in `decode_logits_step`, so only consecutive strays add up.
    if a.think_ended {
        a.think_skip_count = 0;
    }

    // 2026-10-09: under a format whose tool call closes the thinking block, the
    // opener leaves thinking here and is handled below as a content token.
    crate::scheduler::think_close::close_thinking_at_tool_call(
        a,
        tok,
        sched.limits.thinking_closed_by,
    );

    // 2026-09-25: a `<tool_call>` outside thinking satisfies
    // `require_tool_call`.
    if a.require_tool_call && a.tool_call_start_token == Some(tok) && !a.inside_thinking {
        a.require_tool_call = false;
        a.tool_call_opened = true;
    }

    // 2026-09-25: tool-body / parameter-body state. `decode_logits_step`
    // calls the same `update_tool_param_state`, so both paths track it.
    update_tool_param_state(a, tok);

    // 2026-09-25: a `</tool_call>` outside thinking marks a tool call
    // complete, which the `tool_eos_escape` gate below reads.
    if a.tool_call_end_token == Some(tok) && !a.inside_thinking {
        a.tool_call_completed = true;
    }

    // 2026-09-25: a `<tool_call>` outside thinking resets the inter-tool
    // prose count, as in `decode_logits_step`, so the prose budget below
    // counts only tokens since the last tool call opened.
    if a.tool_call_start_token == Some(tok) && !a.inside_thinking {
        a.prose_tokens_since_last_tool = 0;
    }

    // 2026-09-25: advance the grammar, except inside thinking: the matcher
    // never sees thinking tokens or the closing `</think>`.
    // 2026-10-14: a token the matcher refuses is kept and the grammar stays
    // attached at its last accepted position, as vLLM's structured-output
    // manager does (it logs and goes on); `decode_logits_step` does the same.
    if !a.inside_thinking
        && let Some(ref mut gs) = a.grammar_state
        && !gs.accept_token(tok)
    {
        tracing::error!(
            tok,
            output_len = a.output_tokens.len(),
            "grammar refused an emitted token; keeping the token and the grammar unadvanced"
        );
    }

    if let Some(lp) = logprobs {
        a.logprobs_data.push(lp);
    }

    a.output_tokens.push(tok);

    // 2026-09-25: count tokens emitted after thinking ended, for
    // `mtp_gate::spec_dispatch_eligible`. `</think>` itself is not counted:
    // `think_ended` is still false when it arrives. A request with thinking
    // off on a model with a `</think>` token starts with `think_ended` true
    // (prefill), so it counts from its first token.
    if a.think_ended && !a.inside_thinking {
        a.post_think_emitted += 1;
    }

    // 2026-09-25: thinking tokens draw down `remaining` like content tokens,
    // as in `decode_logits_step`; `thinking_budget` is the separate
    // per-block cap armed below.
    if a.inside_thinking {
        a.consume_generation_budget();
        if a.think_end_token == Some(tok) {
            a.inside_thinking = false;
            a.think_force_closed = a.force_end_thinking;
            a.force_end_thinking = false;
            a.sentence_defer_count = 0;
            a.think_ended = true;
            // 2026-09-25: one-shot read by `PinToToolCallStart` on the next
            // step; cleared by the next non-thinking token below.
            a.think_just_ended = true;
            tracing::info!(
                "Thinking ended after {} tokens (budget={:?})",
                a.thinking_tokens,
                a.thinking_budget,
            );
        } else if !reopened_think {
            a.thinking_tokens += 1;
            if let Some(budget) = a.thinking_budget
                && a.thinking_tokens >= budget
                && !a.force_end_thinking
            {
                a.force_end_thinking = true;
                a.sentence_defer_count = 0;
                // 2026-09-25: the log names the budget's source, as in
                // `decode_logits_step`.
                tracing::info!(
                    source = if a.enable_thinking {
                        "request (client budget/effort; scaled by --max-thinking-budget)"
                    } else {
                        "spontaneous <think> (--max-thinking-budget / MODEL.toml)"
                    },
                    "Thinking budget exhausted ({budget} tokens), arming </think>; \
                     deferring to next sentence boundary"
                );
            }
        }
    } else {
        a.consume_generation_budget();
        // 2026-09-25: the token after `</think>` clears the one-shot.
        a.think_just_ended = false;
        // 2026-09-25: content-phase watchdogs. `handle_content_token`
        // (`decode_logits_content.rs`) runs only on the
        // `process_decode_logits` path; the steps that emit through here
        // get the same checks below, with the same gates. `emit_token` has
        // no `&dyn Model`, which `rollback_to_boundary` needs, so where
        // `handle_content_token` first tries a rollback these stops end the
        // response.
        use crate::scheduler::helpers::{
            CONTENT_LOOP_CHECK_STRIDE, CONTENT_LOOP_MIN_TOKENS, CONTENT_LOOP_PERIOD_MAX,
            CONTENT_LOOP_PERIOD_MIN, detect_content_token_loop_normalized_with,
            detect_content_token_loop_with,
        };
        a.content_tokens = a.content_tokens.saturating_add(1);
        // 2026-09-25: post-think content cap, applied whatever
        // `inside_tool_body` says, while a grammar is attached. The cap is
        // MODEL.toml `[behavior].max_post_think_content_tokens`, 100,000
        // when unset.
        if !sched.levers.disable_watchdogs
            && a.grammar_state.is_some()
            && a.content_tokens > sched.watchdog.max_post_think_content_tokens
        {
            tracing::warn!(
                content_tokens = a.content_tokens,
                max = sched.watchdog.max_post_think_content_tokens,
                "post-think content cap exceeded in MTP/emit path; ending response (tool-active request would otherwise burn to max_tokens)"
            );
            a.guard_stop = Some(GUARD_STOP_POST_THINK_CAP);
            a.finished = true;
        }
        // 2026-09-25: request `repetition_detection`, then the operator's
        // min-repeats override, then the built-in constants
        // (`WatchdogParams::content_loop_params`).
        let loop_params = sched.watchdog.content_loop_params(a.repetition_detection);
        if !sched.levers.disable_watchdogs
            && sched.levers.loop_watchdog()
            && !a.inside_tool_body
            && a.content_tokens >= CONTENT_LOOP_MIN_TOKENS
            && a.content_tokens.is_multiple_of(CONTENT_LOOP_CHECK_STRIDE)
            && (detect_content_token_loop_with(&a.output_tokens, loop_params)
                || sched.masks.numeric.as_deref().is_some_and(|m| {
                    detect_content_token_loop_normalized_with(&a.output_tokens, m, loop_params)
                }))
        {
            tracing::warn!(
                content_tokens = a.content_tokens,
                output_len = a.output_tokens.len(),
                "Content-loop watchdog fired in MTP/emit path (period-{}…{} repeat); ending response. \
                 Tune via --content-loop-min-repeats / METRALE_CONTENT_LOOP_MIN_REPEATS, per-request \
                 repetition_detection, or disarm via --content-loop-watchdog off / \
                 METRALE_CONTENT_LOOP_WATCHDOG=0",
                CONTENT_LOOP_PERIOD_MIN,
                CONTENT_LOOP_PERIOD_MAX,
            );
            // 2026-09-25: named, or the finish would report "stop".
            a.guard_stop = Some(GUARD_STOP_CONTENT_LOOP);
            a.finished = true;
        }

        // 2026-09-25: inter-tool prose budget. It counts tokens outside a
        // tool body on requests with `tool_request` set. `tool_request` is
        // fixed at prefill, so the budget still applies after the grammar
        // is dropped.
        if !sched.levers.disable_watchdogs && !a.inside_tool_body && a.tool_request {
            a.prose_tokens_since_last_tool = a.prose_tokens_since_last_tool.saturating_add(1);
            let max_prose = sched.watchdog.max_inter_tool_prose;
            if a.prose_tokens_since_last_tool > max_prose {
                tracing::warn!(
                    prose_tokens = a.prose_tokens_since_last_tool,
                    max = max_prose,
                    output_len = a.output_tokens.len(),
                    "Inter-tool prose budget exhausted in MTP/emit path; ending response \
                     (no tool call after budget — would otherwise burn to max_tokens); \
                     raise via --max-inter-tool-prose / METRALE_MAX_INTER_TOOL_PROSE / \
                     MODEL.toml [behavior].max_inter_tool_prose (0 disables)"
                );
                a.guard_stop = Some(GUARD_STOP_INTER_TOOL_PROSE);
                a.finished = true;
            }
        }
    }

    // 2026-09-25: thinking tokens of a request without thinking enabled are
    // not streamed, the same gate as `process_decode_logits`.
    let suppress_stream = a.inside_thinking && !a.enable_thinking;
    if !suppress_stream {
        let event = if let Some(lp) = a.logprobs_data.last().cloned() {
            StreamEvent::TokenWithLogprobs(tok, lp)
        } else {
            StreamEvent::Token(tok)
        };
        if !sched.io.req.emit(&a.sink, event, "token stream") {
            a.finished = true;
            return;
        }
    }
    // 2026-09-25: length stop: budget used up, or the next step would reach
    // the served `max_seq_len` (`seqlen_force_stop`, never when it is 0).
    if a.remaining == 0 || seqlen_force_stop(a.seq.seq_len, sched.limits.max_seq_len) {
        // 2026-09-25: first try to close an open grammar structure
        // (`emit_grammar_close`).
        emit_grammar_close(a, &sched.io, sched.levers.grammar_budget_close);
        tracing::info!(
            "emit_token: remaining={}, seq_len={}, max_seq_len={}, output_tokens={}, thinking_tokens={}",
            a.remaining,
            a.seq.seq_len,
            sched.limits.max_seq_len,
            a.output_tokens.len(),
            a.thinking_tokens
        );
        a.finished = true;
    }
}
