// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: the masked chat fast path of `verify_pick_all_with_pipeline`:
//! returns the GPU argmax picks without copying the logits rows when the
//! checks below find no stage that would change them.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types. `try_chat_fast_path` does not mutate
//! the sequence.

use crate::scheduler::ActiveSeq;
use crate::scheduler::logit_processors::LogitsContext;
use crate::scheduler::mtp_timing::Phase;
use metrale_model_engine::traits::Model;

/// 2026-09-25: `Some(argmax_ids)` when every check passes, `None` when the
/// caller must try its other paths.
///
/// Requires: the `dflash_masked_verify` and `fast_masked` levers on, no
/// grammar, AdaDec diagnostics off, `F2ConfidenceEarlyStop`, the forced
/// `</think>` (armed now or by a thinking budget the window's picks reach)
/// and the tool-call pin all inactive, penalties not `Blocked`,
/// and at every position an argmax that is not the think-end, think-start
/// or tool-call-start token and that the penalties and the request's
/// `logit_bias` cannot move (`fast_greedy::argmax_survives`).
///
/// An EOS argmax that `eos_hold::presample_hold` masks (`EosHoldMask`)
/// declines the path. Not checked: the temperature, so above 0 this returns
/// the argmax where the host path, with `mtp_verify_sample` on, would sample.
///
/// `row_base` is as for `verify_pick_all_with_pipeline`.
pub(super) fn try_chat_fast_path(
    model: &dyn Model,
    argmax_ids: &[u32],
    a: &ActiveSeq,
    ctx: &LogitsContext,
    row_base: usize,
) -> Option<Vec<u32>> {
    // 2026-09-25: the gate is the `dflash_masked_verify` lever itself (on
    // unless `METRALE_DFLASH_MASKED_VERIFY=0`), not whether the serve has a
    // DFlash drafter, so MTP verify steps reach this path too. Where it
    // fires, the GPU argmax replaces the host argmax, which can break
    // near-ties differently.
    if !ctx.sampling.dflash_masked_verify {
        return None;
    }
    let fast_masked_enabled = ctx.sampling.fast_masked;
    let adadec_recording = ctx.sampling.adadec_diagnostic;
    if !fast_masked_enabled
        || a.grammar_state.is_some()
        || adadec_recording
        || crate::scheduler::eos_hold::eos_masked_in_window(a, argmax_ids)
    {
        return None;
    }
    // 2026-09-25: the same `penalty_params_for` call the host path makes for
    // every verify position. 2026-10-21: its `logit_bias` is checked per
    // position below (`fast_greedy::argmax_survives`).
    let penalties = crate::scheduler::sample_step::penalty_params_for(
        a,
        crate::scheduler::sample_step::PositionKind::Verify,
        0.0,
        None,
        a.logit_bias.clone(),
        ctx.watchdog.min_reasoning_floor,
    );
    let penalty_gate = crate::scheduler::fast_greedy::classify_penalties_except_bias(&penalties);
    if stages_may_act(a, ctx, argmax_ids.len())
        || penalty_gate == crate::scheduler::fast_greedy::PenaltyGate::Blocked
    {
        return None;
    }
    let t_fast = ctx.clock.now();
    let scoped_history: Vec<u32> =
        if penalty_gate == crate::scheduler::fast_greedy::PenaltyGate::ReduceOnly {
            crate::scheduler::sample_step::penalty_history_scope(
                &a.output_tokens,
                ctx.tool_call_end_token,
            )
            .to_vec()
        } else {
            Vec::new()
        };
    let vocab = model.vocab_size();
    let logits_base = model.logits_buffer_ptr();
    let mut all_clear = true;
    for (i, &tok) in argmax_ids.iter().enumerate() {
        if is_think_or_tool_id(tok, a, ctx) {
            all_clear = false;
            break;
        }
        if !crate::scheduler::fast_greedy::argmax_survives(
            model,
            logits_base,
            row_base + i,
            vocab,
            tok,
            penalty_gate,
            &scoped_history,
            &penalties.logit_bias,
        ) {
            all_clear = false;
            break;
        }
    }
    ctx.tel.mark(Phase::FastGreedy, t_fast);
    if all_clear {
        if ctx.tel.stats().once("log:verify_chat_fast_path") {
            tracing::info!(
                "verify chat fast path ACTIVE: masked-greedy == raw argmax, no D2H \
                 (kill-switch: METRALE_DISABLE_FAST_MASKED=1)"
            );
        }
        return Some(argmax_ids.to_vec());
    }
    None
}

/// 2026-10-21: Whether `F2ConfidenceEarlyStop`, `ForcedThinkEndInjector`
/// (inject or defer) or `PinToToolCallStart` may act at some position of a
/// `window`-position pick (see their `apply` bodies): true whenever one would,
/// and sometimes when it would not. The window's thinking picks count toward
/// `thinking_budget` (`pick_positions.rs`), so a budget they reach arms the
/// forced `</think>` for the positions after it.
pub(super) fn stages_may_act(a: &ActiveSeq, ctx: &LogitsContext, window: usize) -> bool {
    use crate::scheduler::confidence::{
        MAX_SENTENCE_DEFER_TOKENS, THINK_DEFER_ABS_CEILING, THINK_DEFER_BUDGET_FACTOR,
    };
    let f2_active = !ctx.sampling.disable_watchdogs
        && a.inside_thinking
        && !a.force_end_thinking
        && a.thinking_tokens >= 400
        && ctx.watchdog.confidence_early_stop;
    let defer_hard_override = match a.thinking_budget {
        Some(b) => a.thinking_tokens >= b.saturating_mul(THINK_DEFER_BUDGET_FACTOR),
        None => a.thinking_tokens >= THINK_DEFER_ABS_CEILING,
    } || a.sentence_defer_count >= MAX_SENTENCE_DEFER_TOKENS;
    let think_end_inject_armed = a.inside_thinking && (a.force_end_thinking || defer_hard_override);
    let window_thinking_picks = window.saturating_sub(1) as u32;
    let budget_reached_in_window = a.inside_thinking
        && a
            .thinking_budget
            .is_some_and(|b| a.thinking_tokens.saturating_add(window_thinking_picks) >= b);
    let pin_tool_armed =
        a.think_just_ended && a.require_tool_call && !a.tool_call_opened && !a.inside_thinking;
    f2_active || think_end_inject_armed || budget_reached_in_window || pin_tool_armed
}

/// 2026-09-25: the ids `MidWordThinkEndMask`, `PostCloseThinkMask` and
/// `ToolCallDuringThinkingMask` can mask or lower.
pub(super) fn is_think_or_tool_id(tok: u32, a: &ActiveSeq, ctx: &LogitsContext) -> bool {
    Some(tok) == ctx.think_end_token
        || Some(tok) == a.think_start_token
        || Some(tok) == ctx.tool_call_start_token
}
