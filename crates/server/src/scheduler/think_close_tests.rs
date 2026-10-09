// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: `think_close`: a GLM-4.7 tool call written while thinking ends
//! the thinking block, so the EOS after the call ends the turn. Driven through
//! the real `emit_token` over a `test_seq` fixture, with GLM-5.3's ids for the
//! call markers and the fixture's EOS standing in for `<|observation|>`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::super::emit_step::emit_token;
use super::super::sched_ctx::SchedCtx;
use super::super::test_support::{EOS, test_seq};
use super::super::types::ActiveSeq;
use super::close_thinking_at_tool_call;

const THINK_END: u32 = 154842;
const TOOL_CALL: u32 = 154843;
const TOOL_CALL_END: u32 = 154844;
/// 2026-10-09: `get`, `_weather`, `<arg_key>`, `city`, `</arg_key>`,
/// `<arg_value>`, `Paris`, `</arg_value>`.
const CALL_BODY: [u32; 8] = [455, 68852, 154847, 8923, 154848, 154849, 59190, 154850];

/// 2026-10-09: A sequence the prompt left inside `<think>` (GLM's template
/// always opens it), with `min_tokens` already met and no grammar.
fn glm_seq() -> ActiveSeq {
    let (mut a, _rx) = test_seq((1000..1007).collect(), 5000, None, 10);
    a.finished = false;
    a.inside_thinking = true;
    a.think_end_token = Some(THINK_END);
    a.tool_call_start_token = Some(TOOL_CALL);
    a.tool_call_end_token = Some(TOOL_CALL_END);
    a
}

fn sched(closed_by: Option<u32>) -> SchedCtx {
    let mut s = SchedCtx::for_test();
    s.limits.thinking_closed_by = closed_by;
    s
}

/// 2026-10-09: "I'll check the weather in Paris for you." (as reasoning, no
/// `</think>`), the call, then the EOS the model writes after it.
fn run_call_written_while_thinking(a: &mut ActiveSeq, s: &SchedCtx) {
    for tok in [40, 3278, 1779, 279, 9101, 304, 12089, 369, 498, 13] {
        emit_token(a, tok, None, s);
    }
    emit_token(a, TOOL_CALL, None, s);
    for tok in CALL_BODY {
        emit_token(a, tok, None, s);
    }
    emit_token(a, TOOL_CALL_END, None, s);
    emit_token(a, EOS[0], None, s);
}

#[test]
fn a_call_written_while_thinking_ends_thinking_and_the_eos_after_it_ends_the_turn() {
    let s = sched(Some(TOOL_CALL));
    let mut a = glm_seq();
    run_call_written_while_thinking(&mut a, &s);
    assert!(!a.inside_thinking);
    assert!(
        a.tool_call_completed,
        "</tool_call> marks the call complete"
    );
    assert!(a.finished, "the EOS after the call must end the turn");
    let n = a.output_tokens.len();
    assert_eq!(&a.output_tokens[n - 2..], &[TOOL_CALL_END, EOS[0]]);
    assert_eq!(
        a.output_tokens.iter().filter(|&&t| t == TOOL_CALL).count(),
        1,
        "the opener is kept in the output, once"
    );
}

#[test]
fn without_the_format_flag_the_eos_after_the_call_is_held_back() {
    // 2026-10-09: The failure this module fixes, on the same tokens: thinking is
    // still open, the EOS is dropped and the turn runs on.
    let s = sched(None);
    let mut a = glm_seq();
    run_call_written_while_thinking(&mut a, &s);
    assert!(a.inside_thinking);
    assert!(!a.tool_call_completed);
    assert!(!a.finished);
}

#[test]
fn only_the_named_opener_closes_and_only_while_thinking() {
    let mut a = glm_seq();
    assert!(!close_thinking_at_tool_call(&mut a, TOOL_CALL, None));
    assert!(!close_thinking_at_tool_call(&mut a, 1000, Some(TOOL_CALL)));
    assert!(a.inside_thinking);
    a.force_end_thinking = true;
    assert!(close_thinking_at_tool_call(
        &mut a,
        TOOL_CALL,
        Some(TOOL_CALL)
    ));
    assert!(!a.inside_thinking && a.think_ended && !a.force_end_thinking);
    assert!(
        !close_thinking_at_tool_call(&mut a, TOOL_CALL, Some(TOOL_CALL)),
        "outside thinking an opener is plain content"
    );
}

/// 2026-10-09: `ToolCallDuringThinkingMask` leaves the opener alone while thinking
/// only when the opener closes thinking; it still masks it for every other format.
#[test]
fn the_during_thinking_mask_spares_an_opener_that_closes_thinking() {
    use crate::scheduler::logit_processors::tool_during_think::ToolCallDuringThinkingMask;
    use crate::scheduler::logit_processors::{LogitsContext, LogitsProcessor, SamplingLevers};
    let scratch = crate::scheduler::sched_ctx::DecodeScratch::default();
    let io = crate::scheduler::io::SchedIo::for_test();
    let logit_after = |closed_by: Option<u32>| {
        let ctx = LogitsContext {
            scratch: &scratch,
            tel: &*io.tel,
            clock: &*io.clock,
            watchdog: crate::scheduler::helpers::WatchdogParams::default(),
            boundary_mask: None,
            mid_word_mask: None,
            sampling: SamplingLevers::default(),
            think_end_token: Some(THINK_END),
            think_start_token: None,
            tool_call_start_token: Some(TOOL_CALL),
            tool_call_end_token: Some(TOOL_CALL_END),
            thinking_closed_by: closed_by,
            verify_pos: 0,
        };
        let mut a = glm_seq();
        let mut logits = vec![1.0f32; TOOL_CALL as usize + 1];
        ToolCallDuringThinkingMask.apply(&mut logits, &mut a, &ctx);
        logits[TOOL_CALL as usize]
    };
    assert_eq!(logit_after(None), f32::NEG_INFINITY);
    assert_eq!(logit_after(Some(TOOL_CALL)), 1.0);
}
