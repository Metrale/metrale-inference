// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: Tests for the vLLM EOS rule (`scheduler::eos_hold`) on every
//! path that can sample an EOS: `process_decoded_token` (decode),
//! `emit_token` (MTP, speculative and DFlash steps) and `k4_apply_verdict`
//! (verify).
//!
//! vLLM masks EOS only for `min_tokens` and for a structured-output grammar,
//! and does not apply the grammar inside the reasoning block. The case that
//! returned garbage on GLM-5.3: the template opens `<think>`, the model
//! answers "4" and samples EOS with no `</think>`. vLLM ends the request
//! there and reports the "4" as reasoning.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::per_token::process_decoded_token;
use crate::grammar::tests::{test_tool_defs, test_vocab};
use crate::grammar::{GrammarEngine, GrammarState};
use crate::scheduler::emit_step::emit_token;
use crate::scheduler::io::SchedIo;
use crate::scheduler::lifecycle::finish_sequence;
use crate::scheduler::sched_ctx::SchedCtx;
use crate::scheduler::test_support::{EOS, PreemptStubModel, RespRx, test_seq};
use crate::scheduler::types::ActiveSeq;
use crate::scheduler::verify_k4_verdict::{K4Hidden, k4_apply_verdict};
use std::sync::Arc;
use std::time::Instant;

/// 2026-10-13: The token for "4": any id that is not a control token.
const ANSWER: u32 = 1004;
const THINK_START: u32 = 151_667;
const THINK_END: u32 = 151_668;
const TOOL_CALL_START: u32 = 151_657;
/// 2026-10-13: Served ceiling for `finish_sequence`, far above every
/// fixture's `seq_len`, so no finish here reports "length" for the context.
const MAX_SEQ_LEN: usize = 4096;
/// 2026-10-13: `<eos>` in the toy vocab of `grammar::tests`.
const TOY_EOS: u32 = 130;

#[derive(Clone, Copy, Debug)]
enum Path {
    Decode,
    Emit,
}

const PATHS: [Path; 2] = [Path::Decode, Path::Emit];

/// 2026-10-13: An unfinished sequence with `output` already generated and
/// `min_tokens` 0, so nothing but the term a test arms can hold an EOS
/// back. `remaining` is 5000, so no budget stop fires.
fn seq(output: Vec<u32>, inside_thinking: bool) -> (ActiveSeq, RespRx) {
    let seq_len = 10 + output.len();
    let (mut a, rx) = test_seq(output, 5000, None, seq_len);
    a.finished = false;
    a.min_tokens = 0;
    a.enable_thinking = true;
    a.inside_thinking = inside_thinking;
    a.think_ended = !inside_thinking;
    a.think_start_token = Some(THINK_START);
    a.think_end_token = Some(THINK_END);
    a.tool_call_start_token = Some(TOOL_CALL_START);
    (a, rx)
}

/// 2026-10-13: A scheduler context over the stub model's sync device router:
/// a content token on the decode path applies a Marconi checkpoint, which
/// `SchedCtx::for_test()` (no device) refuses.
fn sched() -> SchedCtx {
    SchedCtx::for_test_with(Arc::new(PreemptStubModel::default()))
}

/// 2026-10-13: Feed one sampled token down `path`, as the decode step or
/// the MTP/speculative steps would.
fn feed(path: Path, a: &mut ActiveSeq, tok: u32, sched: &SchedCtx) {
    match path {
        Path::Decode => {
            let think_end = a.think_end_token;
            let think_start = a.think_start_token;
            let tool_start = a.tool_call_start_token;
            let tool_end = a.tool_call_end_token;
            process_decoded_token(
                a,
                tok,
                None,
                Instant::now(),
                think_end,
                think_start,
                None,
                tool_start,
                tool_end,
                &PreemptStubModel::default(),
                sched,
            );
        }
        Path::Emit => emit_token(a, tok, None, sched),
    }
}

/// 2026-10-13: The `tool_choice="required"` Hermes grammar over the toy
/// vocab (as in `first_token_policy_tests.rs`). At its start the output
/// must open a tool call, so it cannot stop.
fn required_tool_grammar() -> GrammarState {
    let vocab = test_vocab();
    let mut engine = GrammarEngine::new(&vocab, &[TOY_EOS as i32]).unwrap();
    let compiled = engine
        .compile_hermes_tool_grammar(&test_tool_defs(), false)
        .unwrap();
    GrammarState::new(&compiled, engine.vocab_size())
        .unwrap()
        .with_stop_tokens(&[TOY_EOS])
}

#[test]
fn eos_inside_an_unclosed_think_ends_the_turn_as_reasoning() {
    // 2026-10-13: The GLM-5.3 garbage case. Before the fix the EOS was held
    // back here and the model ran on into "What is 2+2?...".
    for path in PATHS {
        let sched = sched();
        let (mut a, rx) = seq(vec![], true);
        feed(path, &mut a, ANSWER, &sched);
        assert!(!a.finished, "{path:?}: the answer token ended the turn");
        feed(path, &mut a, EOS[0], &sched);
        assert!(
            a.finished,
            "{path:?}: EOS inside <think> did not end the turn"
        );
        assert_eq!(a.output_tokens, vec![ANSWER, EOS[0]], "{path:?}");
        // 2026-10-13: vLLM counts every output id before `</think>`, the EOS
        // included, as reasoning.
        assert_eq!(a.thinking_tokens, 2, "{path:?}");

        finish_sequence(
            &SchedIo::for_test_with(Arc::new(PreemptStubModel::default())),
            &mut a,
            MAX_SEQ_LEN,
        );
        let mut rx = rx;
        let response = rx
            .try_recv()
            .expect("finish_sequence sends the blocking response")
            .expect("response is Ok");
        assert_eq!(response.finish_reason, "stop", "{path:?}");
        assert_eq!(response.reasoning_tokens, 2, "{path:?}");
        assert_eq!(response.output_tokens, vec![ANSWER, EOS[0]], "{path:?}");
    }
}

#[test]
fn eos_outside_thinking_ends_the_turn() {
    for path in PATHS {
        let sched = sched();
        let (mut a, _rx) = seq(vec![ANSWER], false);
        feed(path, &mut a, EOS[0], &sched);
        assert!(a.finished, "{path:?}");
        assert_eq!(a.output_tokens, vec![ANSWER, EOS[0]], "{path:?}");
        assert_eq!(a.thinking_tokens, 0, "{path:?}");
    }
}

#[test]
fn verify_verdict_with_eos_inside_thinking_ends_the_turn_at_the_eos() {
    // 2026-10-13: Full accept of drafts ["4", EOS]: the EOS ends the turn and
    // the bonus token after it is never emitted.
    let sched = sched();
    let model = PreemptStubModel::default();
    let (mut a, _rx) = seq(vec![], true);
    k4_apply_verdict(
        &model,
        &mut a,
        &sched,
        &[ANSWER, EOS[0]],
        &[ANSWER, EOS[0], 2000],
        vec![],
        2,
        2,
        K4Hidden::DeferPropose,
        0,
    );
    assert!(
        a.finished,
        "EOS accepted inside <think> did not end the turn"
    );
    assert_eq!(a.output_tokens, vec![ANSWER, EOS[0]]);
    assert_eq!(a.thinking_tokens, 2);
}

#[test]
fn grammar_does_not_hold_eos_inside_thinking() {
    // 2026-10-13: vLLM applies no grammar inside the reasoning block
    // (`enable_in_reasoning=False`), so an armed grammar cannot hold an EOS
    // there. The emit path used to hold it.
    for path in PATHS {
        let sched = sched();
        let (mut a, _rx) = seq(vec![], true);
        a.eos_tokens = vec![TOY_EOS];
        a.grammar_state = Some(required_tool_grammar());
        feed(path, &mut a, u32::from(b'4'), &sched);
        feed(path, &mut a, TOY_EOS, &sched);
        assert!(a.finished, "{path:?}: a grammar held EOS inside <think>");
        assert_eq!(a.output_tokens, vec![u32::from(b'4'), TOY_EOS], "{path:?}");
    }
}

#[test]
fn grammar_holds_eos_outside_thinking_on_the_decode_path() {
    // 2026-10-13: The control for the test above: the same grammar and EOS,
    // outside thinking, is held back. Only the decode path is checked:
    // `emit_token` advances the matcher before the hold and drops a grammar
    // that refuses the token, and its steps sample under the grammar mask,
    // so a forbidden EOS never reaches its hold.
    let sched = sched();
    let (mut a, _rx) = seq(vec![], false);
    a.eos_tokens = vec![TOY_EOS];
    a.grammar_state = Some(required_tool_grammar());
    feed(Path::Decode, &mut a, TOY_EOS, &sched);
    assert!(
        !a.finished,
        "a grammar that cannot stop let EOS end the turn"
    );
    assert!(a.output_tokens.is_empty(), "a held EOS was recorded");
}

#[test]
fn required_tool_call_holds_eos_only_outside_thinking() {
    // 2026-10-13: With no grammar, `require_tool_call` stands in for the
    // grammar vLLM attaches to a required call, so it follows the grammar's
    // rule: no hold inside thinking, a hold outside it.
    for path in PATHS {
        let sched = sched();
        let (mut a, _rx) = seq(vec![], true);
        a.require_tool_call = true;
        feed(path, &mut a, ANSWER, &sched);
        feed(path, &mut a, EOS[0], &sched);
        assert!(
            a.finished,
            "{path:?}: require_tool_call held EOS inside <think>"
        );

        let (mut a, _rx) = seq(vec![ANSWER], false);
        a.require_tool_call = true;
        feed(path, &mut a, EOS[0], &sched);
        assert!(
            !a.finished,
            "{path:?}: EOS ended a turn that owes a tool call"
        );
    }
}

#[test]
fn tool_armed_turn_ends_on_eos_right_after_think_close() {
    // 2026-10-13: The decode path used to hold EOS for 16 content tokens
    // after `</think>` on a tool-armed turn. vLLM has no such hold.
    for path in PATHS {
        let sched = sched();
        let (mut a, _rx) = seq(vec![ANSWER], true);
        a.tool_request = true;
        a.tools_present = true;
        feed(path, &mut a, THINK_END, &sched);
        assert!(
            !a.inside_thinking,
            "{path:?}: </think> did not close thinking"
        );
        assert!(!a.finished, "{path:?}");
        feed(path, &mut a, EOS[0], &sched);
        assert!(a.finished, "{path:?}: EOS after </think> was held back");
        assert_eq!(a.output_tokens.last(), Some(&EOS[0]), "{path:?}");
    }
}

#[test]
fn min_tokens_holds_eos_until_that_many_tokens_came_before_it() {
    // 2026-10-13: vLLM masks EOS while fewer than `min_tokens` tokens are
    // generated, inside thinking too. The emit path counted the EOS itself,
    // so it let an EOS through one token early.
    for path in PATHS {
        let sched = sched();
        let (mut a, _rx) = seq(vec![1000, 1001], true);
        a.min_tokens = 3;
        feed(path, &mut a, EOS[0], &sched);
        assert!(
            !a.finished,
            "{path:?}: EOS after 2 of 3 min_tokens ended the turn"
        );

        let (mut a, _rx) = seq(vec![1000, 1001, 1002], true);
        a.min_tokens = 3;
        feed(path, &mut a, EOS[0], &sched);
        assert!(
            a.finished,
            "{path:?}: EOS after 3 of 3 min_tokens was held back"
        );
    }
}
