// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Real grammar/emission regressions for short JSON after </think>.
use super::per_token::process_decoded_token;
use crate::grammar::tests::test_vocab;
use crate::grammar::{GrammarEngine, GrammarState};
use crate::scheduler::first_token_policy::tool_request_at_birth;
use crate::scheduler::sched_ctx::SchedCtx;
use crate::scheduler::test_support::{PreemptStubModel, test_seq};

fn json_eos_state(
    text: &str,
    tools: bool,
    old_classification: bool,
    legacy: bool,
    min_tokens: usize,
    drop_grammar: bool,
) -> crate::scheduler::types::ActiveSeq {
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let compiled = engine.compile_json_grammar().unwrap();
    let mut gs = GrammarState::new(&compiled, 131)
        .unwrap()
        .with_stop_tokens(&[130]);
    let tokens: Vec<_> = text.bytes().map(u32::from).collect();
    for &token in &tokens {
        assert!(gs.accept_token(token));
    }
    let (mut a, _rx) = test_seq(tokens, 100, None, 50);
    a.finished = false;
    a.inside_thinking = false;
    a.think_ended = true;
    a.thinking_tokens = 0;
    a.min_tokens = min_tokens;
    a.eos_tokens = vec![130];
    a.require_tool_call = legacy;
    a.tools_present = tools;
    a.tool_request = old_classification || tool_request_at_birth(true, tools, legacy);
    a.grammar_state = if drop_grammar { None } else { Some(gs) };
    process_decoded_token(
        &mut a,
        130,
        None,
        std::time::Instant::now(),
        None,
        None,
        None,
        None,
        None,
        &PreemptStubModel::default(),
        &SchedCtx::for_test(),
    );
    a
}

fn json_eos_finishes(text: &str, tools: bool, old: bool) -> bool {
    let state = json_eos_state(text, tools, old, false, 0, false);
    assert!(
        state.remaining > 0,
        "fixture must not end by budget exhaustion"
    );
    state.finished
}

#[test]
fn short_complete_json_stops_without_consuming_the_remaining_budget() {
    assert!(json_eos_finishes("{\"answer\":4}", false, false));
    assert!(json_eos_finishes("{}", false, false));
    assert!(
        json_eos_finishes("{\"answer\":4}", false, true),
        "sticky classification alone must not suppress a grammar-legal EOS"
    );
}

#[test]
fn incomplete_json_keeps_its_guard_while_optional_tools_allow_eos() {
    assert!(!json_eos_finishes("{\"answer\":", false, false));
    assert!(json_eos_finishes("{\"answer\":4}", true, false));
    assert!(tool_request_at_birth(false, false, true));
    assert!(!tool_request_at_birth(false, true, false));
}

#[test]
fn legacy_minimum_guards_and_sticky_tool_classification_remain_armed() {
    assert!(!json_eos_state("{}", false, false, true, 0, true).finished);
    assert!(!json_eos_state("{}", false, false, false, 8, false).finished);
    let dropped = json_eos_state("{}", true, false, false, 0, true);
    assert!(dropped.tool_request);
    assert!(dropped.finished);
}

#[test]
fn optional_tools_short_answer_eos_finishes() {
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let compiled = engine
        .compile_hermes_tool_grammar(&crate::grammar::tests::test_tool_defs(), true)
        .unwrap();
    let mut grammar = GrammarState::new(&compiled, engine.vocab_size())
        .unwrap()
        .with_stop_tokens(&[130]);
    assert!(grammar.accept_token(u32::from(b'7')));
    let (mut a, _rx) = test_seq(vec![u32::from(b'7')], 255, None, 224);
    a.finished = false;
    a.inside_thinking = false;
    a.think_ended = true;
    a.min_tokens = 0;
    a.eos_tokens = vec![130];
    a.tools_present = true;
    a.tool_request = tool_request_at_birth(true, true, false);
    a.require_tool_call = false;
    a.grammar_state = Some(grammar);
    process_decoded_token(
        &mut a,
        130,
        None,
        std::time::Instant::now(),
        None,
        None,
        None,
        Some(128),
        Some(129),
        &PreemptStubModel::default(),
        &SchedCtx::for_test(),
    );
    assert!(
        a.finished,
        "optional tools must allow a short answer to end at EOS"
    );
    assert!(
        a.remaining > 0,
        "EOS must finish before exhausting the budget"
    );
    assert_eq!(a.output_tokens, [u32::from(b'7'), 130]);
    assert!(
        a.tool_request,
        "the prose watchdog's sticky tool flag stays armed"
    );
    assert_eq!(
        crate::scheduler::lifecycle::derive_finish_reason(
            a.guard_stop,
            a.output_tokens.last().copied(),
            &a.eos_tokens,
            a.tool_call_end_token,
            a.remaining,
            a.seq.seq_len,
            4096,
        ),
        "stop"
    );
}
