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
        !json_eos_finishes("{\"answer\":4}", false, true),
        "old any-grammar classification reproduces the suppressed EOS"
    );
}

#[test]
fn incomplete_json_and_real_tool_requests_keep_their_guards() {
    assert!(!json_eos_finishes("{\"answer\":", false, false));
    assert!(!json_eos_finishes("{\"answer\":4}", true, false));
    assert!(tool_request_at_birth(false, false, true));
    assert!(!tool_request_at_birth(false, true, false));
}

#[test]
fn legacy_minimum_and_sticky_tool_guards_remain_armed() {
    assert!(!json_eos_state("{}", false, false, true, 0, true).finished);
    assert!(!json_eos_state("{}", false, false, false, 8, false).finished);
    let dropped = json_eos_state("{}", true, false, false, 0, true);
    assert!(dropped.tool_request);
    assert!(!dropped.finished);
}
