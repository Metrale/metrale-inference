// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: the MTP bootstrap token (`bootstrap_seq`, and the batched
//! `step_mtp_bootstrap_batched`) is the token the serial decode path
//! (`process_seq_logits`) picks from the same row: no grammar inside thinking,
//! the forced `</think>`, and the request's `logit_bias`. Each case has a
//! negative twin where both paths keep the unchanged pick.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::serial_bootstrap::bootstrap_seq;
use crate::scheduler::spec_pipeline_fixture::*;
use crate::scheduler::test_support::{PreemptStubModel, bf16_bytes};
use crate::scheduler::types::ActiveSeq;

/// 2026-10-21: the decode path's pick for `r` from `a`'s state.
fn serial_pick(r: &[f32], mut a: ActiveSeq) -> u32 {
    let s = sched(std::sync::Arc::new(PreemptStubModel::default()), None);
    let c = ctx(&s);
    let m = PreemptStubModel::default();
    crate::scheduler::process_seq_logits(&m, &mut a, &bf16_bytes(r), 0, VOCAB, 2, false, &c, false)
        .0
}

/// 2026-10-21: one serial MTP bootstrap of `a` over the decode row `r`;
/// returns the emitted token and the row copies the pick made.
fn bootstrap(r: Vec<f32>, a: &mut ActiveSeq) -> (u32, usize) {
    let m = model(&[r]);
    let s = sched(m.clone(), None);
    let c = ctx(&s);
    bootstrap_seq(&*m, a, &s, 1, 1, &c, false);
    (*a.output_tokens.last().expect("a token was emitted"), row_copies(&m))
}

fn thinking_grammar_seq() -> ActiveSeq {
    let mut a = thinking_seq();
    a.grammar_state = Some(required_tool_grammar());
    a
}

#[test]
fn bootstrap_inside_thinking_ignores_the_grammar() {
    let r = row(&[(HELLO, 10.0), (TOOL_CALL, 5.0)]);
    assert_eq!(serial_pick(&r, thinking_grammar_seq()), HELLO);
    let (tok, _) = bootstrap(r, &mut thinking_grammar_seq());
    assert_eq!(tok, HELLO, "reasoning is not constrained by the tool grammar");
}

#[test]
fn bootstrap_outside_thinking_keeps_the_grammar() {
    let r = row(&[(HELLO, 10.0), (TOOL_CALL, 5.0)]);
    let mut a = seq();
    a.grammar_state = Some(required_tool_grammar());
    let mut b = seq();
    b.grammar_state = Some(required_tool_grammar());
    assert_eq!(serial_pick(&r, a), TOOL_CALL);
    assert_eq!(bootstrap(r, &mut b).0, TOOL_CALL);
}

#[test]
fn bootstrap_honours_an_armed_forced_think_end() {
    let r = row(&[(HELLO, 10.0)]);
    let armed = || {
        let mut a = thinking_seq();
        a.force_end_thinking = true;
        a
    };
    assert_eq!(serial_pick(&r, armed()), THINK_END);
    assert_eq!(bootstrap(r, &mut armed()).0, THINK_END);
}

#[test]
fn bootstrap_inside_thinking_unarmed_keeps_the_argmax_on_the_fast_path() {
    let r = row(&[(HELLO, 10.0)]);
    assert_eq!(serial_pick(&r, thinking_seq()), HELLO);
    assert_eq!(bootstrap(r, &mut thinking_seq()), (HELLO, 0));
}

fn biased_seq() -> ActiveSeq {
    let mut a = seq();
    a.logit_bias = vec![(TOOL_CALL, 3.0)];
    a
}

#[test]
fn bootstrap_applies_the_request_bias() {
    let r = row(&[(HELLO, 10.0), (TOOL_CALL, 8.0)]);
    assert_eq!(serial_pick(&r, biased_seq()), TOOL_CALL);
    assert_eq!(bootstrap(r, &mut biased_seq()).0, TOOL_CALL);
}

#[test]
fn bootstrap_bias_that_cannot_flip_keeps_the_fast_path() {
    let r = row(&[(HELLO, 10.0), (TOOL_CALL, 6.0)]);
    assert_eq!(serial_pick(&r, biased_seq()), HELLO);
    assert_eq!(bootstrap(r, &mut biased_seq()), (HELLO, 0));
}

#[test]
fn batched_bootstrap_checks_the_batched_argmax_against_the_pipeline() {
    // 2026-10-21: two greedy rows, so `argmax_batch` runs: row 0 is thinking
    // with the forced `</think>` armed, row 1 is plain prose.
    let rows = [row(&[(HELLO, 10.0)]), row(&[(HELLO, 10.0)])];
    let m = model(&rows);
    let s = sched(m.clone(), None);
    let c = ctx(&s);
    let mut armed = thinking_seq();
    armed.force_end_thinking = true;
    let mut plain = seq();
    plain.seq.slot_idx = 1;
    let mut active = vec![armed, plain];
    crate::scheduler::mtp_bootstrap_step::step_mtp_bootstrap_batched(&*m, &mut active, &s, &[0, 1], 1, &c);
    assert_eq!(active[0].output_tokens.last(), Some(&THINK_END));
    assert_eq!(active[1].output_tokens.last(), Some(&HELLO));
    assert_eq!(row_copies(&m), 1, "only the armed row went to the host");
}
