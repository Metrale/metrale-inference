// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: the verify window (`verify_pick_all_with_pipeline`, the entry
//! every speculative verify step calls) picks what the serial decode path
//! (`process_seq_logits` then `process_decoded_token`, one token at a time)
//! picks from the same logits rows: the request's `logit_bias`, a GLM
//! `<tool_call>` that closes thinking inside the window, and a thinking budget
//! reached inside the window. Each case has a negative twin where both paths
//! must agree on the unchanged pick.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::per_token::process_decoded_token;
use crate::scheduler::sched_ctx::SchedCtx;
use crate::scheduler::spec_pipeline_fixture::*;
use crate::scheduler::test_support::{PreemptStubModel, bf16_bytes};
use crate::scheduler::types::ActiveSeq;
use metrale_model_engine::traits::ModelLogits;
use std::sync::Arc;

/// 2026-10-21: the serial decode path over `rows`, one decode step per row.
fn serial_picks(rows: &[Vec<f32>], a: &mut ActiveSeq, s: &SchedCtx) -> Vec<u32> {
    let m = PreemptStubModel::default();
    let mut picks = Vec::new();
    for r in rows {
        let buf = bf16_bytes(r);
        let c = ctx(s);
        let (tok, _) =
            crate::scheduler::process_seq_logits(&m, a, &buf, 0, VOCAB, 2, false, &c, false);
        process_decoded_token(
            a,
            tok,
            None,
            std::time::Instant::now(),
            Some(THINK_END),
            Some(THINK_START),
            None,
            Some(TOOL_CALL),
            Some(TOOL_CALL_END),
            &m,
            s,
        );
        picks.push(tok);
    }
    picks
}

/// 2026-10-21: one verify window over `rows`, with the device argmax of each
/// row as the fast paths see it, and the sequence state it leaves.
fn window_picks(
    rows: &[Vec<f32>],
    a: &mut ActiveSeq,
    closed_by: Option<u32>,
    masked_verify: bool,
) -> (Vec<u32>, Arc<PreemptStubModel>) {
    let m = model(rows);
    let s = sched(m.clone(), closed_by);
    let mut c = ctx(&s);
    c.sampling.dflash_masked_verify = masked_verify;
    let base = m.logits_buffer_ptr();
    let argmax: Vec<u32> = (0..rows.len())
        .map(|i| m.argmax_on_device(base.offset(i * VOCAB * 2), 0).unwrap())
        .collect();
    let picks = crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
        &*m, &argmax, a, &c, 0,
    );
    (picks, m)
}

fn serial_sched(closed_by: Option<u32>) -> SchedCtx {
    sched(Arc::new(PreemptStubModel::default()), closed_by)
}

// ---- 1. the request's logit_bias -------------------------------------------

/// 2026-10-21: a tools-active request's `<tool_call>` +3.0 bias, outside
/// thinking and outside a tool body, no grammar.
fn biased_seq() -> ActiveSeq {
    let mut a = seq();
    a.logit_bias = vec![(TOOL_CALL, 3.0)];
    a
}

#[test]
fn bias_flips_a_low_margin_pick_in_the_window_as_on_the_decode_path() {
    // 2026-10-21: HELLO leads `<tool_call>` by 2.0 < 3.0 at position 0.
    let rows = [row(&[(HELLO, 10.0), (TOOL_CALL, 8.0)]), row(&[(HELLO, 10.0)])];
    let serial = serial_picks(&rows[..1], &mut biased_seq(), &serial_sched(None));
    assert_eq!(serial, vec![TOOL_CALL], "the decode path applies the bias");
    let (picks, m) = window_picks(&rows, &mut biased_seq(), None, false);
    assert_eq!(picks[0], TOOL_CALL, "the window applies the request's bias");
    assert_eq!(row_copies(&m), 1, "the chat fast path declined; the host picked");
}

#[test]
fn bias_flips_the_host_pick_with_every_fast_path_off() {
    let rows = [row(&[(HELLO, 10.0), (TOOL_CALL, 8.0)])];
    let m = model(&rows);
    let s = sched(m.clone(), None);
    let mut c = ctx(&s);
    c.sampling.fast_greedy_chat = false;
    let picks = crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
        &*m,
        &[HELLO],
        &mut biased_seq(),
        &c,
        0,
    );
    assert_eq!(picks, vec![TOOL_CALL], "verify_pick_with_pipeline applies the bias");
}

#[test]
fn bias_that_cannot_flip_keeps_the_fast_path() {
    // 2026-10-21: HELLO leads by 4.0 > 3.0: both paths keep HELLO, and the
    // window takes it from the device argmax with no row copy.
    let rows = [row(&[(HELLO, 10.0), (TOOL_CALL, 6.0)]), row(&[(HELLO, 10.0)])];
    let serial = serial_picks(&rows, &mut biased_seq(), &serial_sched(None));
    let (picks, m) = window_picks(&rows, &mut biased_seq(), None, false);
    assert_eq!(serial, vec![HELLO, HELLO]);
    assert_eq!(picks, serial);
    assert_eq!(row_copies(&m), 0, "the bias check kept the fast path");
}

#[test]
fn bias_rules_out_the_device_verdict() {
    // 2026-10-21: the device verdict is armed before any logit exists, so it
    // cannot check the bias; a biased request must not arm it.
    let s = serial_sched(None);
    let c = ctx(&s);
    assert!(
        crate::scheduler::verify_k2_device_step::identity_picks(&seq(), &c),
        "control: a plain greedy request arms it"
    );
    assert!(!crate::scheduler::verify_k2_device_step::identity_picks(
        &biased_seq(),
        &c
    ));
}

// ---- 2. a GLM `<tool_call>` that closes thinking in the window -------------

fn tool_call_rows() -> [Vec<f32>; 2] {
    // 2026-10-21: position 0 writes the call; position 1's argmax is prose the
    // grammar refuses after `<tool_call>`.
    [
        row(&[(TOOL_CALL, 10.0), (HELLO, 5.0)]),
        row(&[(HELLO, 10.0), (first_id_after_tool_call(), 5.0)]),
    ]
}

fn thinking_grammar_seq() -> ActiveSeq {
    let mut a = thinking_seq();
    a.grammar_state = Some(required_tool_grammar());
    a
}

#[test]
fn tool_call_closing_thinking_in_the_window_constrains_the_next_position() {
    let rows = tool_call_rows();
    let serial = serial_picks(&rows, &mut thinking_grammar_seq(), &serial_sched(Some(TOOL_CALL)));
    assert_eq!(serial, vec![TOOL_CALL, first_id_after_tool_call()]);
    let mut a = thinking_grammar_seq();
    let (picks, _) = window_picks(&rows, &mut a, Some(TOOL_CALL), false);
    assert_eq!(picks, serial, "the window closes thinking at the opener");
    assert!(a.inside_thinking && !a.think_ended, "thinking state restored");
    assert_eq!(
        a.grammar_state.as_ref().unwrap().num_history_steps(),
        0,
        "the opener's speculative advance rolled back"
    );
}

#[test]
fn tool_call_that_does_not_close_thinking_stays_masked() {
    // 2026-10-21: without `thinking_closed_by` the opener is masked inside
    // thinking, and both positions stay unconstrained reasoning.
    let rows = tool_call_rows();
    let serial = serial_picks(&rows, &mut thinking_grammar_seq(), &serial_sched(None));
    let (picks, _) = window_picks(&rows, &mut thinking_grammar_seq(), None, false);
    assert_eq!(serial, vec![HELLO, HELLO]);
    assert_eq!(picks, serial);
}

// ---- 3. a thinking budget reached inside the window ------------------------

fn budget_seq(budget: u32) -> ActiveSeq {
    let mut a = thinking_seq();
    a.thinking_tokens = 100;
    a.thinking_budget = Some(budget);
    a
}

const BUDGET_ROWS: fn() -> [Vec<f32>; 2] = || [row(&[(HELLO, 10.0)]), row(&[(HELLO, 10.0)])];

#[test]
fn budget_reached_at_position_0_forces_think_end_at_position_1() {
    let rows = BUDGET_ROWS();
    let serial = serial_picks(&rows, &mut budget_seq(101), &serial_sched(None));
    assert_eq!(serial, vec![HELLO, THINK_END]);
    let mut a = budget_seq(101);
    let (picks, _) = window_picks(&rows, &mut a, None, false);
    assert_eq!(picks, serial, "the window counts its thinking picks");
    assert_eq!(
        (a.thinking_tokens, a.force_end_thinking),
        (100, false),
        "emit_token re-counts the emitted tokens"
    );
}

#[test]
fn budget_reached_in_the_window_declines_the_masked_fast_path() {
    let mut a = budget_seq(101);
    let (picks, m) = window_picks(&BUDGET_ROWS(), &mut a, None, true);
    assert_eq!(picks, vec![HELLO, THINK_END]);
    assert_eq!(row_copies(&m), 1, "fast_masked declined; the host picked");
}

#[test]
fn budget_beyond_the_window_changes_nothing() {
    let rows = BUDGET_ROWS();
    let serial = serial_picks(&rows, &mut budget_seq(102), &serial_sched(None));
    assert_eq!(serial, vec![HELLO, HELLO]);
    let (picks, _) = window_picks(&rows, &mut budget_seq(102), None, false);
    assert_eq!(picks, serial);
    let (fast, m) = window_picks(&rows, &mut budget_seq(102), None, true);
    assert_eq!(fast, serial);
    assert_eq!(row_copies(&m), 0, "fast_masked still fires");
}
