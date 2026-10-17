// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The think-block stages with `WatchdogParams::thinking_steering`
//! on and off. Off is what vLLM does: `</think>` and `<think>` are never
//! masked, and an armed close is injected at once.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::forced_think_end::ForcedThinkEndInjector;
use super::mid_word::MidWordThinkEndMask;
use super::post_close::PostCloseThinkMask;
use super::{LogitsContext, LogitsProcessor, SamplingLevers};
use crate::scheduler::helpers::WatchdogParams;
use crate::scheduler::test_support::test_seq;
use crate::scheduler::types::ActiveSeq;

const THINK_START: u32 = 5;
const THINK_END: u32 = 6;
/// 2026-10-14: A token whose text ends mid-word (`mid_word_mask`) and is not
/// a sentence boundary (`boundary_mask`).
const MID_WORD: u32 = 3;
const VOCAB: usize = 8;

fn run(stage: &dyn LogitsProcessor, steering: bool, a: &mut ActiveSeq) -> Vec<f32> {
    let scratch = crate::scheduler::sched_ctx::DecodeScratch::default();
    let io = crate::scheduler::io::SchedIo::for_test();
    let mut mid_word = vec![false; VOCAB];
    mid_word[MID_WORD as usize] = true;
    let ctx = LogitsContext {
        scratch: &scratch,
        tel: &*io.tel,
        clock: &*io.clock,
        watchdog: WatchdogParams {
            thinking_steering: steering,
            ..WatchdogParams::default()
        },
        boundary_mask: Some(vec![false; VOCAB].into()),
        mid_word_mask: Some(mid_word.into()),
        sampling: SamplingLevers::default(),
        think_end_token: Some(THINK_END),
        think_start_token: Some(THINK_START),
        tool_call_start_token: None,
        tool_call_end_token: None,
        thinking_closed_by: None,
        verify_pos: 0,
    };
    let mut logits = vec![1.0f32; VOCAB];
    stage.apply(&mut logits, a, &ctx);
    logits
}

/// 2026-10-14: Inside thinking, the last token mid-word.
fn thinking_seq() -> ActiveSeq {
    let (mut a, _rx) = test_seq(vec![MID_WORD], 1000, None, 10);
    a.finished = false;
    a.inside_thinking = true;
    a.think_start_token = Some(THINK_START);
    a.think_end_token = Some(THINK_END);
    a
}

#[test]
fn mid_word_close_is_masked_only_with_steering() {
    let on = run(&MidWordThinkEndMask, true, &mut thinking_seq());
    assert_eq!(on[THINK_END as usize], f32::NEG_INFINITY);
    let off = run(&MidWordThinkEndMask, false, &mut thinking_seq());
    assert_eq!(off[THINK_END as usize], 1.0);
}

#[test]
fn after_the_close_both_markers_stay_sampleable_without_steering() {
    let closed = || {
        let mut a = thinking_seq();
        a.inside_thinking = false;
        a.think_ended = true;
        a
    };
    let on = run(&PostCloseThinkMask, true, &mut closed());
    assert_eq!(on[THINK_END as usize], f32::NEG_INFINITY);
    assert_eq!(on[THINK_START as usize], f32::NEG_INFINITY);
    let off = run(&PostCloseThinkMask, false, &mut closed());
    assert_eq!(off[THINK_END as usize], 1.0);
    assert_eq!(off[THINK_START as usize], 1.0);
}

#[test]
fn an_armed_close_mid_word_defers_with_steering_and_injects_at_once_without() {
    let armed = || {
        let mut a = thinking_seq();
        a.force_end_thinking = true;
        a.thinking_budget = Some(100);
        a.thinking_tokens = 100;
        a
    };
    let mut a_on = armed();
    let on = run(&ForcedThinkEndInjector, true, &mut a_on);
    assert_eq!(
        on[THINK_END as usize], 1.0,
        "deferred to a sentence boundary"
    );
    assert_eq!(a_on.sentence_defer_count, 1);

    let mut a_off = armed();
    let off = run(&ForcedThinkEndInjector, false, &mut a_off);
    assert_eq!(off[THINK_END as usize], 0.0);
    assert!(
        off.iter()
            .enumerate()
            .all(|(i, &l)| i == THINK_END as usize || l == f32::NEG_INFINITY),
        "`</think>` is the only finite logit"
    );
}

#[test]
fn without_steering_nothing_is_injected_until_a_budget_arms_the_close() {
    let mut a = thinking_seq();
    a.thinking_tokens = 1_000_000;
    let off = run(&ForcedThinkEndInjector, false, &mut a);
    assert!(off.iter().all(|&l| l == 1.0));
}
