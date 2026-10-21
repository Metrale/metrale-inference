// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: [`super::screen_picks_ok`] inside thinking, on the watchdog a model's MODEL.toml
//! `[behavior]` gives (`WatchdogParams::from_behavior`): a model without thinking steering
//! (GLM-5.3) takes the screen on a long thinking step, one with steering declines it once its
//! confidence stop may act, and an armed close or a budget the window reaches declines it for
//! both.

use super::screen_picks_ok;
use crate::scheduler::WatchdogParams;
use crate::scheduler::spec_pipeline_fixture::{ctx, model, sched, thinking_seq};

/// 2026-10-21: The screen decision for a thinking step `thinking_tokens` deep under a model
/// whose `[behavior]` has `thinking_steering = steering` (and the confidence stop asked for).
fn decide(
    steering: bool,
    thinking_tokens: u32,
    tune: impl Fn(&mut crate::scheduler::ActiveSeq),
) -> bool {
    let mut s = sched(model(&[]), None);
    s.watchdog = WatchdogParams::from_behavior(
        &metrale_kernels::ModelBehavior {
            thinking_steering: steering,
            confidence_early_stop: true,
            ..metrale_kernels::ModelBehavior::default()
        },
        None,
        None,
    );
    let mut a = thinking_seq();
    a.thinking_tokens = thinking_tokens;
    tune(&mut a);
    let mut c = ctx(&s);
    // 2026-10-21: The serve's default (`METRALE_DFLASH_MASKED_VERIFY` on); the test levers
    // leave it off.
    c.sampling.dflash_masked_verify = true;
    screen_picks_ok(&a, &c)
}

/// 2026-10-21: Without steering no stage can act on a thinking step, however long the span:
/// the confidence stop is off and no close is armed (past the 2,048-token defer ceiling too).
#[test]
fn a_thinking_step_without_steering_takes_the_screen() {
    for tokens in [0, 399, 400, 2047, 2048, 20_000] {
        assert!(
            decide(false, tokens, |_| {}),
            "steering off, {tokens} thinking tokens"
        );
    }
}

/// 2026-10-21: With steering the confidence stop may arm the close from 400 thinking tokens on,
/// so the step declines; below that it takes the screen.
#[test]
fn a_thinking_step_with_steering_declines_once_the_confidence_stop_may_act() {
    assert!(decide(true, 399, |_| {}));
    for tokens in [400, 2048, 20_000] {
        assert!(
            !decide(true, tokens, |_| {}),
            "steering on, {tokens} thinking tokens"
        );
    }
}

/// 2026-10-21: An armed close, or a thinking budget the 2-row window reaches, declines it with
/// or without steering.
#[test]
fn an_armed_close_or_a_reached_budget_declines_the_screen() {
    for steering in [false, true] {
        assert!(
            !decide(steering, 10, |a| a.force_end_thinking = true),
            "armed, {steering}"
        );
        assert!(
            !decide(steering, 10, |a| a.thinking_budget = Some(11)),
            "budget, {steering}"
        );
        assert!(
            decide(steering, 10, |a| a.thinking_budget = Some(12)),
            "budget ahead, {steering}"
        );
    }
}
