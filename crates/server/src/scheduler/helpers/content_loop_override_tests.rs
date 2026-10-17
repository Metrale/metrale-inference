// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the content-loop watchdog's repeat-threshold overrides (operator
//! and per-request). Its arming precedence is a settings resolution (`serve_settings`).
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::{
    CONTENT_LOOP_MIN_REPEATS, CONTENT_LOOP_PERIOD_MAX, CONTENT_LOOP_PERIOD_MIN, WatchdogParams,
    detect_content_token_loop_with,
};
use crate::api::inference_types::RepetitionDetectionParams;

#[test]
fn min_repeats_cli_reaches_the_resolved_params() {
    let p =
        WatchdogParams::from_behavior(&metrale_kernels::ModelBehavior::default(), None, Some(5));
    assert_eq!(p.content_loop_min_repeats, Some(5));
    let eff = p.content_loop_params(None).expect("override present");
    assert_eq!(eff.min_count, 5);
    assert_eq!(eff.min_pattern_size as usize, CONTENT_LOOP_PERIOD_MIN);
    assert_eq!(eff.max_pattern_size as usize, CONTENT_LOOP_PERIOD_MAX);
}

#[test]
fn request_params_outrank_the_operator_override() {
    let p = WatchdogParams {
        content_loop_min_repeats: Some(5),
        ..WatchdogParams::default()
    };
    let req = RepetitionDetectionParams {
        min_pattern_size: 4,
        max_pattern_size: 8,
        min_count: 2,
    };
    let eff = p.content_loop_params(Some(req)).expect("request present");
    assert_eq!(eff.min_count, 2);
    assert_eq!(eff.min_pattern_size, 4);
}

#[test]
fn unset_override_keeps_the_historical_constants() {
    let p = WatchdogParams::from_behavior(&metrale_kernels::ModelBehavior::default(), None, None);
    assert_eq!(p.content_loop_min_repeats, None);
    assert!(p.content_loop_params(None).is_none());
}

#[test]
fn raised_min_repeats_passes_code_shaped_period_2_tails() {
    // 2026-09-25: 48 distinct tokens, then a period-2 tail repeated 3 times.
    // The built-in threshold (3) fires on it, a threshold of 5 does not, and
    // 5 copies of the tail trip the threshold of 5.
    let mut toks: Vec<u32> = (0..48u32).collect();
    toks.extend_from_slice(&[7, 8, 7, 8, 7, 8]);
    assert!(detect_content_token_loop_with(&toks, None));
    const { assert!(CONTENT_LOOP_MIN_REPEATS == 3) };
    let relaxed = RepetitionDetectionParams {
        min_pattern_size: CONTENT_LOOP_PERIOD_MIN as u32,
        max_pattern_size: CONTENT_LOOP_PERIOD_MAX as u32,
        min_count: 5,
    };
    assert!(!detect_content_token_loop_with(&toks, Some(relaxed)));
    toks.extend_from_slice(&[7, 8, 7, 8]);
    assert!(detect_content_token_loop_with(&toks, Some(relaxed)));
}
