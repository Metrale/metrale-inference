// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Tests that MODEL.toml `[behavior].thinking_steering` reaches
//! every thinking watchdog through `WatchdogParams::from_behavior`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::WatchdogParams;

fn params(thinking_steering: bool) -> WatchdogParams {
    let b = metrale_kernel_set::ModelBehavior {
        thinking_steering,
        confidence_early_stop: true,
        enable_think_loop_watchdog: true,
        min_reasoning_floor_tokens: 16,
        ..metrale_kernel_set::ModelBehavior::default()
    };
    WatchdogParams::from_behavior(&b, None, None)
}

#[test]
fn steering_off_disarms_the_watchdogs_that_force_or_bias_the_close() {
    let p = params(false);
    assert!(!p.thinking_steering);
    assert!(!p.confidence_early_stop);
    assert!(!p.enable_think_loop_watchdog);
    assert_eq!(p.min_reasoning_floor, 0);
}

#[test]
fn steering_on_keeps_the_model_toml_values() {
    let p = params(true);
    assert!(p.thinking_steering);
    assert!(p.confidence_early_stop);
    assert!(p.enable_think_loop_watchdog);
    assert_eq!(p.min_reasoning_floor, 16);
}
