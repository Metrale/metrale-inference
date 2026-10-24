// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: PELT locates a planted warm-up shift and finds nothing in stationary noise; the
//! steady-state rule trims the warm-up and leaves a late shift (a drift, not warm-up) alone.

use alloc::vec::Vec;

use super::*;
use crate::rng::SplitMix64;

fn warmup(n_warm: usize, n: usize, seed: u64) -> Vec<f64> {
    let mut r = SplitMix64::new(seed);
    (0..n)
        .map(|i| if i < n_warm { 30.0 } else { 44.0 } + r.normal())
        .collect()
}

#[test]
fn pelt_finds_a_planted_shift_within_a_few_samples() {
    let xs = warmup(40, 400, 71);
    let cps = pelt_mean(&xs, bic_penalty(xs.len()));
    assert_eq!(cps.len(), 1, "{cps:?}");
    assert!(cps[0].abs_diff(40) <= 2, "{cps:?}");
    assert!(cusum(&xs, 0.5, 5.0).is_some());
}

#[test]
fn stationary_noise_has_no_change() {
    let mut r = SplitMix64::new(72);
    for _ in 0..20 {
        let xs: Vec<f64> = (0..400).map(|_| 12.5 + 0.7 * r.normal()).collect();
        assert!(pelt_mean(&xs, bic_penalty(400)).is_empty());
    }
}

#[test]
fn steady_start_trims_warm_up_only() {
    let power = warmup(60, 480, 73);
    let mut r = SplitMix64::new(74);
    let clock: Vec<f64> = (0..480)
        .map(|i| if i < 25 { 1700.0 } else { 2435.0 } + 3.0 * r.normal())
        .collect();
    let s = steady_start(&[&power, &clock]);
    assert!(s.abs_diff(60) <= 2, "{s}");
    // 2026-10-22: A shift in the second half is not warm-up.
    let late = warmup(400, 480, 75);
    assert_eq!(steady_start(&[&late]), 0);
}
