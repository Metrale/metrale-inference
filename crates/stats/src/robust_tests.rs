// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: On a contaminated sample the robust estimators stay near the truth while the mean
//! does not; the outlier rule flags exactly the planted values and nothing on clean data.

use alloc::vec::Vec;

use super::*;
use crate::rng::SplitMix64;

fn sample(n: usize, seed: u64) -> Vec<f64> {
    let mut r = SplitMix64::new(seed);
    (0..n).map(|_| 10.0 + r.normal()).collect()
}

#[test]
fn robust_estimators_resist_contamination_and_the_mean_does_not() {
    let mut xs = sample(200, 1);
    for x in xs.iter_mut().take(10) {
        *x = 1000.0;
    }
    let mean = xs.iter().sum::<f64>() / 200.0;
    assert!(mean > 50.0, "the contamination must move the mean: {mean}");
    assert!((median(&xs).unwrap() - 10.0).abs() < 0.3);
    assert!((hodges_lehmann(&xs).unwrap() - 10.0).abs() < 0.4);
    assert!((huber(&xs, HUBER_K).unwrap() - 10.0).abs() < 0.3);
}

#[test]
fn estimators_agree_with_hand_values() {
    assert_eq!(median(&[3.0, 1.0, 2.0]).unwrap(), 2.0);
    assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]).unwrap(), 2.5);
    assert_eq!(mad(&[1.0, 2.0, 3.0, 4.0, 100.0]).unwrap(), 1.0);
    // 2026-10-22: Walsh averages of {0, 1, 5}, sorted: 0, .5, 1, 2.5, 3, 5; median 1.75.
    assert_eq!(hodges_lehmann(&[0.0, 1.0, 5.0]).unwrap(), 1.75);
}

#[test]
fn the_rule_flags_planted_outliers_only() {
    let mut xs = sample(300, 2);
    assert!(flag_outliers(&xs, &OutlierRule::DECLARED).is_empty());
    xs[7] = 30.0;
    xs[123] = -15.0;
    assert_eq!(flag_outliers(&xs, &OutlierRule::DECLARED), [7, 123]);
    assert_eq!(xs.len(), 300, "flagging never drops data");
}

#[test]
fn a_zero_mad_falls_back_to_the_mean_deviation() {
    let xs = [5.0, 5.0, 5.0, 5.0, 5.0, 9.0];
    assert_eq!(flag_outliers(&xs, &OutlierRule::DECLARED), [5]);
    assert!(flag_outliers(&[2.0; 4], &OutlierRule::DECLARED).is_empty());
}

#[test]
fn empty_or_non_finite_input_is_refused() {
    use metrale_error::ErrorCode;
    assert_eq!(
        median(&[]).unwrap_err().code(),
        ErrorCode::StatsInsufficient
    );
    assert_eq!(
        huber(&[1.0, f64::NAN], HUBER_K).unwrap_err().code(),
        ErrorCode::StatsInvalid
    );
}
