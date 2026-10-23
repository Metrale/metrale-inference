// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: With error in both instruments Deming recovers the true slope where OLS is
//! attenuated; Bland-Altman returns the constructed bias and spread.

use alloc::vec::Vec;

use super::*;
use crate::rng::SplitMix64;

#[test]
fn deming_recovers_the_slope_ols_attenuates() {
    let mut r = SplitMix64::new(51);
    let truth: Vec<f64> = (0..400).map(|_| 30.0 + 10.0 * r.normal()).collect();
    let x: Vec<f64> = truth.iter().map(|t| t + 5.0 * r.normal()).collect();
    let y: Vec<f64> = truth
        .iter()
        .map(|t| 2.0 + 1.3 * t + 5.0 * r.normal())
        .collect();
    let d = deming(&x, &y, 1.0).unwrap();
    assert!((d.slope - 1.3).abs() < 0.08, "deming slope {}", d.slope);
    assert!(d.slope_ci.contains(1.3));
    let (mx, my) = (mean(&x), mean(&y));
    let ols = x
        .iter()
        .zip(&y)
        .map(|(a, b)| (a - mx) * (b - my))
        .sum::<f64>()
        / x.iter().map(|a| (a - mx) * (a - mx)).sum::<f64>();
    assert!(ols < 1.15, "OLS must be attenuated: {ols}");
    assert!(!d.slope_ci.contains(ols));
}

#[test]
fn bland_altman_returns_the_constructed_bias_and_limits() {
    let x = [10.0, 20.0, 30.0, 40.0, 50.0];
    let y = [11.0, 23.0, 31.0, 43.0, 52.0];
    let b = bland_altman(&x, &y, 0.95).unwrap();
    // 2026-10-22: differences 1, 3, 1, 3, 2: mean 2, SD 1.
    assert!((b.bias - 2.0).abs() < 1e-12);
    assert!((b.sd - 1.0).abs() < 1e-12);
    assert!((b.loa_hi - (2.0 + 1.959_964)).abs() < 1e-5);
    assert!(b.bias_ci.contains(2.0) && !b.bias_ci.contains(0.0));
}

#[test]
fn mismatched_or_degenerate_pairs_are_refused() {
    use metrale_error::ErrorCode;
    assert_eq!(
        deming(&[1.0, 2.0, 3.0], &[1.0, 2.0, 4.0, 5.0], 1.0)
            .unwrap_err()
            .code(),
        ErrorCode::StatsInvalid
    );
    assert_eq!(
        deming(&[1.0, 2.0, 3.0], &[5.0, 5.0, 5.0], 1.0)
            .unwrap_err()
            .code(),
        ErrorCode::StatsInvalid
    );
}
