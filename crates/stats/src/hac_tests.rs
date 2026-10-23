// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: HAC widens the interval on autocorrelated data by about √((1+φ)/(1−φ)), leaves it
//! alone on i.i.d. data, and reaches nominal coverage where the i.i.d. SE does not (the negative
//! control proving the coverage test has teeth).

use super::*;
use crate::interval::CiMethod;
use crate::rng::SplitMix64;
use crate::sim::{BAND, ar1_series, coverage};

#[test]
fn autocorrelation_widens_the_se_and_shrinks_the_ess() {
    let mut r = SplitMix64::new(11);
    let xs = ar1_series(4000, 0.7, 5.0, &mut r);
    let iid = libm::sqrt(var(&xs) / 4000.0);
    let hac = mean_summary(&xs).unwrap().se;
    let want = libm::sqrt(1.7 / 0.3);
    assert!(
        (hac / iid / want - 1.0).abs() < 0.15,
        "ratio {} vs {want}",
        hac / iid
    );
    assert!(ess(&xs).unwrap() < 4000.0 / 4.0);
    assert!((ar1(&xs) - 0.7).abs() < 0.05);
}

#[test]
fn iid_data_keeps_its_sample_size() {
    let mut r = SplitMix64::new(12);
    let xs = ar1_series(4000, 0.0, 0.0, &mut r);
    let e = ess(&xs).unwrap();
    assert!(e > 3200.0, "ess {e}");
}

#[test]
fn lag_rules_match_hand_values() {
    assert_eq!(newey_west_lag(100), 4);
    assert_eq!(newey_west_lag(1), 1);
    assert_eq!(andrews_lag(0.0, 1000), 1);
    // 2026-10-22: ρ = 0.6, n = 200: α = 3.5156, 1.1447·703.1^(1/3) = 10.2.
    assert_eq!(andrews_lag(0.6, 200), 10);
}

fn hac_covers(seed: u64) -> f64 {
    let mut r = SplitMix64::new(seed);
    coverage(1000, |_| {
        let xs = ar1_series(200, 0.6, 3.0, &mut r);
        mean_summary(&xs)
            .unwrap()
            .interval(0.95, CiMethod::HacT)
            .contains(3.0)
    })
}

#[test]
fn hac_mean_interval_reaches_nominal_coverage_on_ar1() {
    let c = hac_covers(21);
    assert!(c >= BAND.0 && c <= BAND.1, "HAC coverage {c}");
}

/// 2026-10-22: The negative control: the same simulation with the i.i.d. SE under-covers, so the
/// band above is a real test of the HAC estimator, not of the data.
#[test]
fn the_iid_se_under_covers_the_same_series() {
    let mut r = SplitMix64::new(21);
    let c = coverage(1000, |_| {
        let xs = ar1_series(200, 0.6, 3.0, &mut r);
        let s = Summary {
            mean: mean(&xs),
            se: libm::sqrt(var(&xs) / 200.0),
            df: 199.0,
        };
        s.interval(0.95, CiMethod::T).contains(3.0)
    });
    assert!(c < BAND.0, "iid coverage {c} should be far below nominal");
}

#[test]
fn too_few_values_are_refused() {
    use metrale_error::ErrorCode;
    assert_eq!(
        long_run_var_of_mean(&[1.0; 5]).unwrap_err().code(),
        ErrorCode::StatsInsufficient
    );
}
