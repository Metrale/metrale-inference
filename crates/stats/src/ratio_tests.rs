// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The delta method matches a simulated ratio's spread for a precise denominator;
//! `agree` flags a skewed bootstrap interval.

use alloc::vec::Vec;

use super::*;
use crate::interval::CiMethod;
use crate::num::{f, mean, var};
use crate::rng::SplitMix64;

#[test]
fn delta_ratio_matches_simulation() {
    let (mn, sn, md, sd) = (50.0, 2.0, 10.0, 0.3);
    let d = delta_ratio(
        Summary {
            mean: mn,
            se: sn,
            df: f64::INFINITY,
        },
        Summary {
            mean: md,
            se: sd,
            df: f64::INFINITY,
        },
        0.0,
    );
    let mut r = SplitMix64::new(61);
    let draws: Vec<f64> = (0..20000)
        .map(|_| (mn + sn * r.normal()) / (md + sd * r.normal()))
        .collect();
    let sim_sd = libm::sqrt(var(&draws));
    assert!((d.mean - 5.0).abs() < 1e-12);
    assert!((d.se / sim_sd - 1.0).abs() < 0.05, "{} vs {sim_sd}", d.se);
    assert!((mean(&draws) - 5.0).abs() < 0.05);
    let _ = f(1);
}

#[test]
fn delta_product_includes_both_terms_and_covariance() {
    let a = Summary {
        mean: 2.0,
        se: 0.1,
        df: 10.0,
    };
    let b = Summary {
        mean: 3.0,
        se: 0.2,
        df: 10.0,
    };
    let p = delta_product(a, b, 0.0);
    assert!((p.se - libm::sqrt(9.0 * 0.01 + 4.0 * 0.04)).abs() < 1e-12);
    assert!(delta_product(a, b, 0.01).se > p.se);
}

#[test]
fn agree_flags_disagreeing_intervals() {
    let i = |lo, hi| Interval {
        lo,
        hi,
        level: 0.95,
        method: CiMethod::Delta,
    };
    assert!(agree(&i(1.0, 2.0), &i(1.02, 2.03), 0.1));
    assert!(!agree(&i(1.0, 2.0), &i(1.3, 2.9), 0.1));
}
