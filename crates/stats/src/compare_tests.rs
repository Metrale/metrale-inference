// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: TOST refuses equivalence when the true difference exceeds the margin and when the
//! run is too noisy, and grants it for a precise null; superiority needs a clear margin; Holm
//! and BH match hand-computed examples; the MDE matches its formula's table value.

use super::*;

fn s(mean: f64, se: f64, df: f64) -> Summary {
    Summary { mean, se, df }
}

#[test]
fn tost_rejects_a_difference_beyond_the_margin() {
    assert!(!tost(s(0.05, 0.005, 20.0), 0.02, 0.05).equivalent);
    assert!(tost(s(0.0, 0.005, 20.0), 0.02, 0.05).equivalent);
    // 2026-10-22: A noisy run cannot show equivalence even when its point estimate is 0.
    assert!(!tost(s(0.0, 0.05, 20.0), 0.02, 0.05).equivalent);
}

#[test]
fn superiority_needs_the_margin_cleared() {
    assert!(superiority(s(0.10, 0.01, 30.0), 0.02, 0.05).reject);
    assert!(!superiority(s(0.025, 0.01, 30.0), 0.02, 0.05).reject);
    assert!(!superiority(s(-0.1, 0.01, 30.0), 0.0, 0.05).reject);
}

#[test]
fn welch_combines_ses_and_df() {
    let d = welch(s(10.0, 1.0, 9.0), s(8.0, 1.0, 9.0));
    assert!((d.mean - 2.0).abs() < 1e-12);
    assert!((d.se - libm::sqrt(2.0)).abs() < 1e-12);
    assert!((d.df - 18.0).abs() < 1e-9);
}

#[test]
fn holm_and_bh_match_hand_examples() {
    let p = [0.01, 0.04, 0.03, 0.005];
    // 2026-10-22: Holm at 0.05: 0.005 ≤ .0125, 0.01 ≤ .0167, 0.03 > .025 → stop.
    assert_eq!(holm(&p, 0.05), [true, false, false, true]);
    // 2026-10-22: BH at 0.05: thresholds .0125 .025 .0375 .05; largest k with p_(k) ≤: k=4
    // (0.04 ≤ 0.05) → all rejected.
    assert_eq!(benjamini_hochberg(&p, 0.05), [true, true, true, true]);
    assert_eq!(benjamini_hochberg(&[0.2, 0.5], 0.05), [false, false]);
}

#[test]
fn mde_matches_the_table() {
    // 2026-10-22: Normal: (1.959964 + 0.841621)·1 = 2.801585 (80% power, α = 0.05).
    assert!((mde(1.0, f64::INFINITY, 0.05, 0.8) - 2.801_585).abs() < 1e-5);
    assert!(mde(1.0, 5.0, 0.05, 0.8) > 3.3);
}
