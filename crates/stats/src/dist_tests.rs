// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Quantiles against published tables; CDF/quantile round trips; a wrong df moves
//! the quantile (the t is not silently the normal).

use super::*;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn quantiles_match_the_tables() {
    assert!(close(norm_ppf(0.975), 1.959_964, 1e-6));
    assert!(close(norm_ppf(0.5), 0.0, 1e-12));
    assert!(close(norm_ppf(0.001), -3.090_232, 1e-6));
    assert!(close(t_ppf(0.975, 2.0), 4.302_653, 1e-6));
    assert!(close(t_ppf(0.975, 10.0), 2.228_139, 1e-6));
    assert!(close(t_ppf(0.95, 1.0), 6.313_752, 1e-6));
    assert!(close(t_ppf(0.025, 5.0), -2.570_582, 1e-6));
    assert!(close(t_ppf(0.975, f64::INFINITY), 1.959_964, 1e-6));
}

#[test]
fn the_t_is_not_the_normal_at_small_df() {
    assert!(t_ppf(0.975, 3.0) > 3.0);
    assert!(t_cdf(2.0, 3.0) < norm_cdf(2.0));
    // 2026-10-22: Non-integer df interpolate between the neighbours.
    let (a, b, c) = (t_ppf(0.975, 4.0), t_ppf(0.975, 4.5), t_ppf(0.975, 5.0));
    assert!(a > b && b > c);
}

#[test]
fn cdf_and_quantile_round_trip() {
    for &p in &[1e-6, 0.01, 0.2, 0.5, 0.8, 0.99, 1.0 - 1e-6] {
        assert!(close(norm_cdf(norm_ppf(p)), p, 1e-12), "{p}");
        for &df in &[1.0, 2.5, 7.0, 40.0] {
            assert!(close(t_cdf(t_ppf(p, df), df), p, 1e-10), "{p} {df}");
        }
    }
}

#[test]
fn out_of_range_probabilities_are_not_numbers() {
    assert!(norm_ppf(1.5).is_nan());
    assert!(t_ppf(-0.1, 3.0).is_nan());
    assert!(t_cdf(1.0, 0.0).is_nan());
    assert_eq!(norm_ppf(0.0), f64::NEG_INFINITY);
}
