// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Resample indices keep the structure they promise; the automatic block grows with
//! the autocorrelation; BCa reaches nominal coverage on an autocorrelated series where an
//! i.i.d. (block length 1) bootstrap does not.

use alloc::vec::Vec;

use super::*;
use crate::sim::{BAND, ar1_series, coverage};

#[test]
fn indices_are_in_range_and_blocks_are_contiguous() {
    let mut r = SplitMix64::new(3);
    let idx = moving_block_indices(100, 7, &mut r);
    assert_eq!(idx.len(), 100);
    assert!(idx.iter().all(|&i| i < 100));
    assert!(idx[..7].windows(2).all(|w| w[1] == w[0] + 1));
    let st = stationary_indices(50, 5.0, &mut r);
    assert_eq!(st.len(), 50);
    let runs = st.windows(2).filter(|w| w[1] != (w[0] + 1) % 50).count();
    assert!(
        runs > 2 && runs < 25,
        "{runs} block starts for mean length 5"
    );
}

#[test]
fn the_block_rule_tracks_dependence() {
    assert_eq!(block_length(1000), 10);
    let mut r = SplitMix64::new(4);
    let ar = ar1_series(1000, 0.8, 0.0, &mut r);
    let iid = ar1_series(1000, 0.0, 0.0, &mut r);
    assert!(politis_white(&ar) > 3.0 * politis_white(&iid));
}

#[test]
fn bca_reduces_to_the_percentile_interval_without_bias_or_skew() {
    let boot: Vec<f64> = (0..1001).map(|i| f(i) / 1000.0).collect();
    let jack = [0.4, 0.5, 0.6];
    let bca = bca_interval(0.5, &boot, &jack, 0.9).unwrap();
    let pct = percentile_interval(&boot, 0.9).unwrap();
    assert!((bca.lo - pct.lo).abs() < 2e-3 && (bca.hi - pct.hi).abs() < 2e-3);
    // 2026-10-22: A biased estimate (every replicate above it) shifts the interval down.
    let shifted = bca_interval(0.2, &boot, &jack, 0.9).unwrap();
    assert!(shifted.hi < pct.hi);
}

fn mean_of(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / f(xs.len())
}

#[test]
fn bca_series_reaches_nominal_coverage_on_ar1() {
    let mut r = SplitMix64::new(31);
    let mut seed = 1000;
    // 2026-10-22: 1000 samples: below that the block bootstrap under-covers (measured 87% at
    // n = 200 and 400 for this series; the HAC-t is the window interval of record there).
    let n = 1000;
    let c = coverage(500, |_| {
        let xs = ar1_series(n, 0.5, 2.0, &mut r);
        seed += 1;
        bca_series(&xs, mean_of, 499, 0.95, seed)
            .unwrap()
            .contains(2.0)
    });
    assert!(c >= BAND.0 && c <= BAND.1, "BCa coverage {c}");
}

/// 2026-10-22: The negative control: single-sample resampling (block length 1) of the same
/// series under-covers.
#[test]
fn an_iid_bootstrap_under_covers_the_same_series() {
    let mut r = SplitMix64::new(31);
    let mut rr = SplitMix64::new(77);
    let c = coverage(500, |_| {
        let xs = ar1_series(1000, 0.5, 2.0, &mut r);
        let boot: Vec<f64> = (0..499)
            .map(|_| {
                let v: Vec<f64> = stationary_indices(1000, 1.0, &mut rr)
                    .iter()
                    .map(|&i| xs[i])
                    .collect();
                mean_of(&v)
            })
            .collect();
        percentile_interval(&boot, 0.95).unwrap().contains(2.0)
    });
    assert!(c < BAND.0, "iid bootstrap coverage {c}");
}
