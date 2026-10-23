// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The interim headroom tests, moved here; the interim formulas kept as reference
//! implementations and compared bit for bit on random data (the headroom golden reports depend on
//! it); and a golden pin of exact output bits so a drift in either is caught.

use alloc::vec::Vec;

use super::*;

// 2026-10-22: The interim `hardware/stats.rs` bodies, verbatim apart from lint-required
// conversions, as the reference the delegating functions must equal bit for bit.
mod interim {
    use alloc::vec::Vec;

    pub fn below(state: &mut u64, n: usize) -> usize {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let wide = u128::from(z) * (n as u128);
        usize::try_from(wide >> 64).unwrap_or(0)
    }

    pub fn median(xs: &[f64]) -> Option<f64> {
        if xs.is_empty() {
            return None;
        }
        let mut v = xs.to_vec();
        v.sort_by(f64::total_cmp);
        let m = v.len() / 2;
        Some(if v.len() % 2 == 1 {
            v[m]
        } else {
            (v[m - 1] + v[m]) / 2.0
        })
    }

    pub fn block_len(n: usize) -> usize {
        let l = libm::ceil(libm::cbrt(n as f64));
        let mut b = 1;
        while (b as f64) < l {
            b += 1;
        }
        b.min(n.max(1))
    }

    pub fn block_resample(segments: &[usize], block: usize, state: &mut u64) -> Vec<usize> {
        let mut out = Vec::new();
        let mut base = 0;
        for &len in segments {
            let b = block.clamp(1, len.max(1));
            let starts = len + 1 - b;
            let mut filled = 0;
            while filled < len {
                let s = below(state, starts);
                for j in 0..b.min(len - filled) {
                    out.push(base + s + j);
                }
                filled += b.min(len - filled);
            }
            base += len;
        }
        out
    }

    pub fn percentile_ci(reps: &[f64], level: f64) -> Option<(f64, f64)> {
        if reps.is_empty() {
            return None;
        }
        let mut v = reps.to_vec();
        v.sort_by(f64::total_cmp);
        let q = |p: f64| {
            let x = p * (v.len() - 1) as f64;
            let i = libm::floor(x);
            let f = x - i;
            let i = (0..v.len()).take_while(|&k| (k as f64) < i).count();
            let j = (i + 1).min(v.len() - 1);
            v[i] + f * (v[j] - v[i])
        };
        let a = (1.0 - level) / 2.0;
        Some((q(a), q(1.0 - a)))
    }

    pub fn hac_variance_of_mean(xs: &[f64], lag: usize) -> Option<f64> {
        let n = xs.len();
        if n < 2 {
            return None;
        }
        let nf = n as f64;
        let mean = xs.iter().sum::<f64>() / nf;
        let gamma = |k: usize| {
            xs[k..]
                .iter()
                .zip(xs)
                .map(|(a, b)| (a - mean) * (b - mean))
                .sum::<f64>()
                / nf
        };
        let mut s = gamma(0);
        for k in 1..=lag.min(n - 1) {
            s += 2.0 * (1.0 - k as f64 / (lag as f64 + 1.0)) * gamma(k);
        }
        Some(s.max(0.0) / nf)
    }
}

#[test]
fn every_function_equals_the_interim_formula_bit_for_bit() {
    let mut data = SplitMix64::new(0xC0FFEE);
    for trial in 0..300u64 {
        let n = 1 + data.below(400);
        let xs: Vec<f64> = (0..n).map(|_| 40.0 + 7.0 * data.normal()).collect();
        assert_eq!(
            median(&xs).map(f64::to_bits),
            interim::median(&xs).map(f64::to_bits)
        );
        for lag in [0usize, 1, 4, 17] {
            assert_eq!(
                hac_variance_of_mean(&xs, lag).map(f64::to_bits),
                interim::hac_variance_of_mean(&xs, lag).map(f64::to_bits)
            );
        }
        for level in [0.5, 0.9, 0.95, 0.99] {
            let ours = percentile_ci(&xs, level).map(|c| (c.lo.to_bits(), c.hi.to_bits()));
            let theirs = interim::percentile_ci(&xs, level).map(|c| (c.0.to_bits(), c.1.to_bits()));
            assert_eq!(ours, theirs);
        }
        assert_eq!(block_len(n), interim::block_len(n));
        let segs = [n, 1 + (n % 7), n / 3];
        let block = 1 + data.below(12);
        let (mut a, mut b) = (SplitMix64::new(trial), trial);
        for _ in 0..3 {
            assert_eq!(
                block_resample(&segs, block, &mut a),
                interim::block_resample(&segs, block, &mut b)
            );
        }
    }
}

#[test]
fn below_matches_the_interim_formula_for_large_n() {
    for &n in &[1usize, 2, 7, 1 << 20, usize::MAX / 3, usize::MAX] {
        let (mut a, mut b) = (SplitMix64::new(42), 42u64);
        for _ in 0..64 {
            assert_eq!(a.below(n), interim::below(&mut b, n));
        }
    }
}

#[test]
fn golden_bits_are_pinned() {
    let mut rng = SplitMix64::new(0x6865_6164_726f_6f6d);
    let idx = block_resample(&[5, 3], 2, &mut rng);
    assert_eq!(idx, GOLDEN_IDX);
    let reps: Vec<f64> = (0..37)
        .map(|i| libm::sin(f64::from(i)) * 3.0 + 10.0)
        .collect();
    let c = percentile_ci(&reps, 0.95).unwrap();
    assert_eq!((c.lo.to_bits(), c.hi.to_bits()), GOLDEN_CI);
    assert_eq!(
        hac_variance_of_mean(&reps, 3).unwrap().to_bits(),
        GOLDEN_HAC
    );
}

const GOLDEN_IDX: [usize; 8] = [3, 4, 0, 1, 2, 5, 6, 6];
const GOLDEN_CI: (u64, u64) = (4_619_592_312_791_204_213, 4_623_493_676_074_310_899);
const GOLDEN_HAC: u64 = 4_592_793_532_852_533_300;

#[test]
fn the_median_and_the_percentile_interval_on_hand_values() {
    assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
    assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), Some(2.5));
    assert_eq!(median(&[]), None);
    // 2026-10-22: 0..=100: the 2.5% and 97.5% points are 2.5 and 97.5.
    let v: Vec<f64> = (0..=100).map(f64::from).collect();
    let c = percentile_ci(&v, 0.95).unwrap();
    assert!(
        (c.lo - 2.5).abs() < 1e-9 && (c.hi - 97.5).abs() < 1e-9,
        "{c:?}"
    );
    assert!(c.overlaps(&Interval {
        lo: 97.0,
        hi: 200.0
    }));
    assert!(!c.overlaps(&Interval {
        lo: 98.0,
        hi: 200.0
    }));
}

#[test]
fn blocks_stay_inside_their_segment_and_rebuild_its_length() {
    let mut rng = SplitMix64::new(7);
    for _ in 0..200 {
        let idx = block_resample(&[5, 3], 2, &mut rng);
        assert_eq!(idx.len(), 8);
        assert!(idx[..5].iter().all(|&i| i < 5), "{idx:?}");
        assert!(idx[5..].iter().all(|&i| (5..8).contains(&i)), "{idx:?}");
    }
    assert_eq!(block_len(245), 7);
    assert_eq!(block_len(1), 1);
}

#[test]
fn blocks_carry_autocorrelation_that_single_draws_lose() {
    // 2026-10-22: A series in runs of 16: the block bootstrap's spread of the mean is wider than
    // the i.i.d. bootstrap's (block 1), which treats correlated steps as independent.
    let x: Vec<f64> = (0..256)
        .map(|i| if (i / 16) % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let spread = |block: usize| {
        let mut rng = SplitMix64::new(1);
        let reps: Vec<f64> = (0..2000)
            .map(|_| {
                let idx = block_resample(&[256], block, &mut rng);
                idx.iter().map(|&i| x[i]).sum::<f64>() / 256.0
            })
            .collect();
        let c = percentile_ci(&reps, 0.95).unwrap();
        c.hi - c.lo
    };
    let (iid, blocked) = (spread(1), spread(16));
    assert!(blocked > 1.5 * iid, "block {blocked} vs iid {iid}");
}

#[test]
fn hac_variance_grows_with_positive_autocorrelation() {
    // 2026-10-22: Alternating and blocky series with the same values: lag-1 covariance negative
    // for one, positive for the other, and the HAC variance orders them.
    let alt: Vec<f64> = (0..64)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let blocky: Vec<f64> = (0..64)
        .map(|i| if (i / 8) % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let iid = 1.0 / 64.0;
    let a = hac_variance_of_mean(&alt, 4).unwrap();
    let b = hac_variance_of_mean(&blocky, 4).unwrap();
    assert!(a < iid && b > iid, "{a} {b} {iid}");
    // 2026-10-22: Lag 0 is the i.i.d. variance of the mean (population variance / n).
    assert!((hac_variance_of_mean(&blocky, 0).unwrap() - iid).abs() < 1e-12);
}
