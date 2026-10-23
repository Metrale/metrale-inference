// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The statistics headroom needs, and only those: the moving-block bootstrap over
//! autocorrelated per-step series (Künsch 1989), percentile intervals, the median, and the
//! Newey-West (HAC) variance of a mean for the delta-method cross-check.
//!
//! **Interim.** The kernel-energy work (branch `kernel-energy`, crate `metrale-stats`) builds the
//! shared pure stats module (BCa and block bootstrap, HAC ESS, mixed effects, TOST, Holm/BH),
//! still a placeholder on 2026-10-22; see docs/kernel-perf/HEADROOM.md. These functions
//! carry the names that module is to export (`median`, `block_len`, `block_resample`,
//! `percentile_ci`, `hac_variance_of_mean`) so the merge is a change of `use` path: when it
//! lands, this file is deleted and `headroom_ci.rs` imports the shared module (one source of
//! truth); its tests move with it.
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - Pure and deterministic: the resampling stream is SplitMix64 from a stated seed, so the same
//!   series and seed give the same interval bytes (the reports are golden-checked).
//! - A block never crosses a segment boundary (one capture's steps are one segment), so pooling
//!   captures never invents a continuity between them.

use crate::prelude::*;

/// 2026-10-22: A two-sided interval.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct Interval {
    /// 2026-10-22: Lower bound.
    pub lo: f64,
    /// 2026-10-22: Upper bound.
    pub hi: f64,
}

impl Interval {
    /// 2026-10-22: The two intervals share a point.
    pub fn overlaps(&self, o: &Interval) -> bool {
        self.lo <= o.hi && o.lo <= self.hi
    }
}

/// 2026-10-22: SplitMix64 (Steele, Lea, Flood 2014).
#[derive(Debug, Clone)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    /// 2026-10-22: A stream from `seed`.
    pub fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    /// 2026-10-22: The next 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// 2026-10-22: Uniform in `0..n` (`n > 0`), by Lemire's multiply-shift.
    pub fn below(&mut self, n: usize) -> usize {
        let wide = u128::from(self.next_u64()) * (n as u128);
        usize::try_from(wide >> 64).unwrap_or(0)
    }
}

/// 2026-10-22: The median of `xs` (`None` when empty).
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

/// 2026-10-22: The block length for `n` autocorrelated observations: `ceil(n^(1/3))`, the
/// MSE-optimal order for the moving-block bootstrap of a mean (Hall, Horowitz, Jing 1995).
pub fn block_len(n: usize) -> usize {
    let l = libm::ceil(libm::cbrt(n as f64));
    let mut b = 1;
    while (b as f64) < l {
        b += 1;
    }
    b.min(n.max(1))
}

/// 2026-10-22: One moving-block resample of the indices of `segments` (consecutive segment
/// lengths): each segment is rebuilt to its own length from blocks of `block` consecutive indices
/// drawn uniformly inside it, truncated at the end.
pub fn block_resample(segments: &[usize], block: usize, rng: &mut SplitMix64) -> Vec<usize> {
    let mut out = Vec::with_capacity(segments.iter().sum());
    let mut base = 0;
    for &len in segments {
        let b = block.clamp(1, len.max(1));
        let starts = len + 1 - b;
        let mut filled = 0;
        while filled < len {
            let s = rng.below(starts);
            for j in 0..b.min(len - filled) {
                out.push(base + s + j);
            }
            filled += b.min(len - filled);
        }
        base += len;
    }
    out
}

/// 2026-10-22: The percentile interval at `level` (e.g. 0.95) of bootstrap replicates; `None`
/// when there are none. Linear interpolation between order statistics.
pub fn percentile_ci(reps: &[f64], level: f64) -> Option<Interval> {
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
    Some(Interval {
        lo: q(a),
        hi: q(1.0 - a),
    })
}

/// 2026-10-22: The Newey-West (Bartlett kernel) variance of the mean of `xs` with `lag`
/// autocovariances; `None` for fewer than two observations.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_median_and_the_percentile_interval_on_hand_values() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), Some(2.5));
        assert_eq!(median(&[]), None);
        // 0..=100: the 2.5% and 97.5% points are 2.5 and 97.5.
        let v: Vec<f64> = (0..=100).map(f64::from).collect();
        let c = percentile_ci(&v, 0.95).unwrap();
        assert!(
            (c.lo - 2.5).abs() < 1e-9 && (c.hi - 97.5).abs() < 1e-9,
            "{c:?}"
        );
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
        // A series in runs of 16: the block bootstrap's spread of the mean is wider than the
        // i.i.d. bootstrap's (block 1), which treats correlated steps as independent.
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
        // An alternating series and a blocky series with the same values: lag-1 covariance is
        // negative for one and positive for the other, and the HAC variance orders them.
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
        // Lag 0 is the i.i.d. variance of the mean (population variance / n).
        assert!((hac_variance_of_mean(&blocky, 0).unwrap() - iid).abs() < 1e-12);
    }
}
