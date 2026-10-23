// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Change points, so warm-up is trimmed by rule rather than by eye. PELT (Killick,
//! Fearnhead & Eckley 2012) finds the optimal segmentation of a mean-shift model exactly in
//! linear expected time; the Gaussian cost is normalized by σ̂², estimated robustly from the MAD
//! of first differences (insensitive to the shifts themselves), and the default penalty is the
//! Schwarz/BIC 2·ln n per change (one location and one mean). A two-sided tabular CUSUM (Page
//! 1954) is the online detector. The steady-state rule: a load window starts at the last change
//! point PELT finds in the first half of any of its series (power, SM clock), else at 0.
//!
//! Owner: metrale-stats.
//! Invariants: segments are at least [`MIN_SEGMENT`] samples.

use alloc::vec;
use alloc::vec::Vec;

use crate::num::f;
use crate::robust::{MAD_TO_SIGMA, mad, median};

/// 2026-10-22: The shortest segment PELT may return.
pub const MIN_SEGMENT: usize = 3;

/// 2026-10-22: The BIC penalty per change for a σ²-normalized Gaussian mean cost.
#[must_use]
pub fn bic_penalty(n: usize) -> f64 {
    2.0 * libm::log(f(n.max(2)))
}

fn sigma2(xs: &[f64]) -> f64 {
    let d: Vec<f64> = xs.windows(2).map(|w| w[1] - w[0]).collect();
    let s = mad(&d).map_or(0.0, |m| MAD_TO_SIGMA * m / core::f64::consts::SQRT_2);
    if s > 0.0 {
        return s * s;
    }
    let m = xs.iter().sum::<f64>() / f(xs.len());
    (xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / f(xs.len())).max(f64::MIN_POSITIVE)
}

/// 2026-10-22: Start indices of the segments after the first (empty: no change), by PELT with
/// `penalty` per change.
#[must_use]
pub fn pelt_mean(xs: &[f64], penalty: f64) -> Vec<usize> {
    let n = xs.len();
    if n < 2 * MIN_SEGMENT || xs.iter().any(|x| !x.is_finite()) {
        return Vec::new();
    }
    let s2 = sigma2(xs);
    let mut c1 = vec![0.0; n + 1];
    let mut c2 = vec![0.0; n + 1];
    for i in 0..n {
        c1[i + 1] = c1[i] + xs[i];
        c2[i + 1] = c2[i] + xs[i] * xs[i];
    }
    let cost = |a: usize, b: usize| {
        let (s, q, m) = (c1[b] - c1[a], c2[b] - c2[a], f(b - a));
        (q - s * s / m).max(0.0) / s2
    };
    let mut best = vec![f64::INFINITY; n + 1];
    let mut prev = vec![0usize; n + 1];
    best[0] = -penalty;
    let mut cands: Vec<usize> = vec![0];
    for t in MIN_SEGMENT..=n {
        let mut bt = f64::INFINITY;
        let mut arg = 0;
        for &c in cands.iter().filter(|&&c| t - c >= MIN_SEGMENT) {
            let v = best[c] + cost(c, t) + penalty;
            if v < bt {
                bt = v;
                arg = c;
            }
        }
        best[t] = bt;
        prev[t] = arg;
        cands.retain(|&c| t - c < MIN_SEGMENT || best[c] + cost(c, t) <= bt);
        if t + MIN_SEGMENT <= n {
            cands.push(t + 1 - MIN_SEGMENT);
            cands.sort_unstable();
            cands.dedup();
        }
    }
    let mut out = Vec::new();
    let mut t = n;
    while t > 0 {
        let p = prev[t];
        if p > 0 {
            out.push(p);
        }
        t = p;
    }
    out.reverse();
    out
}

/// 2026-10-22: The first index at which a two-sided CUSUM of the standardized series (centre
/// the median, scale σ̂ from first differences) exceeds `h` with slack `k`.
#[must_use]
pub fn cusum(xs: &[f64], k: f64, h: f64) -> Option<usize> {
    let centre = median(xs).ok()?;
    let s = libm::sqrt(sigma2(xs));
    let (mut hi, mut lo) = (0.0f64, 0.0f64);
    for (i, x) in xs.iter().enumerate() {
        let z = (x - centre) / s;
        hi = (hi + z - k).max(0.0);
        lo = (lo - z - k).max(0.0);
        if hi > h || lo > h {
            return Some(i);
        }
    }
    None
}

/// 2026-10-22: The steady-state start of a window: the largest change point that BIC-penalized
/// PELT finds in the first half of any of `series`, else 0.
#[must_use]
pub fn steady_start(series: &[&[f64]]) -> usize {
    series
        .iter()
        .flat_map(|s| {
            let half = s.len() / 2;
            pelt_mean(s, bic_penalty(s.len()))
                .into_iter()
                .filter(move |&c| c <= half)
        })
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
#[path = "changepoint_tests.rs"]
mod tests;
