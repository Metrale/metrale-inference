// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Dependence-preserving bootstrap. Resampling single samples of an autocorrelated
//! power series destroys the correlation and gives the same too-narrow interval as the i.i.d. SE;
//! resampling blocks keeps it. The moving-block bootstrap (Künsch 1989) draws fixed-length blocks;
//! the stationary bootstrap (Politis & Romano 1994) draws geometric-length circular blocks, whose
//! mean length is chosen by Politis & White's (2004; corrected by Patton, Politis & White 2009)
//! automatic rule. Intervals are BCa (Efron 1987): the percentile interval corrected for median
//! bias (z₀, from the bootstrap distribution) and skew (acceleration a, from a delete-one-block
//! jackknife), second-order accurate where the percentile interval is first-order.
//!
//! Owner: metrale-stats.
//! Invariants: every resample is a function of the stated seed alone.

use alloc::vec::Vec;

use metrale_error::{ErrorCode, error};

use crate::Result;
use crate::dist::{norm_cdf, norm_ppf, t_ppf};
use crate::interval::{CiMethod, Interval};
use crate::num::{check, f, floor_usize, mean, quantile_sorted, sorted};
use crate::rng::SplitMix64;

/// 2026-10-22: The rate rule for a fixed block length: round(n^(1/3)), at least 1 (Hall,
/// Horowitz & Jing 1995: the MSE-optimal block for a variance grows as n^(1/3)).
#[must_use]
pub fn block_length(n: usize) -> usize {
    floor_usize(libm::cbrt(f(n)) + 0.5).max(1)
}

/// 2026-10-22: Indices of one moving-block resample of `0..n` with blocks of `block`.
pub fn moving_block_indices(n: usize, block: usize, r: &mut SplitMix64) -> Vec<usize> {
    let b = block.clamp(1, n.max(1));
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let start = r.below(n - b + 1);
        out.extend((start..start + b).take(n - out.len()));
    }
    out
}

/// 2026-10-22: Indices of one stationary-bootstrap resample of `0..n` (circular, geometric block
/// lengths of mean `mean_block`).
pub fn stationary_indices(n: usize, mean_block: f64, r: &mut SplitMix64) -> Vec<usize> {
    let p = 1.0 / mean_block.max(1.0);
    let mut out = Vec::with_capacity(n);
    let mut i = r.below(n);
    while out.len() < n {
        out.push(i);
        i = if r.next_f64() < p {
            r.below(n)
        } else {
            (i + 1) % n
        };
    }
    out
}

fn acov(xs: &[f64], m: f64, k: usize) -> f64 {
    xs[k..]
        .iter()
        .zip(xs)
        .map(|(a, b)| (a - m) * (b - m))
        .sum::<f64>()
        / f(xs.len())
}

/// 2026-10-22: The spectral quantities of Politis & White: (G, g(0)) from flat-top-weighted
/// autocovariances; `None` for a short or constant series.
fn pw_spectral(xs: &[f64]) -> Option<(f64, f64)> {
    let n = xs.len();
    if n < 8 {
        return None;
    }
    let m = mean(xs);
    let r0 = acov(xs, m, 0);
    if r0 == 0.0 {
        return None;
    }
    let kn = floor_usize(libm::ceil(libm::sqrt(libm::log10(f(n))))).max(5);
    let crit = 2.0 * libm::sqrt(libm::log10(f(n)) / f(n));
    let max_lag = (n / 2).max(1);
    let rho = |k: usize| acov(xs, m, k) / r0;
    let mut mhat = max_lag;
    for cand in 1..max_lag {
        if (1..=kn).all(|k| cand + k >= n || rho(cand + k).abs() < crit) {
            mhat = cand;
            break;
        }
    }
    let big_m = (2 * mhat).min(max_lag).max(1);
    let lambda = |t: f64| {
        let t = t.abs();
        if t <= 0.5 {
            1.0
        } else if t <= 1.0 {
            2.0 * (1.0 - t)
        } else {
            0.0
        }
    };
    let (mut g, mut g0) = (0.0, r0);
    for k in 1..=big_m.min(n - 1) {
        let w = lambda(f(k) / f(big_m));
        let rk = acov(xs, m, k);
        g += 2.0 * w * f(k) * rk;
        g0 += 2.0 * w * rk;
    }
    Some((g, g0))
}

fn max_block(n: usize) -> f64 {
    (3.0 * libm::sqrt(f(n))).min(f(n) / 3.0).max(1.0)
}

/// 2026-10-22: Politis & White's automatic mean block length for the stationary bootstrap of
/// the mean (MSE-optimal for the variance), clamped to [1, min(3√n, n/3)].
#[must_use]
pub fn politis_white(xs: &[f64]) -> f64 {
    let Some((g, g0)) = pw_spectral(xs) else {
        return 1.0;
    };
    let d = 2.0 * g0 * g0;
    if d == 0.0 || g == 0.0 {
        return 1.0;
    }
    let b = libm::cbrt(2.0 * g * g / d) * libm::cbrt(f(xs.len()));
    b.clamp(1.0, max_block(xs.len()))
}

/// 2026-10-22: The percentile interval of `boot`.
pub fn percentile_interval(boot: &[f64], level: f64) -> Result<Interval> {
    check("percentile_interval", boot, 20)?;
    let s = sorted(boot);
    let a = (1.0 - level) / 2.0;
    Ok(Interval {
        lo: quantile_sorted(&s, a),
        hi: quantile_sorted(&s, 1.0 - a),
        level,
        method: CiMethod::PercentileBlockBootstrap,
    })
}

/// 2026-10-22: The BCa interval from the estimate `theta_hat`, its bootstrap replicates and its
/// jackknife replicates (any deletion scheme; the acceleration uses their skew).
pub fn bca_interval(
    theta_hat: f64,
    boot: &[f64],
    jackknife: &[f64],
    level: f64,
) -> Result<Interval> {
    bca_expanded(theta_hat, boot, jackknife, level, f64::INFINITY)
}

/// 2026-10-22: [`bca_interval`] with Hesterberg's (2015) small-sample expansion: each tail
/// probability α becomes Φ(√(m/(m−1))·t_{α, m−1}) for `m` independent units (for a block
/// bootstrap, the number of blocks n/b), which corrects the percentile family's narrowness at
/// small m. `m = ∞` is plain BCa.
pub fn bca_expanded(
    theta_hat: f64,
    boot: &[f64],
    jackknife: &[f64],
    level: f64,
    m: f64,
) -> Result<Interval> {
    check("bca_interval", boot, 20)?;
    check("bca_interval jackknife", jackknife, 2)?;
    if !(level > 0.0 && level < 1.0 && theta_hat.is_finite()) {
        return Err(error!(
            ErrorCode::StatsInvalid,
            "bca_interval", "level or estimate invalid"
        ));
    }
    let b = f(boot.len());
    let below = boot.iter().filter(|&&v| v < theta_hat).count();
    let tied = boot.iter().filter(|&&v| v == theta_hat).count();
    let p0 = ((f(below) + 0.5 * f(tied)) / b).clamp(0.5 / b, 1.0 - 0.5 / b);
    let z0 = norm_ppf(p0);
    let jm = mean(jackknife);
    let s2: f64 = jackknife.iter().map(|t| (jm - t) * (jm - t)).sum();
    let s3: f64 = jackknife
        .iter()
        .map(|t| (jm - t) * (jm - t) * (jm - t))
        .sum();
    let acc = if s2 > 0.0 {
        s3 / (6.0 * libm::pow(s2, 1.5))
    } else {
        0.0
    };
    let s = sorted(boot);
    let adj = |alpha: f64| {
        let z = if m.is_finite() && m > 2.0 {
            libm::sqrt(m / (m - 1.0)) * t_ppf(alpha, m - 1.0)
        } else {
            norm_ppf(alpha)
        };
        let q = norm_cdf(z0 + (z0 + z) / (1.0 - acc * (z0 + z)));
        quantile_sorted(&s, q)
    };
    let a = (1.0 - level) / 2.0;
    let (lo, hi) = (adj(a), adj(1.0 - a));
    Ok(Interval {
        lo: lo.min(hi),
        hi: hi.max(lo),
        level,
        method: CiMethod::BcaBlockBootstrap,
    })
}

/// 2026-10-22: BCa interval of `stat` over the autocorrelated series `xs`: `reps` stationary
/// bootstrap resamples at the Politis-White mean block, acceleration from a delete-one-block
/// jackknife over non-overlapping blocks of that length, expanded for the n/b blocks
/// ([`bca_expanded`]): without the expansion a 95% interval covered 86% at n = 400, AR(1) 0.5
/// (the bootstrap SE of a few dozen blocks is itself noisy, as a t with few df).
pub fn bca_series<F: Fn(&[f64]) -> f64>(
    xs: &[f64],
    stat: F,
    reps: usize,
    level: f64,
    seed: u64,
) -> Result<Interval> {
    check("bca_series", xs, 8)?;
    let n = xs.len();
    let mean_block = politis_white(xs);
    let mut r = SplitMix64::new(seed);
    let mut buf = Vec::with_capacity(n);
    let boot: Vec<f64> = (0..reps)
        .map(|_| {
            buf.clear();
            buf.extend(
                stationary_indices(n, mean_block, &mut r)
                    .into_iter()
                    .map(|i| xs[i]),
            );
            stat(&buf)
        })
        .collect();
    let jb = floor_usize(mean_block + 0.5).clamp(1, n / 2);
    let blocks = n / jb;
    let jack: Vec<f64> = (0..blocks)
        .map(|k| {
            buf.clear();
            buf.extend_from_slice(&xs[..k * jb]);
            buf.extend_from_slice(&xs[(k + 1) * jb..]);
            stat(&buf)
        })
        .collect();
    bca_expanded(stat(xs), &boot, &jack, level, f(n) / mean_block)
}

#[cfg(test)]
#[path = "bootstrap_tests.rs"]
mod tests;
