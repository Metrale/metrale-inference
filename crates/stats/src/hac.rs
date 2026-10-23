// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Autocorrelation-consistent variance of a mean. Power samples inside a window are
//! serially correlated (a 100 ms counter read at 40 Hz repeats; thermals drift), so the i.i.d.
//! s²/n understates the variance of the mean by up to (1+φ)/(1−φ) for AR(1) φ. The estimator of
//! record, [`long_run_var_of_mean`], is Andrews & Monahan's (1992) AR(1)-prewhitened
//! Newey-West: prewhiten with φ̂, take the Bartlett-kernel long-run variance of the residuals at
//! Andrews' (1991) AR(1) plug-in bandwidth, and recolour by 1/(1−φ̂)². The effective sample size
//! is s² over that variance. [`newey_west_var_of_mean`] is the plain Bartlett estimator at a
//! stated lag, and [`newey_west_lag`] the textbook fixed rule 4·(n/100)^(2/9).
//!
//! Owner: metrale-stats.
//! Invariants: φ̂ is clamped to [−0.97, 0.97] so the recolouring is finite (declared).

use metrale_error::{ErrorCode, error};

use crate::Result;
use crate::interval::Summary;
use crate::num::{check, f, floor_usize, mean, var};

/// 2026-10-22: The prewhitening coefficient's clamp.
pub const MAX_PHI: f64 = 0.97;

/// 2026-10-22: Lag-1 sample autocorrelation; 0 for fewer than 3 values or zero variance.
#[must_use]
pub fn ar1(xs: &[f64]) -> f64 {
    if xs.len() < 3 {
        return 0.0;
    }
    let m = mean(xs);
    let den: f64 = xs.iter().map(|x| (x - m) * (x - m)).sum();
    if den == 0.0 {
        return 0.0;
    }
    let num: f64 = xs.windows(2).map(|w| (w[0] - m) * (w[1] - m)).sum();
    num / den
}

/// 2026-10-22: Newey & West's (1994) fixed rule: ⌊4·(n/100)^(2/9)⌋, at least 1.
#[must_use]
pub fn newey_west_lag(n: usize) -> usize {
    floor_usize(4.0 * libm::pow(f(n) / 100.0, 2.0 / 9.0)).max(1)
}

/// 2026-10-22: Andrews' (1991) Bartlett bandwidth for an AR(1) with coefficient `rho`:
/// 1.1447·(α·n)^(1/3), α = 4ρ² / ((1−ρ)²(1+ρ)²); at least 1, at most n − 1.
#[must_use]
pub fn andrews_lag(rho: f64, n: usize) -> usize {
    let r = rho.clamp(-MAX_PHI, MAX_PHI);
    let a = 4.0 * r * r / ((1.0 - r) * (1.0 - r) * (1.0 + r) * (1.0 + r));
    floor_usize(1.1447 * libm::cbrt(a * f(n)))
        .max(1)
        .min(n.saturating_sub(1).max(1))
}

/// 2026-10-22: The Bartlett-kernel long-run variance of `xs` divided by n: the HAC variance of
/// the mean at `lag`.
pub fn newey_west_var_of_mean(xs: &[f64], lag: usize) -> Result<f64> {
    check("newey_west_var_of_mean", xs, 2)?;
    let n = xs.len();
    let m = mean(xs);
    let gamma = |j: usize| -> f64 {
        xs[j..]
            .iter()
            .zip(xs)
            .map(|(a, b)| (a - m) * (b - m))
            .sum::<f64>()
            / f(n)
    };
    let mut lrv = gamma(0);
    for j in 1..=lag.min(n - 1) {
        lrv += 2.0 * (1.0 - f(j) / f(lag + 1)) * gamma(j);
    }
    Ok(lrv.max(0.0) / f(n))
}

/// 2026-10-22: The HAC variance of the mean of record (AR(1)-prewhitened Newey-West, Andrews
/// bandwidth); see the module docs. Needs 8 values.
pub fn long_run_var_of_mean(xs: &[f64]) -> Result<f64> {
    check("long_run_var_of_mean", xs, 8)?;
    let n = xs.len();
    let m = mean(xs);
    let phi = ar1(xs).clamp(-MAX_PHI, MAX_PHI);
    let e: alloc::vec::Vec<f64> = xs
        .windows(2)
        .map(|w| (w[1] - m) - phi * (w[0] - m))
        .collect();
    let lag = andrews_lag(ar1(&e), e.len());
    let lrv_e = newey_west_var_of_mean(&e, lag)? * f(e.len());
    let lrv = lrv_e / ((1.0 - phi) * (1.0 - phi));
    if !lrv.is_finite() {
        return Err(error!(
            ErrorCode::StatsInvalid,
            "long_run_var_of_mean", "the long-run variance is not finite"
        ));
    }
    Ok(lrv / f(n))
}

/// 2026-10-22: Effective sample size: s² / HAC variance of the mean, clamped to [1, n].
pub fn ess(xs: &[f64]) -> Result<f64> {
    let v = long_run_var_of_mean(xs)?;
    let s2 = var(xs);
    if v == 0.0 {
        return Ok(f(xs.len()));
    }
    Ok((s2 / v).clamp(1.0, f(xs.len())))
}

/// 2026-10-22: The mean with its HAC standard error and df = ESS − 1 (at least 1).
pub fn mean_summary(xs: &[f64]) -> Result<Summary> {
    let v = long_run_var_of_mean(xs)?;
    let e = ess(xs)?;
    Ok(Summary {
        mean: mean(xs),
        se: libm::sqrt(v),
        df: (e - 1.0).max(1.0),
    })
}

#[cfg(test)]
#[path = "hac_tests.rs"]
mod tests;
