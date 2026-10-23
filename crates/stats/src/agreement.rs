// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Agreement between two instruments that both measure with error (the NVML counter
//! and sensor disagree by up to 31% on GB10). Ordinary least squares of one on the other is
//! attenuated toward zero by the regressor's own error; Deming regression (errors in both
//! variables, error-variance ratio λ; λ = 1 is orthogonal regression) is not. Its slope and
//! intercept intervals are jackknife (Linnet 1990). Bland & Altman's (1986) limits of agreement
//! state how far one instrument's reading can sit from the other's: bias ± z·SD of the
//! differences.
//!
//! Owner: metrale-stats.
//! Invariants: `x` and `y` are paired and equally long.

use alloc::vec::Vec;

use metrale_error::{ErrorCode, error};

use crate::Result;
use crate::dist::{norm_ppf, t_ppf};
use crate::interval::{CiMethod, Interval, Summary};
use crate::num::{check, f, mean, var};

/// 2026-10-22: A Deming fit `y ≈ intercept + slope·x`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Deming {
    pub slope: f64,
    pub intercept: f64,
    pub slope_ci: Interval,
    pub intercept_ci: Interval,
}

/// 2026-10-22: Bland-Altman agreement of `y − x`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlandAltman {
    pub bias: f64,
    pub sd: f64,
    pub loa_lo: f64,
    pub loa_hi: f64,
    pub bias_ci: Interval,
    pub n: usize,
}

fn pair(what: &'static str, x: &[f64], y: &[f64], need: usize) -> Result<()> {
    check(what, x, need)?;
    check(what, y, need)?;
    if x.len() != y.len() {
        return Err(error!(
            ErrorCode::StatsInvalid,
            what, "x and y differ in length"
        ));
    }
    Ok(())
}

fn fit(x: &[f64], y: &[f64], lambda: f64) -> Option<(f64, f64)> {
    let (mx, my) = (mean(x), mean(y));
    let n1 = f(x.len() - 1);
    let sxx = x.iter().map(|v| (v - mx) * (v - mx)).sum::<f64>() / n1;
    let syy = y.iter().map(|v| (v - my) * (v - my)).sum::<f64>() / n1;
    let sxy = x
        .iter()
        .zip(y)
        .map(|(a, b)| (a - mx) * (b - my))
        .sum::<f64>()
        / n1;
    if sxy == 0.0 {
        return None;
    }
    let d = syy - lambda * sxx;
    let slope = (d + libm::sqrt(d * d + 4.0 * lambda * sxy * sxy)) / (2.0 * sxy);
    Some((slope, my - slope * mx))
}

/// 2026-10-22: Deming regression of `y` on `x`; `lambda` = Var(error of y) / Var(error of x).
pub fn deming(x: &[f64], y: &[f64], lambda: f64) -> Result<Deming> {
    pair("deming", x, y, 3)?;
    if lambda.is_nan() || lambda <= 0.0 {
        return Err(error!(
            ErrorCode::StatsInvalid,
            "deming", "lambda must be positive"
        ));
    }
    let degenerate = || {
        error!(
            ErrorCode::StatsInvalid,
            "deming", "x and y are uncorrelated"
        )
    };
    let (slope, intercept) = fit(x, y, lambda).ok_or_else(degenerate)?;
    let n = x.len();
    let mut ps = Vec::with_capacity(n);
    let mut pi = Vec::with_capacity(n);
    let (mut xs, mut ys) = (Vec::with_capacity(n - 1), Vec::with_capacity(n - 1));
    for i in 0..n {
        xs.clear();
        ys.clear();
        xs.extend(
            x.iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, v)| *v),
        );
        ys.extend(
            y.iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, v)| *v),
        );
        let (s, b) = fit(&xs, &ys, lambda).ok_or_else(degenerate)?;
        ps.push(f(n) * slope - f(n - 1) * s);
        pi.push(f(n) * intercept - f(n - 1) * b);
    }
    let jack = |p: &[f64], est: f64| {
        Summary {
            mean: est,
            se: libm::sqrt(var(p) / f(n)),
            df: f(n - 1),
        }
        .interval(0.95, CiMethod::Jackknife)
    };
    Ok(Deming {
        slope,
        intercept,
        slope_ci: jack(&ps, slope),
        intercept_ci: jack(&pi, intercept),
    })
}

/// 2026-10-22: Bias, SD and the `level` limits of agreement of `y − x`, with the bias's t interval.
pub fn bland_altman(x: &[f64], y: &[f64], level: f64) -> Result<BlandAltman> {
    pair("bland_altman", x, y, 3)?;
    let d: Vec<f64> = y.iter().zip(x).map(|(b, a)| b - a).collect();
    let n = d.len();
    let bias = mean(&d);
    let sd = libm::sqrt(var(&d));
    let z = norm_ppf(0.5 + level / 2.0);
    let h = t_ppf(0.5 + level / 2.0, f(n - 1)) * sd / libm::sqrt(f(n));
    Ok(BlandAltman {
        bias,
        sd,
        loa_lo: bias - z * sd,
        loa_hi: bias + z * sd,
        bias_ci: Interval {
            lo: bias - h,
            hi: bias + h,
            level,
            method: CiMethod::T,
        },
        n,
    })
}

#[cfg(test)]
#[path = "agreement_tests.rs"]
mod tests;
