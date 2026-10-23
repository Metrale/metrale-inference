// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Robust location and the declared outlier rule. The median and MAD are
//! breakdown-50% anchors; Hodges-Lehmann (median of Walsh averages) keeps ~95% of the mean's
//! efficiency on normal data with a 29% breakdown; Huber's M-estimator (k = 1.345, MAD scale,
//! IRLS) is 95% efficient at the normal and bounded-influence. The outlier rule is
//! Iglewicz-Hoaglin's modified z-score: |x − median| / (1.4826 · MAD) > 3.5. Flagged values are
//! reported, never removed.
//!
//! Owner: metrale-stats.
//! Invariants: inputs are finite; an empty input is `StatsInsufficient`.

use alloc::vec::Vec;

use metrale_error::{ErrorCode, error};

use crate::Result;
use crate::num::{check, f, sorted};

/// 2026-10-22: MAD × 1.4826 estimates σ for normal data (1 / Φ⁻¹(3/4)).
pub const MAD_TO_SIGMA: f64 = 1.4826;
/// 2026-10-22: Huber's tuning constant: 95% asymptotic efficiency at the normal.
pub const HUBER_K: f64 = 1.345;
/// 2026-10-22: Mean absolute deviation × 1.2533 estimates σ for normal data (√(π/2)); the
/// outlier rule's scale when the MAD is 0.
const MEANAD_TO_SIGMA: f64 = 1.253_314;

fn median_sorted(v: &[f64]) -> f64 {
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// 2026-10-22: The sample median.
pub fn median(xs: &[f64]) -> Result<f64> {
    check("median", xs, 1)?;
    Ok(median_sorted(&sorted(xs)))
}

/// 2026-10-22: The raw median absolute deviation (multiply by [`MAD_TO_SIGMA`] for σ).
pub fn mad(xs: &[f64]) -> Result<f64> {
    let m = median(xs)?;
    let dev: Vec<f64> = xs.iter().map(|x| (x - m).abs()).collect();
    median(&dev)
}

/// 2026-10-22: The Hodges-Lehmann estimator: the median of the n(n+1)/2 Walsh averages.
pub fn hodges_lehmann(xs: &[f64]) -> Result<f64> {
    check("hodges_lehmann", xs, 1)?;
    let mut w = Vec::with_capacity(xs.len() * (xs.len() + 1) / 2);
    for i in 0..xs.len() {
        for j in i..xs.len() {
            w.push(0.5 * (xs[i] + xs[j]));
        }
    }
    median(&w)
}

/// 2026-10-22: Huber's M-estimator of location with tuning `k` and the MAD scale, by iteratively
/// reweighted least squares from the median; `StatsNoConvergence` after 200 iterations.
pub fn huber(xs: &[f64], k: f64) -> Result<f64> {
    check("huber", xs, 1)?;
    if k.is_nan() || k <= 0.0 {
        return Err(error!(
            ErrorCode::StatsInvalid,
            "huber", "k must be positive"
        ));
    }
    let mut mu = median(xs)?;
    let s = MAD_TO_SIGMA * mad(xs)?;
    if s == 0.0 {
        return Ok(mu);
    }
    for _ in 0..200 {
        let (mut num, mut den) = (0.0, 0.0);
        for &x in xs {
            let r = ((x - mu) / s).abs();
            let w = if r <= k { 1.0 } else { k / r };
            num += w * x;
            den += w;
        }
        let next = num / den;
        if (next - mu).abs() <= 1e-12 * s {
            return Ok(next);
        }
        mu = next;
    }
    Err(error!(ErrorCode::StatsNoConvergence, "huber", 200u32))
}

/// 2026-10-22: The declared outlier rule: modified z-score above `z`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutlierRule {
    pub z: f64,
}

impl OutlierRule {
    /// 2026-10-22: Iglewicz & Hoaglin (1993): |x − median| / (1.4826 · MAD) > 3.5.
    pub const DECLARED: OutlierRule = OutlierRule { z: 3.5 };
}

/// 2026-10-22: Indices of the values `rule` flags. When the MAD is 0 the scale is the mean
/// absolute deviation × 1.2533; when that is 0 too (all equal) nothing is flagged.
#[must_use]
pub fn flag_outliers(xs: &[f64], rule: &OutlierRule) -> Vec<usize> {
    let (Ok(m), Ok(d)) = (median(xs), mad(xs)) else {
        return Vec::new();
    };
    let mut s = MAD_TO_SIGMA * d;
    if s == 0.0 {
        s = MEANAD_TO_SIGMA * xs.iter().map(|x| (x - m).abs()).sum::<f64>() / f(xs.len());
    }
    if s == 0.0 {
        return Vec::new();
    }
    xs.iter()
        .enumerate()
        .filter(|(_, x)| ((*x - m) / s).abs() > rule.z)
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
#[path = "robust_tests.rs"]
mod tests;
