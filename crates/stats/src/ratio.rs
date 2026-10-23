// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Error propagation for derived ratios and products (J/tok = J/launch ÷ rows,
//! $/Mtok = J/tok × price): the first-order delta method, reported beside a bootstrap interval;
//! [`agree`] flags when the two disagree (a sign the ratio's distribution is skewed or its
//! denominator noisy, where the delta method is unreliable).
//!
//! Owner: metrale-stats.
//! Invariants: df combine as Welch-Satterthwaite over the two variance terms (covariance is
//! taken as known).

use crate::interval::{Interval, Summary};

fn sat(t1: f64, df1: f64, t2: f64, df2: f64) -> f64 {
    let term = |t: f64, df: f64| if df.is_finite() { t * t / df } else { 0.0 };
    let den = term(t1, df1) + term(t2, df2);
    if den > 0.0 {
        (t1 + t2) * (t1 + t2) / den
    } else {
        f64::INFINITY
    }
}

/// 2026-10-22: num / den with Var ≈ r²(σ²ₙ/μ²ₙ + σ²_d/μ²_d − 2·cov/(μₙμ_d)).
#[must_use]
pub fn delta_ratio(num: Summary, den: Summary, cov: f64) -> Summary {
    let r = num.mean / den.mean;
    let tn = num.se * num.se / (num.mean * num.mean);
    let td = den.se * den.se / (den.mean * den.mean);
    let v = r * r * (tn + td - 2.0 * cov / (num.mean * den.mean));
    Summary {
        mean: r,
        se: libm::sqrt(v.max(0.0)),
        df: sat(r * r * tn, num.df, r * r * td, den.df),
    }
}

/// 2026-10-22: a · b with Var ≈ μ_b²σ²_a + μ_a²σ²_b + 2μ_aμ_b·cov.
#[must_use]
pub fn delta_product(a: Summary, b: Summary, cov: f64) -> Summary {
    let (ta, tb) = (b.mean * b.mean * a.se * a.se, a.mean * a.mean * b.se * b.se);
    let v = ta + tb + 2.0 * a.mean * b.mean * cov;
    Summary {
        mean: a.mean * b.mean,
        se: libm::sqrt(v.max(0.0)),
        df: sat(ta, a.df, tb, b.df),
    }
}

/// 2026-10-22: Whether two intervals of one quantity agree: each endpoint of `delta` within
/// `tol_rel` × the bootstrap interval's width of the matching endpoint of `boot`.
#[must_use]
pub fn agree(delta: &Interval, boot: &Interval, tol_rel: f64) -> bool {
    let w = boot.width().abs().max(f64::MIN_POSITIVE);
    (delta.lo - boot.lo).abs() <= tol_rel * w && (delta.hi - boot.hi).abs() <= tol_rel * w
}

#[cfg(test)]
#[path = "ratio_tests.rs"]
mod tests;
