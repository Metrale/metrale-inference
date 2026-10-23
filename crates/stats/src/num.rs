// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Numeric conversions and the small helpers every module shares.
//!
//! Owner: metrale-stats.
//! Invariants: conversions are explicit about precision and saturation.

use alloc::vec::Vec;

use metrale_error::{ErrorCode, error};

use crate::Result;

/// 2026-10-22: A count as `f64`. Exact below 2^53, which every count here is.
pub(crate) fn f(n: usize) -> f64 {
    n as f64
}

/// 2026-10-22: `x` floored to a `usize`, saturating at 0 and `usize::MAX`; NaN is 0.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "saturating float-to-int conversion is the documented intent"
)]
pub(crate) fn floor_usize(x: f64) -> usize {
    if x.is_nan() || x <= 0.0 {
        0
    } else {
        libm::floor(x) as usize
    }
}

/// 2026-10-22: Refuse fewer than `need` values or any non-finite value.
pub(crate) fn check(what: &'static str, xs: &[f64], need: usize) -> Result<()> {
    if xs.len() < need {
        return Err(error!(ErrorCode::StatsInsufficient, what, need, xs.len()));
    }
    if xs.iter().any(|x| !x.is_finite()) {
        return Err(error!(
            ErrorCode::StatsInvalid,
            what, "a value is not finite"
        ));
    }
    Ok(())
}

pub(crate) fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / f(xs.len())
}

/// 2026-10-22: Sample variance (n − 1 denominator).
pub(crate) fn var(xs: &[f64]) -> f64 {
    let m = mean(xs);
    xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / f(xs.len() - 1)
}

pub(crate) fn sorted(xs: &[f64]) -> Vec<f64> {
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    v
}

/// 2026-10-22: The `q` quantile of sorted `v` by linear interpolation (type 7).
pub(crate) fn quantile_sorted(v: &[f64], q: f64) -> f64 {
    let h = f(v.len() - 1) * q.clamp(0.0, 1.0);
    let lo = floor_usize(h).min(v.len() - 1);
    let hi = (lo + 1).min(v.len() - 1);
    v[lo] + (h - f(lo)) * (v[hi] - v[lo])
}
