// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The moving-block bootstrap surface the headroom reports were built on (Künsch
//! 1989), with the names and arithmetic of the interim `metrale-circuit` `hardware/stats.rs` it
//! replaces, so the headroom golden reports stay byte-identical: a segment-respecting moving-block
//! resample, its block rule, the percentile interval, the median and the Newey-West variance of a
//! mean. Each delegates to this crate's single implementation where that is bit-identical for
//! finite data (`mbb_tests.rs` pins it against the interim formulas):
//!
//! - [`median`] is [`crate::robust::median`] (`0.5·(a+b)` and `(a+b)/2` are the same IEEE result).
//! - [`percentile_ci`] uses the crate's type-7 quantile, the interim interpolation term for term.
//! - [`block_resample`] is [`crate::bootstrap::moving_block_indices`] per segment (the same draws).
//! - [`hac_variance_of_mean`] is [`crate::hac::newey_west_var_of_mean`].
//!
//! One difference, on non-finite input only: those functions refuse NaN/∞, so `median` and
//! `hac_variance_of_mean` return `None` where the interim code returned a NaN.
//!
//! [`block_len`] is ⌈n^(1/3)⌉ (the interim rule); [`crate::bootstrap::block_length`] rounds
//! n^(1/3) to nearest. Both are the Hall-Horowitz-Jing n^(1/3) rate; they differ by at most 1, and
//! each caller keeps the one its records were made with.
//!
//! Owner: metrale-stats.
//! Invariants: deterministic from the seed; a block never crosses a segment boundary.

use alloc::vec::Vec;

use crate::bootstrap::moving_block_indices;
use crate::num::{f, floor_usize, quantile_sorted, sorted};

pub use crate::rng::SplitMix64;

/// 2026-10-22: A two-sided interval (bounds only).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    /// 2026-10-22: The two intervals share a point.
    #[must_use]
    pub fn overlaps(&self, o: &Interval) -> bool {
        self.lo <= o.hi && o.lo <= self.hi
    }
}

/// 2026-10-22: The median (`None` when empty or not finite).
#[must_use]
pub fn median(xs: &[f64]) -> Option<f64> {
    crate::robust::median(xs).ok()
}

/// 2026-10-22: ⌈n^(1/3)⌉, at least 1 and at most max(n, 1).
#[must_use]
pub fn block_len(n: usize) -> usize {
    floor_usize(libm::ceil(libm::cbrt(f(n))))
        .max(1)
        .min(n.max(1))
}

/// 2026-10-22: One moving-block resample of the indices of `segments` (consecutive segment
/// lengths): each segment rebuilt to its own length from blocks of `block` consecutive indices
/// drawn uniformly inside it, the last truncated.
#[must_use]
pub fn block_resample(segments: &[usize], block: usize, rng: &mut SplitMix64) -> Vec<usize> {
    let mut out = Vec::with_capacity(segments.iter().sum());
    let mut base = 0;
    for &len in segments {
        out.extend(
            moving_block_indices(len, block, rng)
                .into_iter()
                .map(|i| base + i),
        );
        base += len;
    }
    out
}

/// 2026-10-22: The percentile interval at `level` of bootstrap replicates (linear interpolation
/// between order statistics); `None` when there are none.
#[must_use]
pub fn percentile_ci(reps: &[f64], level: f64) -> Option<Interval> {
    if reps.is_empty() {
        return None;
    }
    let v = sorted(reps);
    let a = (1.0 - level) / 2.0;
    Some(Interval {
        lo: quantile_sorted(&v, a),
        hi: quantile_sorted(&v, 1.0 - a),
    })
}

/// 2026-10-22: The Newey-West (Bartlett) variance of the mean with `lag` autocovariances;
/// `None` for fewer than two (or non-finite) observations.
#[must_use]
pub fn hac_variance_of_mean(xs: &[f64], lag: usize) -> Option<f64> {
    crate::hac::newey_west_var_of_mean(xs, lag).ok()
}

#[cfg(test)]
#[path = "mbb_tests.rs"]
mod tests;
