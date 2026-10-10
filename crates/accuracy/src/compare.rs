// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Comparisons: a kernel output against a bounded reference (max error, max ratio
//! of error to bound, where), and two outputs byte for byte. Neither builds a verdict on an
//! empty, non-finite or trivial comparison: those are typed refusals, never passes.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - `ratio <= 1` everywhere is the derived contract; a NaN or infinite kernel value where the
//!   reference is finite is an infinite ratio, never skipped.

use crate::bounded::Bounded;
use crate::elem::Elem;

/// 2026-10-09: The result of a bounded comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    /// 2026-10-09: Largest distance from `v` to the reals that round to the kernel value.
    pub max_err: f64,
    /// 2026-10-09: Largest `|kernel - v| / e`.
    pub max_ratio: f64,
    /// 2026-10-09: Position (in the compared list) of the largest ratio.
    pub worst: usize,
    /// 2026-10-09: Elements compared.
    pub compared: usize,
    /// 2026-10-09: Elements whose kernel value is not a correct rounding of `v` (a positive
    /// error): the drift statistic, which a legitimate reordering moves by a few elements and a
    /// numerics change by many.
    pub misrounded: usize,
}

/// 2026-10-09: A comparison that cannot support a verdict.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Vacuous {
    /// 2026-10-09: Nothing was compared.
    #[error("nothing was compared")]
    Empty,
    /// 2026-10-09: The two sides differ in length.
    #[error("the kernel produced {got} values for {want} reference values")]
    Length {
        /// 2026-10-09: Kernel side.
        got: usize,
        /// 2026-10-09: Reference side.
        want: usize,
    },
    /// 2026-10-09: The bound is infinite or NaN somewhere: the contract states no tolerance there.
    #[error("the derived bound is not finite at {0} of the compared elements")]
    Unbounded(usize),
    /// 2026-10-09: Every reference value is zero (a zero output cannot tell a kernel from none).
    #[error("every reference value is zero")]
    Trivial,
}

/// 2026-10-09: Compare kernel values `got`, written in `out`, with the reference `want` taken
/// before that final rounding. The error of an element is the distance from `v` to the set of
/// reals that round to the kernel's value (zero when the kernel returned a correct rounding of
/// any value within the bound), so the output rounding itself never counts against the margin
/// and the ratio measures only what the kernel did before it.
pub fn bounded(got: &[f64], want: &[Bounded], out: Elem) -> Result<Bounds, Vacuous> {
    if want.is_empty() {
        return Err(Vacuous::Empty);
    }
    if got.len() != want.len() {
        return Err(Vacuous::Length {
            got: got.len(),
            want: want.len(),
        });
    }
    let unbounded = want
        .iter()
        .filter(|w| !w.e.is_finite() || !w.v.is_finite())
        .count();
    if unbounded > 0 {
        return Err(Vacuous::Unbounded(unbounded));
    }
    if want.iter().all(|w| w.v == 0.0) {
        return Err(Vacuous::Trivial);
    }
    let mut b = Bounds {
        max_err: 0.0,
        max_ratio: 0.0,
        worst: 0,
        compared: want.len(),
        misrounded: 0,
    };
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = if g.is_finite() {
            let (lo, hi) = out.preimage(*g);
            (lo - w.v).max(w.v - hi).max(0.0)
        } else {
            f64::INFINITY
        };
        let ratio = if err == 0.0 {
            0.0
        } else if w.e > 0.0 {
            err / w.e
        } else {
            f64::INFINITY
        };
        if err > 0.0 {
            b.misrounded += 1;
        }
        if err > b.max_err {
            b.max_err = err;
        }
        if ratio > b.max_ratio || (i == 0 && ratio == 0.0) {
            b.max_ratio = ratio;
            b.worst = i;
        }
    }
    Ok(b)
}

/// 2026-10-09: A byte comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bytes {
    /// 2026-10-09: Differing bytes.
    pub differing: usize,
    /// 2026-10-09: First differing offset.
    pub first: Option<usize>,
    /// 2026-10-09: Bytes compared.
    pub compared: usize,
}

/// 2026-10-09: Compare two outputs byte for byte; both must be non-empty, equally long, and not
/// both entirely zero.
pub fn bytes(a: &[u8], b: &[u8]) -> Result<Bytes, Vacuous> {
    if a.is_empty() {
        return Err(Vacuous::Empty);
    }
    if a.len() != b.len() {
        return Err(Vacuous::Length {
            got: a.len(),
            want: b.len(),
        });
    }
    if a.iter().all(|&x| x == 0) && b.iter().all(|&x| x == 0) {
        return Err(Vacuous::Trivial);
    }
    let diff: Vec<usize> = a
        .iter()
        .zip(b)
        .enumerate()
        .filter(|(_, (x, y))| x != y)
        .map(|(i, _)| i)
        .collect();
    Ok(Bytes {
        differing: diff.len(),
        first: diff.first().copied(),
        compared: a.len(),
    })
}

#[cfg(test)]
#[path = "compare_tests.rs"]
mod tests;
