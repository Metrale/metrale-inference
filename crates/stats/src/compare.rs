// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Comparisons for A/B rows and regression gates. "No regression" is an equivalence
//! claim and is tested as one: Schuirmann's (1987) two one-sided tests (TOST) against a declared
//! margin; a non-significant difference is never evidence of equivalence. "Win" is a one-sided
//! superiority test against a margin. Many kernels at once are controlled with Holm's (1979)
//! step-down (family-wise error) or Benjamini & Hochberg's (1995) step-up (false discovery
//! rate). Before "no difference" is claimed, the minimum detectable effect at the run's SE
//! states what the run could have seen.
//!
//! Owner: metrale-stats.
//! Invariants: tests take the difference as a [`Summary`] (estimate, SE, df).

use alloc::vec;
use alloc::vec::Vec;

use crate::dist::{t_cdf, t_ppf};
use crate::interval::Summary;
use crate::num::f;

/// 2026-10-22: A TOST result: the larger one-sided p, and whether equivalence is shown at α.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tost {
    pub p: f64,
    pub equivalent: bool,
}

/// 2026-10-22: A one-sided test result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Test {
    pub p: f64,
    pub reject: bool,
}

/// 2026-10-22: TOST of |diff| < `margin` at level `alpha`.
#[must_use]
pub fn tost(diff: Summary, margin: f64, alpha: f64) -> Tost {
    let lower = 1.0 - t_cdf((diff.mean + margin) / diff.se, diff.df);
    let upper = t_cdf((diff.mean - margin) / diff.se, diff.df);
    let p = lower.max(upper);
    Tost {
        p,
        equivalent: p < alpha,
    }
}

/// 2026-10-22: One-sided test of H1: diff > `margin`.
#[must_use]
pub fn superiority(diff: Summary, margin: f64, alpha: f64) -> Test {
    let p = 1.0 - t_cdf((diff.mean - margin) / diff.se, diff.df);
    Test {
        p,
        reject: p < alpha,
    }
}

/// 2026-10-22: `a − b` with the Welch-Satterthwaite df (an infinite-df side contributes no df
/// term).
#[must_use]
pub fn welch(a: Summary, b: Summary) -> Summary {
    let (va, vb) = (a.se * a.se, b.se * b.se);
    let term = |v: f64, df: f64| if df.is_finite() { v * v / df } else { 0.0 };
    let den = term(va, a.df) + term(vb, b.df);
    Summary {
        mean: a.mean - b.mean,
        se: libm::sqrt(va + vb),
        df: if den > 0.0 {
            (va + vb) * (va + vb) / den
        } else {
            f64::INFINITY
        },
    }
}

fn order(p: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    idx.sort_by(|&i, &j| p[i].total_cmp(&p[j]));
    idx
}

/// 2026-10-22: Holm's step-down rejections at family-wise level `alpha`, in input order.
#[must_use]
pub fn holm(p: &[f64], alpha: f64) -> Vec<bool> {
    let m = p.len();
    let mut out = vec![false; m];
    for (rank, &i) in order(p).iter().enumerate() {
        if p[i] > alpha / f(m - rank) {
            break;
        }
        out[i] = true;
    }
    out
}

/// 2026-10-22: Benjamini-Hochberg rejections at false discovery rate `q`, in input order.
#[must_use]
pub fn benjamini_hochberg(p: &[f64], q: f64) -> Vec<bool> {
    let m = p.len();
    let idx = order(p);
    let k = (0..m).rev().find(|&r| p[idx[r]] <= f(r + 1) * q / f(m));
    let mut out = vec![false; m];
    if let Some(k) = k {
        for &i in &idx[..=k] {
            out[i] = true;
        }
    }
    out
}

/// 2026-10-22: The two-sided minimum detectable effect: (t_{1−α/2} + t_{power})·se.
#[must_use]
pub fn mde(se_diff: f64, df: f64, alpha: f64, power: f64) -> f64 {
    (t_ppf(1.0 - alpha / 2.0, df) + t_ppf(power, df)) * se_diff
}

#[cfg(test)]
#[path = "compare_tests.rs"]
mod tests;
