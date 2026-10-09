// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Bounded values: the arithmetic every reference is written in, so a tolerance is
//! derived from the declared formats and accumulation order instead of guessed.
//!
//! A [`Bounded`] holds `v`, the exact value of the declared computation on the exact operands
//! (evaluated in f64), and `e`, a bound on `|kernel - v|` for every kernel that performs the
//! declared roundings: each [`Bounded::round`] is a point where the kernel rounds into a format,
//! each [`sum`] a reduction the kernel may bracket in any tree no deeper than the declared depth,
//! each approximate function a declared relative error. The bound is worst case (no
//! probabilistic factor), so a kernel that does what its contract declares cannot exceed it: a
//! violation is a numerics change, never noise.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - `e` is a rigorous upper bound: second-order terms are kept, f64 evaluation of `v` and `e`
//!   is covered by [`slack`].
//! - A bound that cannot be stated (a division by an interval containing zero, an `rsqrt` of an
//!   interval reaching zero, `n*u >= 1`) is `f64::INFINITY`; the comparison refuses an infinite
//!   bound as vacuous rather than passing it.

use crate::elem::{Elem, F64};

/// 2026-10-09: A value of the declared computation and the bound on a conforming kernel's
/// deviation from it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounded {
    /// 2026-10-09: Exact value (f64 evaluation of the exact-arithmetic computation).
    pub v: f64,
    /// 2026-10-09: Bound on `|kernel value - v|`.
    pub e: f64,
}

/// 2026-10-09: `gamma_n(u) = n*u / (1 - n*u)`, the relative error of `n` chained roundings with
/// unit roundoff `u` (infinite once `n*u >= 1`).
pub fn gamma(n: u64, u: f64) -> f64 {
    let nu = n as f64 * u;
    if nu >= 1.0 {
        f64::INFINITY
    } else {
        nu / (1.0 - nu)
    }
}

/// 2026-10-09: Widen a bound computed in f64 so the f64 evaluation of `v` (one rounding of
/// relative size `2^-53` per op) and of `e` itself cannot make it optimistic.
fn slack(v: f64, e: f64) -> f64 {
    let u = F64.unit_roundoff();
    (e + 4.0 * u * v.abs()) * (1.0 + 4.0 * u)
}

impl Bounded {
    /// 2026-10-09: An operand the kernel reads exactly (a decoded weight, an input element).
    pub fn exact(v: f64) -> Self {
        Bounded { v, e: 0.0 }
    }

    /// 2026-10-09: A bound on the magnitude of the kernel's value.
    pub fn mag(&self) -> f64 {
        self.v.abs() + self.e
    }

    /// 2026-10-09: The kernel rounds its value into `fmt` (RNE): adds `u*|x| + floor`, where
    /// the floor covers the subnormal range.
    pub fn round(self, fmt: Elem) -> Self {
        let e = self.e + fmt.unit_roundoff() * self.mag() + fmt.underflow_floor();
        Bounded {
            v: self.v,
            e: slack(self.v, e),
        }
    }

    /// 2026-10-09: Exact sum of two kernel values (the rounding, if any, is a separate
    /// [`Bounded::round`] or part of a [`sum`]).
    pub fn add(self, o: Self) -> Self {
        let v = self.v + o.v;
        Bounded {
            v,
            e: slack(v, self.e + o.e),
        }
    }

    /// 2026-10-09: Exact difference.
    pub fn sub(self, o: Self) -> Self {
        self.add(o.neg())
    }

    /// 2026-10-09: Negation (exact in every format).
    pub fn neg(self) -> Self {
        Bounded {
            v: -self.v,
            e: self.e,
        }
    }

    /// 2026-10-09: Exact product: `|ab - a'b'| <= |a|e_b + |b|e_a + e_a e_b`.
    pub fn mul(self, o: Self) -> Self {
        let v = self.v * o.v;
        let e = self.v.abs() * o.e + o.v.abs() * self.e + self.e * o.e;
        Bounded { v, e: slack(v, e) }
    }

    /// 2026-10-09: Exact quotient, defined while `|o.v| > o.e`.
    pub fn div(self, o: Self) -> Self {
        let v = self.v / o.v;
        let lo = o.v.abs() - o.e;
        if lo <= 0.0 {
            return Bounded {
                v,
                e: f64::INFINITY,
            };
        }
        let e = (self.v.abs() * o.e + o.v.abs() * self.e) / (o.v.abs() * lo);
        Bounded { v, e: slack(v, e) }
    }

    /// 2026-10-09: `exp`, computed by the kernel with relative error at most `rel` (an
    /// approximate instruction's documented bound, or the format's rounding for an exact one).
    pub fn exp(self, rel: f64) -> Self {
        let v = self.v.exp();
        let e = v * self.e.exp_m1() + (self.v + self.e).exp() * rel;
        Bounded { v, e: slack(v, e) }
    }

    /// 2026-10-09: `1/sqrt(x)` with relative error at most `rel`, defined while `v > e`.
    pub fn rsqrt(self, rel: f64) -> Self {
        let v = 1.0 / self.v.sqrt();
        let lo = self.v - self.e;
        if lo <= 0.0 {
            return Bounded {
                v,
                e: f64::INFINITY,
            };
        }
        let worst = 1.0 / lo.sqrt();
        let e = (worst - v) + worst * rel;
        Bounded { v, e: slack(v, e) }
    }

    /// 2026-10-09: The larger of two kernel values (a max is exact; the bound is the larger).
    pub fn max(self, o: Self) -> Self {
        Bounded {
            v: self.v.max(o.v),
            e: self.e.max(o.e),
        }
    }

    /// 2026-10-09: A function with Lipschitz constant `lip` on the bound's interval (SiLU: 1.1,
    /// sigmoid: 0.25), evaluated with relative error at most `rel`.
    pub fn lipschitz(self, f: impl Fn(f64) -> f64, lip: f64, rel: f64) -> Self {
        let v = f(self.v);
        let e = lip * self.e + (v.abs() + lip * self.e) * rel;
        Bounded { v, e: slack(v, e) }
    }
}

/// 2026-10-09: A reduction the kernel performs in `acc` with any bracketing whose tree depth is
/// at most `depth` (a sequential sum of n terms has depth n; a shuffle tree over 32 lanes, 5):
/// `|s_kernel - s| <= sum e_i + gamma_depth(u) * sum |x_i| + adds * floor`.
pub fn sum(terms: &[Bounded], acc: Elem, depth: u64) -> Bounded {
    let mut v = 0.0;
    let (mut carried, mut mags) = (0.0, 0.0);
    for t in terms {
        v += t.v;
        carried += t.e;
        mags += t.mag();
    }
    let floors = terms.len() as f64 * acc.underflow_floor();
    // 2026-10-09: The f64 evaluation of `v` is itself a sequential sum: cover it the same way.
    let f64_sum = gamma(terms.len() as u64, F64.unit_roundoff()) * mags;
    let e = carried + gamma(depth, acc.unit_roundoff()) * mags + floors + f64_sum;
    Bounded { v, e: slack(v, e) }
}

#[cfg(test)]
#[path = "bounded_tests.rs"]
mod tests;
