// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The interval and summary records every estimator returns, and the names stored
//! beside them (a CI is meaningless without its method and level).
//!
//! Owner: metrale-stats.
//! Invariants: `lo <= hi` for every interval an estimator returns.

use crate::dist::t_ppf;

/// 2026-10-22: How an interval was computed (stored as `as_str`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CiMethod {
    /// 2026-10-22: Student t with an i.i.d. standard error.
    T,
    /// 2026-10-22: Student t with an autocorrelation-consistent (HAC) standard error and the
    /// effective-sample-size df.
    HacT,
    /// 2026-10-22: Bias-corrected and accelerated bootstrap over dependence-preserving blocks.
    BcaBlockBootstrap,
    /// 2026-10-22: Percentile bootstrap over dependence-preserving blocks.
    PercentileBlockBootstrap,
    /// 2026-10-22: Random-effects variance components, Satterthwaite df.
    VarCompSatterthwaite,
    /// 2026-10-22: First-order delta method.
    Delta,
    /// 2026-10-22: Jackknife pseudo-values, t with n − 1 df.
    Jackknife,
}

impl CiMethod {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::T => "t",
            Self::HacT => "hac-t",
            Self::BcaBlockBootstrap => "bca-block-bootstrap",
            Self::PercentileBlockBootstrap => "percentile-block-bootstrap",
            Self::VarCompSatterthwaite => "varcomp-satterthwaite",
            Self::Delta => "delta",
            Self::Jackknife => "jackknife",
        }
    }
}

/// 2026-10-22: The levels of the measurement hierarchy: samples within windows within runs
/// within boxes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Level {
    Sample,
    Window,
    Run,
    Box,
}

impl Level {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sample => "sample",
            Self::Window => "window",
            Self::Run => "run",
            Self::Box => "box",
        }
    }
}

/// 2026-10-22: A confidence interval with its level and method.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
    pub level: f64,
    pub method: CiMethod,
}

impl Interval {
    #[must_use]
    pub fn contains(&self, x: f64) -> bool {
        self.lo <= x && x <= self.hi
    }

    #[must_use]
    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }
}

/// 2026-10-22: An estimate with its standard error and degrees of freedom (`INFINITY`: normal).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Summary {
    pub mean: f64,
    pub se: f64,
    pub df: f64,
}

impl Summary {
    /// 2026-10-22: `mean ± t_{(1+level)/2, df} · se`, labelled `method`.
    #[must_use]
    pub fn interval(&self, level: f64, method: CiMethod) -> Interval {
        let h = t_ppf(0.5 + level / 2.0, self.df) * self.se;
        Interval {
            lo: self.mean - h,
            hi: self.mean + h,
            level,
            method,
        }
    }
}
