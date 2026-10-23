// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The reporting record: every derived row carries its estimate, the interval (method
//! and level), n at each level of the hierarchy, the effective sample size, the outliers it
//! flagged (never dropped), the test it used and the seed of any resampling.
//!
//! Owner: metrale-stats.
//! Invariants: none beyond the types.

use alloc::vec::Vec;

use crate::interval::{Interval, Level};

#[derive(Clone, Debug, PartialEq)]
pub struct Estimate {
    pub value: f64,
    pub ci: Interval,
    /// 2026-10-22: Observations at each level, e.g. `[(Box, 1), (Run, 3), (Sample, 1440)]`.
    pub n: Vec<(Level, u64)>,
    /// 2026-10-22: Effective sample size of the autocorrelated series behind the value, when
    /// one is.
    pub ess: Option<f64>,
    /// 2026-10-22: Indices flagged by the declared outlier rule (kept in the estimate).
    pub outliers: Vec<usize>,
    /// 2026-10-22: The hypothesis test applied, if any (e.g. `tost`).
    pub test: Option<&'static str>,
    /// 2026-10-22: The resampling seed, when the interval is a bootstrap one.
    pub seed: Option<u64>,
}
