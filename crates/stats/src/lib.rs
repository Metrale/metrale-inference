// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Measurement statistics for energy and speed rows (owner addendum 2026-10-22,
//! DESIGN-kernel-energy.md statistics section): distributions, robust location with a declared
//! outlier rule, autocorrelation-consistent (HAC) variance and effective sample size, seeded
//! resampling (moving-block, stationary and BCa bootstrap), random-effects variance components
//! for samples within runs within boxes, errors-in-variables agreement (Deming, Bland-Altman),
//! non-negative least squares with collinearity checks, equivalence / superiority tests with
//! multiplicity control and power analysis, the delta method, change-point detection and the
//! interleaved measurement order.
//!
//! Owner: metrale-stats.
//! Invariants:
//! - Pure: `no_std` + `alloc`, no I/O, no clock, no global state; randomness only from a seeded
//!   [`rng::SplitMix64`], so every interval is reproducible from its recorded seed (HPAR).
//! - Every fallible function returns a registry error (`StatsInsufficient`, `StatsInvalid`,
//!   `StatsNoConvergence`); nothing panics on caller data.
//! - Data are never dropped: outliers are flagged and reported, not removed.
#![no_std]
#![forbid(unsafe_code)]
#![deny(warnings)]
#![deny(clippy::all)]
#![deny(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

extern crate alloc;

pub mod agreement;
pub mod bootstrap;
pub mod changepoint;
pub mod compare;
pub mod design;
pub mod dist;
pub mod estimate;
pub mod hac;
pub mod interval;
pub mod mbb;
pub mod nnls;
pub mod ratio;
pub mod rng;
pub mod robust;
pub mod varcomp;

mod linalg;
mod num;

pub use estimate::Estimate;
pub use interval::{CiMethod, Interval, Level, Summary};

/// 2026-10-22: The result every fallible statistic returns.
pub type Result<T> = core::result::Result<T, metrale_error::Error>;

#[cfg(test)]
mod sim;
