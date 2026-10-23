// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Per-kernel energy, J and W (DESIGN-kernel-energy.md, P1–P2): the pure core over the
//! power port of `metrale-abstract-io`.
//!
//! - [`trace`]: the raw trace (every source's capabilities and samples, the windows, the launch
//!   checkpoints, the run's metadata) and its text form; [`probe`] imports the 2026-10-22 probe.
//! - [`integrate`], [`steady`], [`policy`]: a channel's energy over a window; one repeat's
//!   steady-state attribution with interleaved baselines; the window rules and refusals.
//! - [`derive`]: a trace to its measurement row ([`row`]), with the statistics of
//!   `metrale-stats` ([`boot`] is the hierarchical bootstrap); [`platform`] declares each
//!   platform's headline order.
//! - [`inserve`]: the in-serve NNLS validator; [`cost`]: $/Mtok, electricity and hardware.
//! - [`meter`]: the replay and scripted meters; [`multi`]: sampling every meter at once;
//!   [`counter`]: the counter tracker shared with serve telemetry; [`external`]: external meter
//!   protocols.
//!
//! Owner: metrale-power.
//! Invariants: no I/O, no clock, no randomness but the seeded generator; `no_std` + `alloc`;
//! builds for both wasm32 targets (NSCC).
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

pub mod boot;
pub mod cost;
pub mod counter;
pub mod derive;
mod derive_pairs;
pub mod external;
pub mod inserve;
pub mod integrate;
pub mod meter;
pub mod multi;
pub mod platform;
pub mod policy;
pub mod probe;
pub mod row;
pub mod steady;
pub mod trace;
mod trace_sample;

/// 2026-10-22: The result every fallible function returns.
pub type Result<T> = core::result::Result<T, metrale_error::Error>;

// 2026-10-22: Test fixtures build windows and launch counts from seconds and repeat indices.
#[cfg(test)]
mod cost_tests;
#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
mod derive_tests;
#[cfg(test)]
mod inserve_tests;
#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
mod meter_tests;
#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
mod probe_tests;
#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
mod synth_tests;
#[cfg(test)]
mod trace_tests;
