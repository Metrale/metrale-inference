// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Contention-aware critical path and what-if simulation over an event capture
//! (`met circuit whatif`; claude-state GRAPH-THEORY-MAP experiment #1). The capture
//! ([`capture`]) becomes an event graph ([`dag`], [`graph`]); max-plus evaluation gives the
//! step, the critical chain and per-event slack ([`maxplus`]); a discrete-event simulation adds
//! SM residency and shared DRAM bandwidth ([`sim`]); rewrites change the graph ([`scenario`]);
//! [`analyze`] puts them together beside the headroom's Amdahl upside, [`backtest`] scores the
//! rewrites against serve measurements and [`render`] prints it.
//!
//! Owner: metrale-circuit (hardware).

pub mod analyze;
pub mod backtest;
pub mod capture;
pub mod dag;
pub mod graph;
pub mod maxplus;
pub mod render;
pub mod scenario;
mod scenario_ops;
pub mod sim;
pub mod sites;

#[cfg(test)]
mod critical_tests;
#[cfg(test)]
mod fixture;
#[cfg(test)]
mod scenario_tests;
#[cfg(test)]
mod sim_tests;
