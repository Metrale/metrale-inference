// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The speculation controller: one model-agnostic policy for how many drafts a
//! step verifies (0 = plain decode). Three inputs and one rule:
//!
//! - an acceptance model ([`accept`]): decayed per-position conditional accept counts per
//!   stream, blended with a serve-wide prior;
//! - one step-cost model ([`cost`], [`calib`]): a measured table, else the circuit x hardware
//!   envelope, else an explicit cold-start prior, times an online calibration fed by measured
//!   step walls and joules;
//! - an explicit objective ([`decide::Objective`]): latency, throughput, or tokens per joule
//!   under a tokens/ms floor.
//!
//! The rule ([`decide::choose`]) maximises the objective over the candidate depths with
//! `E[tokens | K]` from [`chain`], with switch margins and re-probes ([`reprobe`]). Drafters
//! are data ([`source::DraftSource`]). Static ladders are explicit overrides or derived
//! cold-start priors ([`ladder`]). [`replay`] evaluates any policy on a recorded trace.
//!
//! Owner: speculative.
//! Invariants: every module here is pure: no I/O, no clock, no environment reads, no locks.
//! Hosts (the scheduler's rung and planner call sites) own state placement and measurement.

pub mod accept;
pub mod calib;
pub mod chain;
pub mod controller;
pub mod cost;
pub mod decide;
pub mod ladder;
pub mod measured;
pub mod online;
pub mod replay;
pub mod reprobe;
pub mod source;

#[cfg(test)]
mod replay_tests;
