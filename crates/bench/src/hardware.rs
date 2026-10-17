// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The hardware fingerprint and state, and the bench's verdicts
//! on them.
//!
//! 2026-10-17: The probe itself ([`Hardware`], [`collect`], [`ids`],
//! [`parse`], [`state`], [`throttle_monitor`]) moved to `metrale-host-probe`,
//! which `met serve` shares without the bench; it is re-exported here so
//! bench paths are unchanged. What stays is the bench's: the energy meter,
//! the equivalence test, the HARDWARE.toml limits, the run policy and the
//! state report.
//!
//! Owner: bench (hardware).
//! Invariants:
//! - [`policy`] alone decides what a run is gated on.

pub use metrale_host_probe::hardware::{
    Hardware, HardwareState, HardwareStateDelta, collect, ids, parse, state, throttle_monitor,
};

pub mod energy;
pub mod energy_sampler;
pub mod equivalence;
pub mod limits;
pub mod policy;
pub mod report;

pub use policy::{Decision, Sensitivity, Validity};
pub use report::HardwareStateReport;
