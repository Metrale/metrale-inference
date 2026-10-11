// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The runtime's binding of the QCI platform contracts v1 (investor-mvp #41):
//! `InferenceUsage`, `HardwareFingerprint` and `CompatibilityReceipt`, the launch
//! `ModelRecipe` the runtime reads ([`recipe`]), plus the shared
//! shapes they use (provenance [`field::Field`], fixed-width hex [`ids`], canonical JSON).
//!
//! Every field name, unit and closed word the runtime writes for these documents is in
//! this module and nowhere else. The canonical schemas live in `Metrale/mfs`
//! (`schemas/platform/v1`); the repository vendors that bundle unchanged under
//! `schemas/platform/` (pinned in `schemas/platform.pin`), and
//! `crates/qci-dump/tests/platform_contracts.rs` holds this binding to its fixtures. A
//! schema change is an edit to this module and a new vendored copy.
//!
//! Rules of the bundle (`schemas/platform/README.md`):
//! - Every document carries `version`; this binding writes [`PLATFORM_VERSION`].
//! - Objects are closed (`deny_unknown_fields`); optional fields are written as `null`.
//! - A quantity that may be unmeasured is a [`field::Field`]: absent is never zero.
//! - Integers carry their unit in the field name (`_ms`, `_us`, `_unix_ms`, `_bytes`).
//!
//! Owner: qci-dump (contract binding).
//! Invariants: no type here reads a clock, a file or the network.

pub mod canonical;
pub mod field;
pub mod hardware;
pub mod ids;
pub mod receipt;
pub mod recipe;
pub mod rules;
pub mod usage;

/// 2026-10-11: The `version` every document of this binding writes.
pub const PLATFORM_VERSION: u32 = 1;
