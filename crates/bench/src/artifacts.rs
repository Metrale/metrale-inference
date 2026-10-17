// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The artifact home moved to `metrale-host-probe`, which
//! `met serve` shares without the bench; re-exported so bench paths are
//! unchanged.
//!
//! Owner: bench (artifacts).
//! Invariants: none beyond the re-exported types.

pub use metrale_host_probe::artifacts::{
    ArtifactStore, HomeFault, HomeSource, MetraleHome, Stamp, write_asset, write_asset_bytes,
};
