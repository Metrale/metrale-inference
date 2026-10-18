// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The SSM snapshot spill tier: the model fingerprint, the capability gate,
//! and the keyed fixed-size blob stores an evicted snapshot spills into.
//!
//! Owner: model-engine SSM tier.
//! Invariants: none beyond the types.
//!
//! [`SnapshotBlobStore`] is the seam. [`build_tier_store`] picks a backend from the
//! resolved [`SsmTierLevers`]; `impl_a1_init::build_ssm_tier_store` calls it only when
//! `SsmTierLevers::enabled` (`METRALE_SSM_TIER`) and the model has SSM layers. Unless the
//! storage crate is built with both the `cuda` feature and RDMA verbs,
//! `metrale_storage::RdmaSnapshotArena` is a stub whose `connect` always errors, and the
//! selector falls back to host RAM.

mod arena_store;
mod capability;
mod fingerprint;
mod levers;
mod selectors;
mod store;
mod transport;
mod unified;

pub(crate) use arena_store::{ArenaSnapshotStore, PagingSnapshotStore, RdmaSnapshotStore};
pub(crate) use capability::ensure_ssm_tier_capability;
pub(crate) use fingerprint::ModelFingerprint;
pub(crate) use levers::SsmTierLevers;
pub(crate) use selectors::{build_decode_tier_store, build_tier_store};
pub(crate) use store::{BlobStoreStats, MemBlobStore, SnapshotBlobStore};
pub(crate) use transport::{
    FileSnapshotArena, MockSnapshotTransport, PagingTransport, SnapshotTransport,
};
pub(crate) use unified::UnifiedSnapshotStore;
