// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Unit tests for `RadixTree`, and the helper that arms the sub-block tail arms on one tree.
//!
//! Owner: cache.
//! Invariants: none beyond the types.

mod adapter;
mod basic;
mod partial_tail;
mod snapshot;
mod snapshot_reap;

/// 2026-10-17: The declared `CacheTuning`, which the env-free index tests build with; it equals
/// what the deleted readers gave an empty environment (cache `legacy_env_tests.rs`).
pub(crate) fn declared_tuning() -> metrale_settings::groups::device_cache::CacheTuning {
    metrale_settings::groups::device_cache::CacheTuning::declared().expect("declared defaults")
}

/// 2026-09-25: Turn on the sub-block tail arms of `walk` on this tree only.
///
/// `RadixTree::new` copies `METRALE_PREFIX_SUBBLOCK` into
/// `partial_tail_sharing` (off when unset), so a test sets the field on its
/// own tree instead of the environment. `walk` reads the
/// same field in production.
pub(super) fn arm_legacy_partial_tail(tree: &super::RadixTree) {
    tree.inner.lock().partial_tail_sharing = true;
}
