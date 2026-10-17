// SPDX-License-Identifier: MIT OR Apache-2.0

#![deny(warnings)]
#![deny(clippy::all)]

//! 2026-10-17: The compiled hardware tree's serving defaults ([`TARGET_DEFAULTS`]), its SM
//! count ([`TARGET_SM_COUNT`]) and the paged-decode split-K policy that reads both.
//!
//! `build.rs` generates the two consts into `OUT_DIR/target_defaults.rs` from the
//! `kernels/<hw>/HARDWARE.toml` that `METRALE_TARGET_HW` selects. They were part of
//! metrale-kernels' generated registry; they live here so a crate that reads them does not
//! depend on the embedded kernel table and rebuild with every kernel edit.
//!
//! Owner: target-defaults crate.
//! Invariants:
//! - The generated text is the one metrale-kernels hashes into `KERNEL_SET_HASH`.

// 2026-09-25: The compiled target's serving defaults, baked from
// `kernels/<hw>/HARDWARE.toml` `[defaults]`. The type is declared here; the
// `TARGET_DEFAULTS` and `TARGET_SM_COUNT` consts are generated.
mod target_defaults;
pub use target_defaults::TargetDefaults;

// 2026-09-25: The paged-decode attention split-K policy. It lives in this
// crate because two crates need the same answer: the metrale-model-layers
// dispatch that picks `num_splits`, and the metrale-gpu-runtime buffer arena
// (`sizes.rs`) that sizes the split-K workspace. A workspace sized by a
// different rule than the dispatch would let the grid index past it.
pub mod attn_splitk;
pub use attn_splitk::{MAX_DECODE_SPLITS, SplitkPolicy};

// 2026-10-17: The generated `TARGET_DEFAULTS` and `TARGET_SM_COUNT`.
include!(concat!(env!("OUT_DIR"), "/target_defaults.rs"));
