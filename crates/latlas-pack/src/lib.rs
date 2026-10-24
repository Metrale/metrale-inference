// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The metrale pack for Latlas (DESIGN-latlas-ontology.md §4.3): the campaign records
//! (LEDGER, TOOLING, CHARTER, RUNS, designs, research maps) and the LKB (families, rules,
//! critical pairs, measurements, instances, circuits, plans, headroom) read into a Latlas bundle,
//! with the ledger closures `ledger.py` agrees with byte for byte. Was `metrale-kg-derive`; the
//! generic core moved to `latlas-core` (K0/K1).
//!
//! Owner: metrale-latlas-pack.
//! Invariants:
//! - Pure (SBIO): no I/O; host facts (the inventory, git dates) arrive as `LkbFacts`.
//! - `no_std` + `alloc`, `forbid(unsafe_code)`; builds for both wasm32 targets (NSCC).
//! - `metrale_circuit` is imported only by the `lkb*` modules (the schema owners' readers).

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]
#![deny(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

extern crate alloc;

mod prelude;

pub mod check;
pub mod claims;
pub mod config;
pub mod decisions;
pub mod ledger;
pub mod ledger_adapter;
pub mod lkb;
pub mod lkb_circuits;
pub mod lkb_derive;
pub mod lkb_headroom;
pub mod lkb_rules;
pub mod owed;
pub mod pack;
pub mod queries;
pub mod record;
pub mod record_flow;
pub mod research;
pub mod runs;
pub mod sections;
pub mod tooling;
pub mod vocab;

pub use config::{Campaign, Config, Sources};
pub use pack::Metrale;
pub use vocab::{labels, rels};

/// 2026-10-23: Derive the metrale bundle (the core derive over this pack).
pub fn derive(
    cfg: &Config,
    src: &latlas_core::SourceSet,
    lkb: Option<&lkb_derive::LkbFacts>,
) -> latlas_error::Result<latlas_core::Bundle> {
    latlas_core::derive::<Metrale>(cfg, src, lkb)
}
