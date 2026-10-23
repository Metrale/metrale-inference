// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The kernel knowledge graph's derivation (DESIGN-kernel-knowledge-graph.md §2.1):
//! the SSOT files (LEDGER, TOOLING, CHARTER, designs, research maps, principles, memory, and the
//! LKB) in, a deterministic `kg-bundle/v1` out, and the closures, counts and owed rows the MCP
//! serves, computed as pure functions over that bundle. The graph never holds a fact that is not
//! in a file.
//!
//! Owner: metrale-kg-derive.
//! Invariants:
//! - Pure (SBIO): no I/O, no environment, no clock. Bytes arrive as a [`sources::SourceSet`] and
//!   settings texts; hosts live in metrale-io-host-kg, the binary in metrale-kg-mcp.
//! - `no_std` + `alloc`, `forbid(unsafe_code)`; builds for both wasm32 targets (NSCC).
//! - Every fallible function returns the registry error (CERF, block `0x0800`).

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

pub mod bundle;
pub mod claims;
pub mod config;
pub mod decisions;
pub mod derive;
pub mod hash;
pub mod id;
pub mod jsonl;
pub mod ledger;
pub mod ledger_adapter;
pub mod lkb;
pub mod lkb_circuits;
pub mod lkb_derive;
pub mod lkb_headroom;
pub mod lkb_rules;
pub mod memory;
pub mod owed;
pub mod principles;
pub mod queries;
pub mod research;
pub mod secrets;
pub mod settings;
pub mod sources;
pub mod status;
pub mod text;
pub mod tooling;
pub mod tools;

pub use bundle::{Bundle, Edge, Node, Text};
pub use config::Config;
pub use derive::derive;
pub use id::{KgId, Label, Rel};
pub use sources::SourceSet;
