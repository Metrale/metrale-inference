// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The `METRALE_*` lever declarations as typed settings groups.
//!
//! Each `groups/<g>.rs` declares one group with `lever_group!`, which emits the published
//! [`spec::LeverSpec`] rows, the parse [`spec::Rule`] of each row, and (for groups a process
//! reads) a typed struct with a `resolve` constructor. [`resolve()`] builds the runtime settings
//! from a CLI layer, a recipe layer and an [`Env`], CLI first, and records where every value
//! came from in a [`provenance::Provenance`].
//!
//! Owner: settings.
//! Invariants: every lever is declared exactly once, in exactly one group file; the resolver
//! reads the environment only through [`Env`].

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]
#![deny(warnings)]
#![deny(clippy::all)]

extern crate alloc;

#[macro_use]
mod macros;

pub mod groups;
pub mod provenance;
pub mod resolve;
pub mod sources;
pub mod spec;

pub use metrale_abstract_io::{Env, EnvError};
pub use resolve::{InferenceServerSettings, SettingsError, resolve};

/// 2026-10-17: Paths the exported `lever_group!` helpers name through `$crate`; re-exports the
/// item, never the `alloc` crate (that would trip `pub_use_of_private_extern_crate`).
#[doc(hidden)]
pub mod __rt {
    pub use alloc::string::String;
}

#[cfg(test)]
#[path = "groups_tests.rs"]
mod groups_tests;

#[cfg(test)]
#[path = "resolve_tests.rs"]
mod resolve_tests;
