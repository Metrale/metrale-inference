// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The port traits the engine's pure code depends on: [`Clock`],
//! [`Entropy`], [`Env`], and the [`Io`] bundle that names one host's choice of
//! each. Hosts live in `metrale-io-host-mem` (deterministic, for tests) and
//! `metrale-io-host-std` (the process); a binary is the only place a host is
//! chosen.
//!
//! Owner: metrale-abstract-io.
//! Invariants:
//! - No dependencies and no I/O; `no_std` with `alloc`.
//! - `async fn` in traits carries no `Send` bound.
//! - Errors are closed enums; nothing is swallowed.
#![no_std]
#![forbid(unsafe_code)]
#![deny(warnings)]
#![deny(clippy::all)]
#![expect(
    async_fn_in_trait,
    reason = "2026-10-17: single-threaded callers; no Send bound by design"
)]

extern crate alloc;

mod clock;
mod entropy;
mod env;
mod io;

pub use clock::Clock;
pub use entropy::{Entropy, EntropyError};
pub use env::{Env, EnvError};
pub use io::Io;
