// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The port traits the engine's pure code depends on: [`Clock`], [`Timer`],
//! [`Entropy`], [`Env`], the power port ([`power::PowerMeter`], 2026-10-22), the shared [`time`](MonoNs) values, and the [`Io`] bundle that names
//! one host's choice of each. Hosts live in `metrale-io-host-mem` (deterministic, for tests)
//! and `metrale-io-host-std` (the process); a binary is the only place a host is chosen.
//!
//! Owner: metrale-abstract-io.
//! Invariants:
//! - One dependency, `metrale-error`, and no I/O; `no_std` with `alloc`.
//! - A read that answers at once is a plain `fn`; a wait is an `async fn` with no `Send`
//!   bound (ACNB).
//! - Every fallible method returns [`Result`]: a registry code from `metrale-error`, matched on
//!   `err.code()`, never on text (CERF, CLER).
#![no_std]
#![forbid(unsafe_code)]
#![deny(warnings)]
#![deny(clippy::all)]
#![deny(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
#![expect(
    async_fn_in_trait,
    reason = "2026-10-17: single-threaded callers; no Send bound by design"
)]

extern crate alloc;

mod clock;
mod entropy;
mod env;
mod io;
pub mod power;
mod power_caps;
mod power_record;
mod time;
mod timer;

pub use clock::Clock;
pub use entropy::Entropy;
pub use env::Env;
pub use io::Io;
pub use time::{DurNs, MonoNs, UnixMs};
pub use timer::Timer;

/// 2026-10-17: The result every fallible port method returns.
pub type Result<T> = core::result::Result<T, metrale_error::Error>;
