// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: What makes one `met serve` "the same server" as another, as pure data and pure
//! functions: the argv, binary and `METRALE_*` lever digests (`identity`, `env`), the
//! leased-server reuse decision (`lease`), and the `GET /forward` and `GET /memory` wire
//! shapes (`wire`).
//!
//! The server publishes these and the benchmark harness checks them, so both depend on this
//! crate rather than the server on the harness. Reading the process (its arguments, its
//! executable, its environment) is the host's job: `metrale-host-probe` does it and hands the
//! values in.
//!
//! Owner: serve identity.
//! Invariants:
//! - No I/O and no `std`: everything here is a function of its arguments.
//! - Every digest is 64 lower-case hex characters.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]
#![deny(warnings)]
#![deny(clippy::all)]
#![deny(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

extern crate alloc;

pub mod env;
mod hex;
pub mod identity;
pub mod lease;
pub mod wire;

pub use hex::hex_lower;
