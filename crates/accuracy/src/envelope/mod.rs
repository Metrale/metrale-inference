// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The kernel envelope sweep (step 2 of the zero-tuning pathway): for every
//! (projection class, N x K, row count) the described models need, time every candidate entry
//! point the contracts cover, keep only candidates that pass their own accuracy contract, pick
//! the fastest, classify it against today's default (byte-identical, or numerics-changing), and
//! write the winners to `kernels/<hw>/common/SCHEDULES.toml`, which the kernel build bakes in.
//!
//! Owner: metrale-accuracy (envelope).
//! Invariants:
//! - Pure, as the rest of the crate: measurements arrive as records, repository texts as strings.
//! - Bit-identical by default: a winner whose output differs from the default's in any byte on
//!   any input class is recorded as `numerics = "differs"` and is enabled only by an explicit
//!   opt-in; it never becomes a default here.
//! - A candidate that failed or could not run its accuracy contract never wins.

pub mod fusions;
pub mod grid;
pub mod record;
pub mod schedules;
pub mod select;
pub mod sources;
