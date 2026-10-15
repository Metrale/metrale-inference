// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: metprof, a profiling harness over nsys sqlite exports.
//! `core` is pure; `adapters` does every read behind the traits in `ports`;
//! `cli` wires the two together.

pub mod adapters;
pub mod cli;
pub mod core;
pub mod ports;
