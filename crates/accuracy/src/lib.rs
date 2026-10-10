// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Accuracy contracts. Each kernel family has a contract: a reference written in
//! bounded arithmetic, a class (`derived` tolerance or `bit_identical`), seeded inputs and
//! seeded mutations. The tolerance is derived from the family's declared pipeline (formats and
//! accumulation), so a conforming kernel cannot exceed it (no false positive) and every
//! mutation must leave it (no false negative), both proven in the run itself.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - Pure: no GPU, no file I/O, no environment, no clock. Repository texts arrive as strings,
//!   kernel outputs through [`runner::KernelRunner`].
//! - Deterministic: every input is drawn from a keyed SplitMix64 stream ([`inputs`]).
//! - A verdict is never built on an empty or unbounded comparison ([`compare`]).

pub mod bounded;
pub mod case;
pub mod check;
pub mod compare;
pub mod contract;
pub mod elem;
pub mod emulate;
pub mod envelope;
pub mod inputs;
pub mod jobs;
pub mod model_check;
pub mod model_logprobs;
pub mod mutation;
pub mod plan;
pub mod points;
pub mod record;
pub mod refs;
pub mod runner;
