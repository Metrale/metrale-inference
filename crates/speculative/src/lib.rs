// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Metrale Engine speculative-decoding policy: the speculation controller
//! (`spec_ctl`, every draft-depth decision including plain decode), the adaptive MTP rung on
//! it, dispatch eligibility, the n-gram proposer, draft-capacity clamps, per-run speculation
//! counters and the scheduler snapshot.
//!
//! Owner: speculative.
//! Invariants: none beyond the types.

pub mod adaptive_rung;
pub mod ngram;
pub mod prompt_lookup;
pub mod shared_lookup;
pub mod snapshot;
pub mod spec_capacity;
pub mod spec_cost;
pub mod spec_ctl;
pub mod spec_eligibility;
pub mod spec_stats;
