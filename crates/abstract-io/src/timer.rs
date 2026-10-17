// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The waiting port: resolve once a monotonic deadline has passed.
//!
//! Owner: metrale-abstract-io.
//! Invariants: `sleep_until` never resolves before the paired clock reads `>= at`; a deadline
//! already in the past resolves without waiting. The future carries no `Send` bound: the core
//! awaits it and the host's executor drives it (ACNB).

use crate::MonoNs;

/// 2026-10-17: An async wait on the host's monotonic clock, supplied by a host.
pub trait Timer {
    /// Resolves once the host clock's `monotonic_ns() >= at`.
    async fn sleep_until(&self, at: MonoNs);
}
