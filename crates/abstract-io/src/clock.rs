// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The time port: a wall clock, a monotonic clock, and a sleep.
//!
//! Owner: metrale-abstract-io.
//! Invariants: `monotonic_ns` never decreases for one clock; the wall clock may.

/// 2026-10-17: Wall and monotonic time plus an async sleep, supplied by a host.
pub trait Clock {
    /// Wall clock, milliseconds since the Unix epoch.
    fn now_unix_ms(&self) -> i64;
    /// Monotonic nanoseconds from an arbitrary per-clock origin.
    fn monotonic_ns(&self) -> u64;
    async fn sleep_ms(&self, ms: u64);
}
