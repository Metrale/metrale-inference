// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The time port: a wall clock and a monotonic clock, both read at once.
//!
//! Owner: metrale-abstract-io.
//! Invariants: `monotonic_ns` never decreases for one clock; the wall clock may. Waiting is
//! not here: it is the async [`crate::Timer`] port, so a reader of time needs no timer (ACNB).

use crate::{MonoNs, UnixMs};

/// 2026-10-17: Wall and monotonic time, supplied by a host.
pub trait Clock {
    /// Wall clock, milliseconds since the Unix epoch. Protocol `created` fields and logs only.
    fn now_unix_ms(&self) -> UnixMs;
    /// Monotonic nanoseconds from the host's origin: the scheduler's only time source.
    fn monotonic_ns(&self) -> MonoNs;
}
