// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: A tokio sleep to an absolute [`MonoNs`] deadline as a [`Timer`].
//!
//! Owner: metrale-io-host-std.
//! Invariants:
//! - A deadline is read against the same process origin as `SystemClock::monotonic_ns`, so a
//!   sleep to `clock.monotonic_ns().plus(d)` ends no earlier than that clock reading.
//! - The future must be polled on a tokio runtime built with `enable_time` (the scheduler's
//!   current-thread runtime is); polling it elsewhere panics inside tokio.

use std::time::Duration;

use metrale_abstract_io::{MonoNs, Timer};

use crate::clock::origin;

/// 2026-10-17: Waits on tokio's time driver; the host's runtime drives it.
#[derive(Clone, Copy, Debug)]
pub struct TokioTimer;

impl Timer for TokioTimer {
    async fn sleep_until(&self, at: MonoNs) {
        let deadline = origin() + Duration::from_nanos(at.0);
        tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await;
    }
}

#[cfg(test)]
#[path = "timer_tests.rs"]
mod tests;
