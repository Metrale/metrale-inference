// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The system wall clock and the process monotonic clock as a [`Clock`].
//!
//! Owner: metrale-io-host-std.
//! Invariants:
//! - `monotonic_ns` counts from one process origin (the first read by any `SystemClock` or
//!   `TokioTimer`), so every clock in the process agrees and none decreases.
//! - A wall clock before the Unix epoch is a host fault and panics.
//! - No tokio here: reading time needs no runtime (waiting lives in `timer.rs`).

use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use metrale_abstract_io::{Clock, MonoNs, UnixMs};

/// 2026-10-17: The instant `MonoNs(0)` names, fixed once per process.
pub(crate) fn origin() -> Instant {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    *ORIGIN.get_or_init(Instant::now)
}

/// 2026-10-17: The host's clocks; every instance reads the same process origin.
#[derive(Clone, Copy, Debug)]
pub struct SystemClock;

impl SystemClock {
    /// 2026-10-17: Fixes the process origin if no clock or timer has yet.
    #[must_use]
    pub fn new() -> Self {
        let _ = origin();
        Self
    }
}

// 2026-10-17: Default only satisfies clippy::new_without_default; it carries no config value.
impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for SystemClock {
    fn now_unix_ms(&self) -> UnixMs {
        let since_epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemClock: wall clock is before the Unix epoch");
        UnixMs(
            i64::try_from(since_epoch.as_millis())
                .expect("SystemClock: wall clock overflows i64 ms"),
        )
    }

    fn monotonic_ns(&self) -> MonoNs {
        // 2026-10-17: Saturates; u64 nanoseconds cover ~584 years of uptime.
        MonoNs(u64::try_from(origin().elapsed().as_nanos()).unwrap_or(u64::MAX))
    }
}

#[cfg(test)]
#[path = "clock_tests.rs"]
mod tests;
