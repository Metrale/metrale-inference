// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The system wall clock, a monotonic clock from `Instant`, and a
//! tokio sleep, as a [`Clock`].
//!
//! Owner: metrale-io-host-std.
//! Invariants:
//! - `monotonic_ns` counts from the instant the clock was created and never decreases.
//! - A wall clock before the Unix epoch is a host fault and panics.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use metrale_abstract_io::Clock;

/// 2026-10-17: The host's clocks; `origin` anchors `monotonic_ns`.
#[derive(Clone, Copy, Debug)]
pub struct SystemClock {
    origin: Instant,
}

impl SystemClock {
    #[must_use]
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

// 2026-10-17: Default only satisfies clippy::new_without_default; it carries no config value.
impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for SystemClock {
    fn now_unix_ms(&self) -> i64 {
        let since_epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemClock: wall clock is before the Unix epoch");
        i64::try_from(since_epoch.as_millis()).expect("SystemClock: wall clock overflows i64 ms")
    }

    fn monotonic_ns(&self) -> u64 {
        // 2026-10-17: Saturates; u64 nanoseconds cover ~584 years of uptime.
        u64::try_from(self.origin.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    async fn sleep_ms(&self, ms: u64) {
        tokio::time::sleep(Duration::from_millis(ms)).await;
    }
}

#[cfg(test)]
#[path = "clock_tests.rs"]
mod tests;
