// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The time values every port and caller shares: a monotonic instant, a duration
//! and a wall-clock instant, each a newtype so a nanosecond count and a millisecond stamp
//! cannot be mixed up (TIDE).
//!
//! Owner: metrale-abstract-io.
//! Invariants:
//! - `MonoNs` counts nanoseconds from one host origin and never decreases for one clock.
//! - Arithmetic on `MonoNs`/`DurNs` saturates; it never wraps and never panics.

/// 2026-10-17: Monotonic nanoseconds from a per-process (per-worker) origin.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MonoNs(pub u64);

/// 2026-10-17: A span of nanoseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DurNs(pub u64);

/// 2026-10-17: Wall-clock milliseconds since the Unix epoch; may go backwards between reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnixMs(pub i64);

const NS_PER_US: u64 = 1_000;
const NS_PER_MS: u64 = 1_000_000;
const NS_PER_MS_F64: f64 = 1_000_000.0;

impl MonoNs {
    /// 2026-10-17: The time from `earlier` to `self`; zero when `earlier` is later.
    #[must_use]
    pub const fn since(self, earlier: MonoNs) -> DurNs {
        DurNs(self.0.saturating_sub(earlier.0))
    }

    /// 2026-10-17: `self` moved forward by `d`, saturating at `u64::MAX`.
    #[must_use]
    pub const fn plus(self, d: DurNs) -> MonoNs {
        MonoNs(self.0.saturating_add(d.0))
    }
}

impl DurNs {
    /// 2026-10-17: `ms` milliseconds, saturating at `u64::MAX` nanoseconds.
    #[must_use]
    pub const fn from_ms(ms: u64) -> DurNs {
        DurNs(ms.saturating_mul(NS_PER_MS))
    }

    /// 2026-10-17: `us` microseconds, saturating at `u64::MAX` nanoseconds.
    #[must_use]
    pub const fn from_us(us: u64) -> DurNs {
        DurNs(us.saturating_mul(NS_PER_US))
    }

    /// 2026-10-17: Milliseconds as `f64`, the arithmetic today's `Duration::as_secs_f64() *
    /// 1000.0` stamps use (above 2^53 ns, about 104 days, the value rounds).
    #[must_use]
    pub const fn as_ms_f64(self) -> f64 {
        self.0 as f64 / NS_PER_MS_F64
    }
}

#[cfg(test)]
#[path = "time_tests.rs"]
mod tests;
