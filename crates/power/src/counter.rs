// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Raw cumulative counter readings to non-negative deltas: a backwards step is a wrap
//! or a reset, told apart here. Moved from `metrale-telemetry::energy` (2026-09-26) so the
//! serve telemetry, the power meters and the attribution core share one tracker (SSOT);
//! telemetry re-exports it.
//!
//! Owner: metrale-power.
//! Invariants: a delta is never negative; the first reading contributes nothing.

/// 2026-09-26: What one counter reading did to the running total.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Advance {
    /// 2026-09-26: The first reading: it anchors the series and contributes nothing.
    First,
    /// 2026-09-26: The counter moved forward by this many mJ.
    Delta(u64),
    /// 2026-09-26: The counter wrapped past `u64::MAX`; this many mJ elapsed.
    Wrapped(u64),
    /// 2026-09-26: The counter restarted; the new reading, the mJ since the
    /// restart, is charged.
    Reset(u64),
}

impl Advance {
    pub fn millijoules(self) -> u64 {
        match self {
            Self::First => 0,
            Self::Delta(d) | Self::Wrapped(d) | Self::Reset(d) => d,
        }
    }
}

/// 2026-09-26: Turns raw counter readings into non-negative deltas.
#[derive(Clone, Copy, Debug)]
pub struct CounterTracker {
    prev: Option<u64>,
    /// 2026-10-22: The counter's largest value (`2^width - 1`).
    max: u64,
}

/// 2026-10-22: A 64-bit counter (NVML, amdsmi).
impl Default for CounterTracker {
    fn default() -> Self {
        Self::with_width(64)
    }
}

/// 2026-09-26: A backwards step is a wrap only when the previous reading sat in the top quarter
/// of the range and the new one in the bottom quarter; any other backwards step is a restart.
impl CounterTracker {
    /// 2026-10-22: A tracker for a counter that wraps after `width_bits` bits (clamped to 1..=64).
    #[must_use]
    pub fn with_width(width_bits: u32) -> Self {
        let bits = width_bits.clamp(1, 64);
        let max = if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        Self { prev: None, max }
    }

    pub fn advance(&mut self, raw: u64) -> Advance {
        let quarter = self.max / 4;
        let prev = self.prev.replace(raw);
        match prev {
            None => Advance::First,
            Some(p) if raw >= p => Advance::Delta(raw - p),
            Some(p) if p > self.max - quarter && raw < quarter => {
                Advance::Wrapped(raw.wrapping_sub(p) & self.max)
            }
            Some(_) => Advance::Reset(raw),
        }
    }
}

#[cfg(test)]
#[path = "counter_tests.rs"]
mod tests;
