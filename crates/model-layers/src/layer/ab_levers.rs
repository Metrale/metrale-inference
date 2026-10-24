// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The paired in-serve A/B instrument's per-pass lever set (`METRALE_PAIRED_AB`):
//! which of the instrument's named levers this forward runs with. Bit `i` is the `i`-th name of
//! the setting's list; a model module that owns a name maps it to its own kernel choice when it
//! loads (the engine never names a lever), so the type is family-neutral.
//!
//! Owner: model-layers.
//! Invariants: [`AbLevers::NONE`] is every pass outside the instrument, so a model that reads it
//! takes its resolved kernels unchanged.

/// 2026-10-23: A set of at most 32 instrument levers, by list position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AbLevers(u32);

impl AbLevers {
    /// 2026-10-23: No lever on: the A arm and every pass outside the instrument.
    pub const NONE: Self = Self(0);
    /// 2026-10-23: The most levers one instrument run names.
    pub const MAX: usize = 32;

    /// 2026-10-23: The set holding only list position `i` (`i < MAX`).
    pub fn only(i: usize) -> Self {
        assert!(i < Self::MAX, "AbLevers: position {i} of {}", Self::MAX);
        Self(1u32 << i)
    }

    /// 2026-10-23: The first `n` positions.
    pub fn first(n: usize) -> Self {
        assert!(n <= Self::MAX, "AbLevers: {n} levers of {}", Self::MAX);
        Self(if n == Self::MAX {
            u32::MAX
        } else {
            (1u32 << n) - 1
        })
    }

    /// 2026-10-23: Whether position `i` is on.
    pub fn has(self, i: usize) -> bool {
        i < Self::MAX && self.0 & (1u32 << i) != 0
    }

    /// 2026-10-23: The set without position `i`.
    pub fn without(self, i: usize) -> Self {
        Self(self.0 & !Self::only(i).0)
    }

    /// 2026-10-23: The raw bits (a graph key's field).
    pub fn bits(self) -> u32 {
        self.0
    }
}
