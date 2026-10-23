// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: SplitMix64 (Steele, Lea & Flood 2014): the seeded generator behind every
//! resample and randomized order. Integer-only state, so a seed gives the same stream on every
//! target; the seed is stored beside each interval it produced.
//!
//! Owner: metrale-stats.
//! Invariants: none beyond determinism.

/// 2026-10-22: A SplitMix64 stream.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// 2026-10-22: Uniform on [0, 1), 53 bits.
    pub fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / 9_007_199_254_740_992.0;
        // 2026-10-22: < 2^53, so the conversion is exact.
        (self.next_u64() >> 11) as f64 * SCALE
    }

    /// 2026-10-22: Uniform on `0..n` (Lemire's multiply-high; `n = 0` gives 0).
    pub fn below(&mut self, n: usize) -> usize {
        let n64 = u64::try_from(n).unwrap_or(u64::MAX);
        let hi = (u128::from(self.next_u64()) * u128::from(n64)) >> 64;
        usize::try_from(hi).unwrap_or(0)
    }

    /// 2026-10-22: Standard normal (Box-Muller, one of the pair).
    pub fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.next_f64();
        let u2 = self.next_f64();
        libm::sqrt(-2.0 * libm::log(u1)) * libm::cos(2.0 * core::f64::consts::PI * u2)
    }
}
