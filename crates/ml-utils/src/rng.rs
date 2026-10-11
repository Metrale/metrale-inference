// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The deterministic value streams synthetic weights are drawn from.
//!
//! A stream is keyed by `sha256(seed || name || dtype || shape)` and element `i` is a pure
//! function of `(key, i)` (SplitMix64 over a Weyl counter), so a tensor's bytes do not depend on
//! which other tensors exist, on thread count or on chunking.
//!
//! Owner: metrale-ml-utils.
//! Invariants:
//! - Only integer arithmetic and IEEE `+ - * /` on exactly representable operands: no libm
//!   transcendental, so every platform produces the same bits. Normal-like values are
//!   Irwin-Hall sums of 16-bit uniforms, scaled to unit variance.
//! - The SplitMix64 constants are the published ones; changing any of them, or the key
//!   derivation, changes every mock and is a spec schema change.

use sha2::{Digest, Sha256};

/// 2026-10-03: The Weyl increment of SplitMix64.
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// 2026-10-03: SplitMix64's output mix of `z`.
#[inline(always)]
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// 2026-10-03: One keyed stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stream {
    key: u64,
}

impl Stream {
    /// 2026-10-03: The stream of the named tensor under `seed`.
    pub fn for_tensor(seed: u64, name: &str, dtype: &str, shape: &[u64]) -> Self {
        let mut h = Sha256::new();
        h.update(seed.to_le_bytes());
        h.update([0]);
        h.update(name.as_bytes());
        h.update([0]);
        h.update(dtype.as_bytes());
        for d in shape {
            h.update(d.to_le_bytes());
        }
        let d = h.finalize();
        let mut k = [0u8; 8];
        k.copy_from_slice(&d[..8]);
        Stream {
            key: u64::from_le_bytes(k),
        }
    }

    /// 2026-10-03: A sub-stream for a purpose other than the tensor's values (the routing fit's
    /// noise), so the two never share draws.
    pub fn derive(self, purpose: u64) -> Self {
        Stream {
            key: mix(self.key ^ purpose.wrapping_mul(GOLDEN)),
        }
    }

    /// 2026-10-03: Raw 64 bits for element `i`.
    #[inline(always)]
    pub fn bits(self, i: u64) -> u64 {
        mix(self
            .key
            .wrapping_add(i.wrapping_add(1).wrapping_mul(GOLDEN)))
    }

    /// 2026-10-03: A unit-variance, zero-mean value for element `i`: the sum of four 16-bit
    /// uniforms (Irwin-Hall n = 4), bounded by +-sqrt(12).
    #[inline(always)]
    pub fn normal4(self, i: u64) -> f32 {
        irwin_hall4(self.bits(i))
    }

    /// 2026-10-03: A unit-variance, zero-mean value closer to Gaussian in the tails: twelve
    /// 16-bit uniforms from three draws (Irwin-Hall n = 12). Element `i` uses draws `3i..3i+2`.
    pub fn normal12(self, i: u64) -> f32 {
        let base = i.wrapping_mul(3);
        let s = sum16(self.bits(base)) + sum16(self.bits(base + 1)) + sum16(self.bits(base + 2));
        // 2026-10-03: Each 16-bit uniform is (u + 0.5) / 65536, mean 0.5, variance 1/12; twelve
        // of them have mean 6 and variance 1. `s` <= 12 * 65535 is exact in f32.
        (s as f32 + 6.0) / 65536.0 - 6.0
    }

    /// 2026-10-03: A uniform index in `0..n` for element `i` (n > 0).
    pub fn below(self, i: u64, n: u64) -> u64 {
        ((self.bits(i) as u128 * n as u128) >> 64) as u64
    }
}

#[inline(always)]
fn sum16(b: u64) -> u32 {
    (b & 0xFFFF) as u32
        + ((b >> 16) & 0xFFFF) as u32
        + ((b >> 32) & 0xFFFF) as u32
        + (b >> 48) as u32
}

/// 2026-10-03: `sqrt(3)`, the scale that gives four uniforms unit variance (the sum's variance
/// is 4/12). The nearest f32, written out so no runtime square root is involved.
const SQRT3: f32 = 1.732_050_8;

#[inline(always)]
fn irwin_hall4(b: u64) -> f32 {
    let s = sum16(b);
    ((s as f32 + 2.0) / 65536.0 - 2.0) * SQRT3
}

#[cfg(test)]
#[path = "rng_tests.rs"]
mod rng_tests;
