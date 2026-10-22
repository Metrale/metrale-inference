// SPDX-License-Identifier: MIT OR Apache-2.0

//! The logistic domain in integers: `squash` maps a stretched value in `[-2047, 2047]` (units of
//! 1/256 nat) to a 12-bit probability, `stretch` is its inverse. Both are pure integer functions
//! so every target produces identical bits (determinism is a format requirement).

use crate::cast::ixi;
use alloc::vec::Vec;

/// 33 samples of `4096 / (1 + e^(-x))` at `x = -8..=8` in half-nat steps, interpolated
/// linearly between them (the lpaq construction).
const SQUASH_KNOTS: [i32; 33] = [
    1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101, 1546, 2047, 2549, 2994, 3348,
    3607, 3785, 3901, 3975, 4024, 4050, 4068, 4079, 4085, 4089, 4092, 4093, 4094,
];

/// `4096 / (1 + e^(-d/256))`, clamped to `[1, 4095]`.
#[inline]
pub fn squash(d: i32) -> i32 {
    if d > 2047 {
        return 4095;
    }
    if d < -2047 {
        return 1;
    }
    let w = d & 127;
    let i = ixi(((d >> 7) + 16).clamp(0, 31));
    let lo = SQUASH_KNOTS[i];
    let hi = SQUASH_KNOTS[i + 1];
    (lo * (128 - w) + hi * w + 64) >> 7
}

/// The inverse of [`squash`] over the 12-bit probability range, as a table.
pub struct Stretch {
    table: Vec<i16>,
}

impl Stretch {
    /// Build the table by inverting `squash` (4096 entries).
    pub fn new() -> Self {
        let mut table = alloc::vec![0i16; 4096];
        let mut pi = 0usize;
        for x in -2047i32..=2047 {
            let v = ixi(squash(x));
            let mut j = pi;
            while j <= v && j < 4096 {
                table[j] = i16::try_from(x).unwrap_or(0);
                j += 1;
            }
            pi = v + 1;
        }
        let mut j = pi;
        while j < 4096 {
            table[j] = 2047;
            j += 1;
        }
        Self { table }
    }

    /// `ln(p / (1 - p))` in 1/256 nat units, for a 12-bit `p`.
    #[inline]
    pub fn stretch(&self, p: i32) -> i32 {
        i32::from(self.table[ixi(p & 4095)])
    }
}

impl Default for Stretch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squash_is_monotone_and_bounded() {
        let mut prev = 0;
        for d in -3000..=3000 {
            let p = squash(d);
            assert!((1..=4095).contains(&p), "squash({d}) = {p}");
            assert!(p >= prev, "squash not monotone at {d}");
            prev = p;
        }
        assert_eq!(squash(0), 2047);
    }

    #[test]
    fn stretch_inverts_squash_within_one_step() {
        let s = Stretch::new();
        for d in (-2047..=2047).step_by(13) {
            let back = s.stretch(squash(d));
            assert!((back - d).abs() <= 140, "stretch(squash({d})) = {back}");
        }
        assert!(s.stretch(1) < -1800);
        assert!(s.stretch(4095) > 1800);
        assert!(s.stretch(2048).abs() <= 2);
    }
}
