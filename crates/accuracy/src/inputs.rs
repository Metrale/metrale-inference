// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Seeded input generators: a self-contained SplitMix64 stream (no dependency whose
//! version bump could change a corpus), the input classes every contract draws from, and the
//! structural index sets a sampled comparison must always include.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - A tensor's stream is seeded from (contract seed, family, point key, tensor name, class) by
//!   SHA-256, so adding a tensor or a point never perturbs another's data.
//! - Each class is defined here once, with its constants and the reason for them; a contract
//!   names classes, it does not tune them (a tuned generator could hide a mutation).

use sha2::{Digest, Sha256};

use crate::elem::Elem;

/// 2026-10-09: SplitMix64 (Steele, Lea, Flood 2014): 64-bit state, full period, stable forever.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// 2026-10-09: A stream from a raw seed.
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    /// 2026-10-09: A stream keyed by names: the first 8 bytes of SHA-256 over the parts, each
    /// length-prefixed so `("ab","c")` and `("a","bc")` differ.
    pub fn keyed(seed: u64, parts: &[&str]) -> Self {
        let mut h = Sha256::new();
        h.update(b"metrale-accuracy-gen\0");
        h.update(seed.to_le_bytes());
        for p in parts {
            h.update((p.len() as u64).to_le_bytes());
            h.update(p.as_bytes());
        }
        let d = h.finalize();
        let mut b = [0u8; 8];
        b.copy_from_slice(&d[..8]);
        SplitMix64::new(u64::from_le_bytes(b))
    }

    /// 2026-10-09: Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// 2026-10-09: Uniform in [0, 1) with 53 random bits.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// 2026-10-09: Uniform integer in [0, n), `n > 0` (rejection keeps it unbiased).
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "below(0) has no value");
        let zone = u64::MAX - u64::MAX % n;
        loop {
            let x = self.next_u64();
            if x < zone {
                return x % n;
            }
        }
    }

    /// 2026-10-09: Standard normal (Box-Muller; one value per call, the pair's second dropped
    /// so the stream position does not depend on call parity).
    pub fn gaussian(&mut self) -> f64 {
        let u1 = 1.0 - self.unit();
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// 2026-10-09: The input classes. Every contract runs `gaussian` and the adversarial classes it
/// lists; the class names are the contract spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum InputClass {
    /// 2026-10-09: N(0, 1) scaled by the tensor's stated scale.
    Gaussian,
    /// 2026-10-09: Gaussian with one element in 64 multiplied by 64: the activation outliers of
    /// real residual streams, which stress amax-based scales and the reduction's dynamic range.
    Outliers,
    /// 2026-10-09: Magnitudes in [0.75, 1] of the operand format's largest finite value (for a
    /// 16/32-bit format, of 2^15, the largest residual-stream values seen in served models):
    /// saturation and scale-overflow paths.
    NearOverflow,
    /// 2026-10-09: Magnitudes in the operand format's subnormal range: flush-to-zero and
    /// underflow paths.
    Denormal,
    /// 2026-10-09: Every element the same value (0.75): degenerate maxima, ties in top-k and
    /// argmax, exact amax scales.
    AllEqual,
    /// 2026-10-09: Gaussian with every fourth row zero: a zero amax, a zero norm, an empty
    /// softmax contribution.
    ZeroRows,
}

const CLASSES: [(InputClass, &str); 6] = [
    (InputClass::Gaussian, "gaussian"),
    (InputClass::Outliers, "outliers"),
    (InputClass::NearOverflow, "near_overflow"),
    (InputClass::Denormal, "denormal"),
    (InputClass::AllEqual, "all_equal"),
    (InputClass::ZeroRows, "zero_rows"),
];

impl InputClass {
    /// 2026-10-09: The contract spelling.
    pub fn name(self) -> &'static str {
        CLASSES
            .iter()
            .find(|(c, _)| *c == self)
            .map_or("?", |(_, n)| n)
    }

    /// 2026-10-09: Parse the contract spelling.
    pub fn parse(s: &str) -> Option<Self> {
        CLASSES.iter().find(|(_, n)| *n == s).map(|(c, _)| *c)
    }

    /// 2026-10-09: Every class, in contract order.
    pub fn all() -> impl Iterator<Item = InputClass> {
        CLASSES.iter().map(|(c, _)| *c)
    }
}

/// 2026-10-09: A `rows x cols` tensor of class `class`, values rounded (saturating) into
/// `fmt`, with `scale` the standard deviation of the gaussian part. Row-major.
pub fn tensor(
    rng: &mut SplitMix64,
    class: InputClass,
    rows: usize,
    cols: usize,
    scale: f64,
    fmt: Elem,
) -> Vec<f64> {
    let top = if fmt.precision >= 8 {
        32768.0
    } else {
        fmt.max_finite
    };
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for _ in 0..cols {
            let g = rng.gaussian() * scale;
            let x = match class {
                InputClass::Gaussian => g,
                InputClass::Outliers => {
                    if rng.below(64) == 0 {
                        g * 64.0
                    } else {
                        g
                    }
                }
                InputClass::NearOverflow => {
                    let m = top * (0.75 + 0.25 * rng.unit());
                    if rng.below(2) == 0 { m } else { -m }
                }
                InputClass::Denormal => {
                    let steps = 1 + rng.below((1u64 << (fmt.precision - 1)) - 1);
                    let m = steps as f64 * fmt.min_subnormal();
                    if fmt.signed && rng.below(2) == 0 {
                        -m
                    } else {
                        m
                    }
                }
                InputClass::AllEqual => 0.75,
                InputClass::ZeroRows => {
                    if r % 4 == 3 {
                        0.0
                    } else {
                        g
                    }
                }
            };
            let x = if fmt.signed { x } else { x.abs() };
            out.push(fmt.round_saturating(x).unwrap_or(0.0));
        }
    }
    out
}

/// 2026-10-09: Indices in `[0, n)` a sampled comparison must cover: both ends, both sides of
/// every multiple of each `stride` (tile, group, split and alignment edges), and `extra` seeded
/// random ones. Sorted, distinct.
pub fn structural_indices(
    n: usize,
    strides: &[usize],
    extra: usize,
    rng: &mut SplitMix64,
) -> Vec<usize> {
    let mut v = vec![0, n.saturating_sub(1)];
    for &s in strides.iter().filter(|&&s| s > 0) {
        let mut m = s;
        while m < n {
            v.push(m - 1);
            v.push(m);
            m += s;
        }
    }
    for _ in 0..extra {
        v.push(rng.below(n as u64) as usize);
    }
    v.retain(|&i| i < n);
    v.sort_unstable();
    v.dedup();
    v
}

#[cfg(test)]
#[path = "inputs_tests.rs"]
mod tests;
