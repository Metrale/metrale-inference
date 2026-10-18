// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: Deterministic inputs for the tests: a seeded xorshift64* generator and
//! Box-Muller normals, so every run sees the same vectors without a dependency.

#![allow(dead_code)]

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    pub fn normal(&mut self) -> f64 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }

    pub fn normals(&mut self, n: usize, sigma: f64) -> Vec<f32> {
        (0..n).map(|_| (self.normal() * sigma) as f32).collect()
    }
}
