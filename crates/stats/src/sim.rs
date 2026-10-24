// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Test-only simulation helpers: AR(1) series and coverage counting.

use alloc::vec::Vec;

use crate::num::f;
use crate::rng::SplitMix64;

/// 2026-10-22: `n` values of a stationary AR(1) with coefficient `phi`, innovation σ = 1 and
/// mean `mu` (started from the stationary distribution).
pub fn ar1_series(n: usize, phi: f64, mu: f64, r: &mut SplitMix64) -> Vec<f64> {
    let mut x = r.normal() / libm::sqrt(1.0 - phi * phi);
    (0..n)
        .map(|_| {
            let v = mu + x;
            x = phi * x + r.normal();
            v
        })
        .collect()
}

/// 2026-10-22: The share of `trials` for which `covers(trial)` holds.
pub fn coverage(trials: usize, mut covers: impl FnMut(usize) -> bool) -> f64 {
    let hit = (0..trials).filter(|&t| covers(t)).count();
    f(hit) / f(trials)
}

/// 2026-10-22: The coverage band a nominal 95% interval must land in over these simulations.
pub const BAND: (f64, f64) = (0.90, 0.97);
