// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Online calibration of the step-cost model: one multiplicative scale per
//! power-of-two batch-width bucket for wall time and one for energy, each moved `alpha` of the
//! way towards `measured / source` by every measured step. Context growth, clocks and thermal
//! state move every depth of a width alike, so one scale per width is what a few samples can
//! fit; the source keeps the shape across depths.
//!
//! Owner: speculative.
//! Invariants:
//! - `alpha` 0 is "off": every scale stays exactly 1.0, so calibrated costs equal the source's
//!   bit for bit.
//! - A non-finite or non-positive measurement is ignored; it never moves a scale.

use super::cost::StepCost;

/// 2026-10-10: Width buckets: 1, 2, 4, .. 256 and wider.
pub const BUCKETS: usize = 9;

/// 2026-10-10: The bucket of a batch of `n` sequences.
pub fn bucket(n: usize) -> usize {
    (n.max(1).next_power_of_two().trailing_zeros() as usize).min(BUCKETS - 1)
}

/// 2026-10-10: The scales and their EWMA weight.
#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
    alpha: f64,
    ms: [f64; BUCKETS],
    j: [f64; BUCKETS],
    samples: [u64; BUCKETS],
}

impl Calibration {
    /// 2026-10-10: Scales at 1 with weight `alpha` in `0..=1` (0 = off). Clamped into range.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: if alpha.is_finite() {
                alpha.clamp(0.0, 1.0)
            } else {
                0.0
            },
            ms: [1.0; BUCKETS],
            j: [1.0; BUCKETS],
            samples: [0; BUCKETS],
        }
    }

    /// 2026-10-10: `base` scaled by the bucket of `n`.
    pub fn apply(&self, n: usize, base: StepCost) -> StepCost {
        let b = bucket(n);
        StepCost {
            ms: base.ms * self.ms[b],
            j: base.j.map(|j| j * self.j[b]),
        }
    }

    /// 2026-10-10: Fold one measured step whose source cost was `base`.
    pub fn observe(&mut self, n: usize, base: StepCost, ms: f64, j: Option<f64>) {
        let b = bucket(n);
        if ms.is_finite() && ms > 0.0 && base.ms > 0.0 {
            self.ms[b] += self.alpha * (ms / base.ms - self.ms[b]);
            self.samples[b] += 1;
        }
        if let (Some(m), Some(s)) = (j, base.j)
            && m.is_finite()
            && m > 0.0
            && s > 0.0
        {
            self.j[b] += self.alpha * (m / s - self.j[b]);
        }
    }

    /// 2026-10-10: The wall-time scale of the bucket of `n` (1 until measured).
    pub fn ms_scale(&self, n: usize) -> f64 {
        self.ms[bucket(n)]
    }

    /// 2026-10-10: Measured steps folded into the bucket of `n`.
    pub fn samples(&self, n: usize) -> u64 {
        self.samples[bucket(n)]
    }
}
