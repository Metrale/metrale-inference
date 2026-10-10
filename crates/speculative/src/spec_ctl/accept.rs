// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The acceptance model: decayed per-position counts of conditional outcomes
//! ("draft j accepted given drafts 1..j were"), blended with a prior, read as conditional rates
//! for [`super::chain`]. One estimator serves every drafter and both observation forms.
//!
//! Owner: speculative.
//! Invariants:
//! - Only observed positions decay; a position with no data takes the deepest informed
//!   position's rate; with no data anywhere there are no rates (`None`).
//! - Pure: counts in, rates out. The host decides when to observe and what is shared.
//!
//! Two observation forms:
//! - step form ([`AcceptCounts::observe_step`]): one verify of `d` drafts that accepted `a`
//!   observes accepts at positions `1..=a` and a reject at `a + 1` when `a < d`; nothing past it.
//! - rate form ([`AcceptCounts::observe_rate`]): a flush's measured rate at one position. With
//!   [`SeedWeight::SteadyState`] the first sample carries weight `1 / (1 - decay)`, which makes
//!   the ratio of decayed sums EXACTLY a seeded EWMA with `alpha = 1 - decay`: the denominator
//!   is `d^(t-1)/alpha + sum_{i<t-1} d^i = 1/alpha` at every `t`, and the numerator
//!   `d^(t-1) x_1 / alpha + sum_{i<t-1} d^i x_{t-i}`, so the rate is
//!   `d^(t-1) x_1 + alpha sum_{i<t-1} d^i x_{t-i}`, the seeded EWMA.

use super::chain::MAX_POSITIONS;

/// 2026-10-10: Weight of a position's first observation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeedWeight {
    /// 2026-10-10: Weight 1: early rates are plain averages of the few outcomes seen.
    Unit,
    /// 2026-10-10: Weight `1 / (1 - decay)`: the rate is a seeded EWMA (module doc).
    SteadyState,
}

/// 2026-10-10: How a model forgets and how much the prior weighs. No defaults: every
/// controller states its own (PCND).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AcceptParams {
    /// 2026-10-10: Per-observation decay in `0..1` (memory about `1 / (1 - decay)`).
    pub decay: f64,
    /// 2026-10-10: Observations the prior is worth where it has data; 0 ignores it.
    pub prior_weight: f64,
    /// 2026-10-10: Observations the cold-start rates (a drafter calibration's position priors,
    /// or a reference engine's measured acceptance) are worth; 0 ignores them.
    pub cold_weight: f64,
    pub seed: SeedWeight,
}

/// 2026-10-10: What a position's rate is pulled towards before (and alongside) its own data,
/// weighted `AcceptParams::cold_weight`.
#[derive(Clone, Debug, PartialEq)]
pub enum ColdPrior {
    None,
    /// 2026-10-10: Conditional rates by position (a drafter calibration's position priors, or
    /// a reference engine's measured acceptance); positions past the end take the last.
    Rates(Vec<f64>),
    /// 2026-10-10: Each position from the second on is pulled towards the rate of the position
    /// before it: a deeper draft is, a priori, accepted like the one before it. With no
    /// reference rates this keeps a rarely verified position from freezing at a few unlucky
    /// observations.
    Chained,
}

/// 2026-10-10: Decayed accept and observation counts per draft position (0-based index =
/// position - 1).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AcceptCounts {
    acc: [f64; MAX_POSITIONS],
    obs: [f64; MAX_POSITIONS],
}

impl AcceptCounts {
    fn fold(&mut self, p: &AcceptParams, i: usize, accepted: f64) {
        let w = match (p.seed, self.obs[i] > 0.0) {
            (SeedWeight::SteadyState, false) => 1.0 / (1.0 - p.decay),
            _ => 1.0,
        };
        self.acc[i] = self.acc[i] * p.decay + accepted * w;
        self.obs[i] = self.obs[i] * p.decay + w;
    }

    /// 2026-10-10: Fold one verify of `drafts` drafts that accepted `accepted` (step form).
    pub fn observe_step(&mut self, p: &AcceptParams, drafts: usize, accepted: usize) {
        let seen = drafts.min(accepted + 1).min(MAX_POSITIONS);
        for i in 0..seen {
            self.fold(p, i, if i < accepted { 1.0 } else { 0.0 });
        }
    }

    /// 2026-10-10: Fold one measured rate at 1-based `position` (rate form). A non-finite
    /// rate or a position outside `1..=MAX_POSITIONS` is ignored.
    pub fn observe_rate(&mut self, p: &AcceptParams, position: usize, rate: f64) {
        if rate.is_finite() && (1..=MAX_POSITIONS).contains(&position) {
            self.fold(p, position - 1, rate.clamp(0.0, 1.0));
        }
    }

    /// 2026-10-10: Scale every count by `f` in `0..=1`: older evidence weighs less against the
    /// next observations (a re-probe after a suspension).
    pub fn soften(&mut self, f: f64) {
        for i in 0..MAX_POSITIONS {
            self.acc[i] *= f;
            self.obs[i] *= f;
        }
    }

    /// 2026-10-10: Whether 1-based `position` has any (decayed) observation.
    pub fn observed(&self, position: usize) -> bool {
        (1..=MAX_POSITIONS).contains(&position) && self.obs[position - 1] > 0.0
    }

    /// 2026-10-10: The current rate at 1-based `position` from these counts alone.
    pub fn rate(&self, position: usize) -> Option<f64> {
        self.observed(position)
            .then(|| self.acc[position - 1] / self.obs[position - 1])
    }

    /// 2026-10-10: Per-position conditional rates of these counts blended with `prior`'s (worth
    /// `p.prior_weight` observations where the prior has data) and the `cold` prior (worth
    /// `p.cold_weight`); a position with none of them takes the deepest informed position's
    /// rate. `None` when no position has data.
    pub fn rates(
        &self,
        p: &AcceptParams,
        prior: Option<&AcceptCounts>,
        cold: &ColdPrior,
    ) -> Option<[f64; MAX_POSITIONS]> {
        let mut out = [0.0; MAX_POSITIONS];
        let mut last: Option<f64> = None;
        let cw = p.cold_weight;
        for (i, o) in out.iter_mut().enumerate() {
            let (pa, po) = match prior {
                Some(q) if q.obs[i] > 0.0 && p.prior_weight > 0.0 => {
                    (p.prior_weight * q.acc[i] / q.obs[i], p.prior_weight)
                }
                _ => (0.0, 0.0),
            };
            let cold_rate = match cold {
                ColdPrior::Rates(c) if !c.is_empty() => Some(super::chain::at(c, i + 1)),
                ColdPrior::Chained => last,
                _ => None,
            };
            let (ca, co) = match cold_rate {
                Some(r) if cw > 0.0 => (cw * r, cw),
                _ => (0.0, 0.0),
            };
            let den = self.obs[i] + po + co;
            let r = if den > 0.0 {
                Some((self.acc[i] + pa + ca) / den)
            } else {
                last
            };
            *o = r?;
            last = r;
        }
        Some(out)
    }
}

#[cfg(test)]
#[path = "accept_tests.rs"]
mod tests;
