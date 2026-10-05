// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Fitting and writing an [`AcceptanceCalibration`] from a run's counts (`met
//! benchmark spec-cost-table`). Pure.
//!
//! - P(accept | bucket) = accepted / reached in the bucket. A bucket with fewer than
//!   [`MIN_OUTCOMES`] reached drafts is pooled with its higher-confidence neighbours until the
//!   pool has enough, scanning from the most confident bucket down; a short remainder at the
//!   low end joins the last pool. Every bucket of a pool takes the pool's rate.
//! - prior(j) = P(draft j accepted | drafts 1..j-1 accepted), from verify steps by
//!   (drafts, accepted): steps with drafts >= j and accepted >= j over those with drafts >= j
//!   and accepted >= j - 1. Positions are fitted in order while that denominator has at least
//!   [`MIN_OUTCOMES`] steps.
//!
//! Owner: speculative.
//! Invariants: a fit is refused, not thinned, when the run has fewer than [`MIN_OUTCOMES`]
//! reached drafts or first-position steps; the written text parses back to the fit.

use std::collections::BTreeMap;

use super::{AcceptanceCalibration, DrafterKey};

/// 2026-10-04: The fewest observations behind one fitted probability: a binomial standard
/// error of at most 0.05 at any rate.
pub const MIN_OUTCOMES: u64 = 100;

impl AcceptanceCalibration {
    /// 2026-10-04: The calibration of `drafter` from per-bucket reached-draft outcomes
    /// (`edges` ascending, `accepted[i]` and `rejected[i]` in bucket `i`) and verify steps by
    /// `(drafts, accepted)`.
    pub fn fit(
        drafter: DrafterKey,
        edges: &[f32],
        accepted: &[u64],
        rejected: &[u64],
        steps: &BTreeMap<(usize, usize), u64>,
    ) -> Result<Self, String> {
        if edges.is_empty() || edges.len() != accepted.len() || edges.len() != rejected.len() {
            return Err("acceptance fit: edges and counts differ in length".into());
        }
        let mut p_accept = vec![0.0; edges.len()];
        let mut pools: Vec<(Vec<usize>, u64, u64)> = Vec::new();
        let mut open: (Vec<usize>, u64, u64) = (Vec::new(), 0, 0);
        for i in (0..edges.len()).rev() {
            open.0.push(i);
            open.1 += accepted[i];
            open.2 += accepted[i] + rejected[i];
            if open.2 >= MIN_OUTCOMES {
                pools.push(std::mem::take(&mut open));
            }
        }
        if !open.0.is_empty() {
            let Some(last) = pools.last_mut() else {
                return Err(format!(
                    "acceptance fit: {} reached drafts, fewer than {MIN_OUTCOMES}",
                    open.2
                ));
            };
            last.0.extend(open.0);
            last.1 += open.1;
            last.2 += open.2;
        }
        for (members, acc, total) in &pools {
            for &i in members {
                p_accept[i] = *acc as f64 / *total as f64;
            }
        }
        let max_drafts = steps.keys().map(|&(d, _)| d).max().unwrap_or(0);
        let mut prior_by_position = Vec::new();
        for j in 1..=max_drafts {
            let sum = |min_a: usize| -> u64 {
                steps
                    .iter()
                    .filter(|&(&(d, a), _)| d >= j && a >= min_a)
                    .map(|(_, &n)| n)
                    .sum()
            };
            let reached = sum(j - 1);
            if reached < MIN_OUTCOMES {
                break;
            }
            prior_by_position.push(sum(j) as f64 / reached as f64);
        }
        if prior_by_position.is_empty() {
            return Err(format!(
                "acceptance fit: fewer than {MIN_OUTCOMES} verify steps reached the first draft"
            ));
        }
        Ok(Self {
            drafter,
            edges: edges.to_vec(),
            p_accept,
            prior_by_position,
        })
    }

    /// 2026-10-04: The calibration file text [`Self::parse`] reads, checked by reading it back.
    pub fn render(&self) -> Result<String, String> {
        let d = &self.drafter;
        let text = format!(
            "[drafter]\nweights_sha256 = {:?}\nvocab = {}\nquantization = {:?}\ncontext = {}\n\n\
             [acceptance]\nedges = {:?}\np_accept = {:?}\nprior_by_position = {:?}\n",
            d.weights_sha256,
            d.vocab,
            d.quantization,
            d.context,
            self.edges,
            self.p_accept,
            self.prior_by_position
        );
        if Self::parse(&text)? != *self {
            return Err("acceptance calibration: the rendered file does not read back".into());
        }
        Ok(text)
    }
}

#[cfg(test)]
#[path = "spec_cost_fit_tests.rs"]
mod tests;
