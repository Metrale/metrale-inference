// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Draft outcomes by drafter confidence: for each draft a verify step reached
//! (every draft before it was accepted), its top-1 log-probability bucket and whether it was
//! accepted. `met bench spec-cost` fits the acceptance calibration of `--spec-cost-model
//! measured` from these counts, so P(accept | bucket) is conditional on reaching the draft,
//! which is how the planner chains it.
//!
//! Owner: telemetry.
//! Invariants: a non-finite log-probability is counted in `overflow`, never in a bucket; a
//! positive one (rounding above 0) lands in the last bucket.

use crate::instrument::Counter;

/// 2026-10-04: Ascending upper edges of the buckets, in natural-log probability; a draft
/// lands in the first bucket whose edge is >= its log-probability. Fine near 0, where most
/// MTP drafts sit; the last edge is 0.
pub const EDGES: [f32; 10] = [
    -4.0, -2.0, -1.0, -0.5, -0.25, -0.1, -0.05, -0.02, -0.005, 0.0,
];

#[derive(Debug)]
pub struct ConfidenceOutcomes {
    /// 2026-10-04: `cells[b][accepted]`: reached drafts in bucket `b`.
    cells: [[Counter; 2]; EDGES.len()],
    pub overflow: Counter,
}

impl Default for ConfidenceOutcomes {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfidenceOutcomes {
    pub const fn new() -> Self {
        Self {
            cells: [const { [const { Counter::new() }; 2] }; EDGES.len()],
            overflow: Counter::new(),
        }
    }

    /// 2026-10-04: The bucket of log-probability `lp`; `None` when it is not finite.
    pub fn bucket(lp: f32) -> Option<usize> {
        if !lp.is_finite() {
            return None;
        }
        Some(
            EDGES
                .iter()
                .position(|&e| lp <= e)
                .unwrap_or(EDGES.len() - 1),
        )
    }

    pub(crate) fn record(&self, lp: f32, accepted: bool) {
        match Self::bucket(lp) {
            Some(b) => self.cells[b][usize::from(accepted)].add(1),
            None => self.overflow.add(1),
        }
    }

    /// 2026-10-04: Reached drafts in bucket `b` with outcome `accepted`.
    pub fn count(&self, b: usize, accepted: bool) -> u64 {
        self.cells
            .get(b)
            .map_or(0, |c| c[usize::from(accepted)].get())
    }

    pub fn is_empty(&self) -> bool {
        self.cells.iter().flatten().all(|c| c.get() == 0)
    }
}

#[cfg(test)]
#[path = "spec_confidence_tests.rs"]
mod tests;
