// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The measured-only cost source: step costs learned online per (width bucket,
//! depth) with no table and no envelope. It is what the MTP gate measures today (delivered
//! throughput of plain decode vs the speculative step, per power-of-two width, re-measured on a
//! cadence), expressed as a cost source so the controller can choose any depth from it.
//!
//! Owner: speculative.
//! Invariants:
//! - A cell is priced only after `warmup` measurements; until then, or when it was not measured
//!   in the last `stale_after` observations, [`OnlineTable::needs_probe`], and the controller
//!   runs it before it plans from it.
//! - A cold cell keeps its fastest warm-up sample: the first step at a new width often carries
//!   a one-off (a CUDA graph capture, an allocation), which must not price the depth for good.
//!   After warm-up a sample moves the cell `alpha` of the way, clipped at `clip` times the
//!   current value, so one stalled step cannot flip the decision. (2026-10-10, dgx1 A/B: an
//!   unclipped first-sample seed left the controller plain-decoding 62% of wide-batch steps.)

use std::collections::BTreeMap;

use super::calib::bucket;
use super::cost::StepCost;

#[derive(Clone, Copy, Debug, PartialEq)]
struct OnlineCell {
    ms: f64,
    j: Option<f64>,
    at: u64,
    samples: u32,
}

/// 2026-10-10: Learned step costs by (width bucket, depth).
#[derive(Clone, Debug, PartialEq)]
pub struct OnlineTable {
    alpha: f64,
    stale_after: u64,
    warmup: u32,
    clip: f64,
    tick: u64,
    cells: BTreeMap<(usize, usize), OnlineCell>,
}

impl OnlineTable {
    /// 2026-10-10: Empty, with EWMA weight `alpha` (clamped to `0..=1`), a re-measure
    /// interval of `stale_after` observations (at least 1), `warmup` measurements (at least 1)
    /// before a cell is priced, and the outlier clip `clip` (at least 1).
    pub fn new(alpha: f64, stale_after: u64, warmup: u32, clip: f64) -> Self {
        Self {
            alpha: if alpha.is_finite() {
                alpha.clamp(0.0, 1.0)
            } else {
                0.0
            },
            stale_after: stale_after.max(1),
            warmup: warmup.max(1),
            clip: if clip.is_finite() { clip.max(1.0) } else { 1.0 },
            tick: 0,
            cells: BTreeMap::new(),
        }
    }

    /// 2026-10-10: The learned cost of `k` drafts at width `n`, `None` until measured.
    pub fn cost(&self, n: usize, k: usize) -> Option<StepCost> {
        self.cells
            .get(&(bucket(n), k))
            .filter(|c| c.samples >= self.warmup)
            .map(|c| StepCost { ms: c.ms, j: c.j })
    }

    /// 2026-10-10: Whether the cell must be (re-)measured before it is planned from.
    pub fn needs_probe(&self, n: usize, k: usize) -> bool {
        self.cells.get(&(bucket(n), k)).is_none_or(|c| {
            c.samples < self.warmup || self.tick.saturating_sub(c.at) >= self.stale_after
        })
    }

    /// 2026-10-10: Fold one measured step. A non-finite or non-positive wall is ignored.
    pub fn observe(&mut self, n: usize, k: usize, ms: f64, j: Option<f64>) {
        if !(ms.is_finite() && ms > 0.0) {
            return;
        }
        self.tick += 1;
        let j = j.filter(|v| v.is_finite() && *v > 0.0);
        let (a, at, warmup, clip) = (self.alpha, self.tick, self.warmup, self.clip);
        self.cells
            .entry((bucket(n), k))
            .and_modify(|c| {
                if c.samples < warmup {
                    c.ms = c.ms.min(ms);
                    c.j = match (c.j, j) {
                        (Some(old), Some(new)) => Some(old.min(new)),
                        (_, new) => new,
                    };
                } else {
                    c.ms += a * (ms.min(clip * c.ms) - c.ms);
                    c.j = match (c.j, j) {
                        (Some(old), Some(new)) => Some(old + a * (new.min(clip * old) - old)),
                        (_, new) => new,
                    };
                }
                c.samples = c.samples.saturating_add(1);
                c.at = at;
            })
            .or_insert(OnlineCell {
                ms,
                j,
                at,
                samples: 1,
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cell_is_priced_after_warm_up_and_follows_later_ones() {
        let mut t = OnlineTable::new(0.5, 4, 1, 10.0);
        assert!(t.needs_probe(1, 2) && t.cost(1, 2).is_none());
        t.observe(1, 2, 50.0, Some(3.0));
        assert_eq!(
            t.cost(1, 2),
            Some(StepCost {
                ms: 50.0,
                j: Some(3.0)
            })
        );
        t.observe(1, 2, 60.0, None);
        assert_eq!(t.cost(1, 2).unwrap().ms, 55.0);
        assert_eq!(
            t.cost(1, 2).unwrap().j,
            None,
            "a step without joules forgets them"
        );
        t.observe(1, 2, f64::NAN, None);
        assert_eq!(t.cost(1, 2).unwrap().ms, 55.0);
        assert!(t.needs_probe(2, 2), "another width bucket is another cell");
        assert!(!t.needs_probe(1, 2));
    }

    /// 2026-10-10: A capture spike on a cold cell does not price it: the cell keeps its fastest
    /// warm-up sample; after warm-up one stalled step moves it by at most `alpha` x `clip`.
    #[test]
    fn a_cold_spike_does_not_price_a_depth_and_a_stall_is_clipped() {
        let mut t = OnlineTable::new(0.3, 256, 3, 2.0);
        t.observe(16, 1, 900.0, None);
        assert!(
            t.needs_probe(16, 1) && t.cost(16, 1).is_none(),
            "one sample is not a price"
        );
        t.observe(16, 1, 60.0, None);
        t.observe(16, 1, 62.0, None);
        assert!(!t.needs_probe(16, 1));
        assert_eq!(
            t.cost(16, 1).unwrap().ms,
            60.0,
            "the fastest warm-up sample"
        );
        t.observe(16, 1, 6000.0, None);
        assert!(
            (t.cost(16, 1).unwrap().ms - 78.0).abs() < 1e-9,
            "60 + 0.3 x (120 - 60)"
        );
    }

    /// 2026-10-10: A cell not measured for `stale_after` observations is due again.
    #[test]
    fn a_cell_goes_stale_while_others_are_measured() {
        let mut t = OnlineTable::new(0.3, 3, 1, 10.0);
        t.observe(1, 0, 28.0, None);
        for _ in 0..2 {
            t.observe(1, 2, 50.0, None);
            assert!(!t.needs_probe(1, 0));
        }
        t.observe(1, 2, 50.0, None);
        assert!(t.needs_probe(1, 0));
        assert!(!t.needs_probe(1, 2));
    }
}
