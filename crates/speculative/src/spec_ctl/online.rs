// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The measured-only cost source: step costs learned online per (width bucket,
//! depth) with no table and no envelope. It is what the MTP gate measures today (delivered
//! throughput of plain decode vs the speculative step, per power-of-two width, re-measured on a
//! cadence), expressed as a cost source so the controller can choose any depth from it.
//!
//! Owner: speculative.
//! Invariants:
//! - A cell is priced only after it was measured; an unmeasured cell, or one not measured in
//!   the last `stale_after` observations, [`OnlineTable::needs_probe`], and the controller runs
//!   it once before it plans from it.
//! - The first measurement seeds a cell; later ones move it `alpha` of the way.

use std::collections::BTreeMap;

use super::calib::bucket;
use super::cost::StepCost;

#[derive(Clone, Copy, Debug, PartialEq)]
struct OnlineCell {
    ms: f64,
    j: Option<f64>,
    at: u64,
}

/// 2026-10-10: Learned step costs by (width bucket, depth).
#[derive(Clone, Debug, PartialEq)]
pub struct OnlineTable {
    alpha: f64,
    stale_after: u64,
    tick: u64,
    cells: BTreeMap<(usize, usize), OnlineCell>,
}

impl OnlineTable {
    /// 2026-10-10: Empty, with EWMA weight `alpha` (clamped to `0..=1`) and a re-measure
    /// interval of `stale_after` observations (at least 1).
    pub fn new(alpha: f64, stale_after: u64) -> Self {
        Self {
            alpha: if alpha.is_finite() {
                alpha.clamp(0.0, 1.0)
            } else {
                0.0
            },
            stale_after: stale_after.max(1),
            tick: 0,
            cells: BTreeMap::new(),
        }
    }

    /// 2026-10-10: The learned cost of `k` drafts at width `n`, `None` until measured.
    pub fn cost(&self, n: usize, k: usize) -> Option<StepCost> {
        self.cells
            .get(&(bucket(n), k))
            .map(|c| StepCost { ms: c.ms, j: c.j })
    }

    /// 2026-10-10: Whether the cell must be (re-)measured before it is planned from.
    pub fn needs_probe(&self, n: usize, k: usize) -> bool {
        self.cells
            .get(&(bucket(n), k))
            .is_none_or(|c| self.tick.saturating_sub(c.at) >= self.stale_after)
    }

    /// 2026-10-10: Fold one measured step. A non-finite or non-positive wall is ignored.
    pub fn observe(&mut self, n: usize, k: usize, ms: f64, j: Option<f64>) {
        if !(ms.is_finite() && ms > 0.0) {
            return;
        }
        self.tick += 1;
        let j = j.filter(|v| v.is_finite() && *v > 0.0);
        let (a, at) = (self.alpha, self.tick);
        self.cells
            .entry((bucket(n), k))
            .and_modify(|c| {
                c.ms += a * (ms - c.ms);
                c.j = match (c.j, j) {
                    (Some(old), Some(new)) => Some(old + a * (new - old)),
                    (_, new) => new,
                };
                c.at = at;
            })
            .or_insert(OnlineCell { ms, j, at });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cell_is_priced_after_its_first_measurement_and_follows_later_ones() {
        let mut t = OnlineTable::new(0.5, 4);
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

    /// 2026-10-10: A cell not measured for `stale_after` observations is due again.
    #[test]
    fn a_cell_goes_stale_while_others_are_measured() {
        let mut t = OnlineTable::new(0.3, 3);
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
