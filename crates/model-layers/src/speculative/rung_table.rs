// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: One draft-count-by-width table for every drafter: the MTP K-ladder
//! (`--mtp-k-ladder`, `n_max:drafts`, upper bounds) and the DFlash draft ladder
//! (`--dflash-draft-ladder`, `min_active:drafts`, lower bounds) are two spellings of it.
//!
//! Owner: model-layers (speculative).
//! Invariants:
//! - Rungs are stored as `(min_active, drafts)` with `min_active` non-decreasing; the rung for
//!   `n` active sequences is the last whose `min_active <= n`, else the first. The last rung
//!   covers every wider batch.
//! - [`RungTable::from_upper_bounds`] answers exactly what the upper-bound lookup ("the first
//!   step whose `n_max >= n`, else the last") answers, duplicates included.
//!
//! The speculation controller (`metrale_speculative::spec_ctl`) reads a table as an explicit
//! override or as its cold-start prior; the verify pools (`ssm_reserve`) read the MTP one.

use anyhow::{Result, bail};

/// 2026-10-10: Drafts per verify step by active width (see the module doc).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RungTable(Vec<(usize, usize)>);

impl RungTable {
    /// 2026-10-10: From `(n_max, drafts)` steps sorted by `n_max`: step `i` covers
    /// `n_max[i-1] + 1 ..= n_max[i]` (the first from 0). `None` when `steps` is empty.
    pub fn from_upper_bounds(steps: &[(usize, usize)]) -> Option<Self> {
        let first = steps.first()?;
        let mut rungs = vec![(0, first.1)];
        for w in steps.windows(2) {
            rungs.push((w[0].0 + 1, w[1].1));
        }
        Some(Self(rungs))
    }

    /// 2026-10-10: Parse `min_active:drafts[,min_active:drafts...]`. Errors on an empty or
    /// malformed pair, a table that does not start at 1, thresholds that do not strictly
    /// increase, or a draft count of 0.
    pub fn parse_lower_bounds(spec: &str) -> Result<Self> {
        let mut rungs: Vec<(usize, usize)> = Vec::new();
        for pair in spec.split(',') {
            let Some((a, d)) = pair.trim().split_once(':') else {
                bail!("draft ladder pair {pair:?} is not min_active:drafts");
            };
            let (a, d): (usize, usize) = (a.trim().parse()?, d.trim().parse()?);
            if d == 0 {
                bail!("draft ladder rung {a}:{d}: at least one draft per step");
            }
            if let Some(&(prev, _)) = rungs.last()
                && a <= prev
            {
                bail!("draft ladder thresholds must increase: {a} after {prev}");
            }
            rungs.push((a, d));
        }
        if rungs.first().map(|r| r.0) != Some(1) {
            bail!("draft ladder must start at 1 active sequence: {spec:?}");
        }
        Ok(Self(rungs))
    }

    /// 2026-10-10: From `(min_active, drafts)` rungs already in lower-bound form (a derived
    /// cold-start table). `None` when empty or when `min_active` decreases.
    pub fn from_lower_bounds(rungs: Vec<(usize, usize)>) -> Option<Self> {
        let ordered = rungs.windows(2).all(|w| w[0].0 <= w[1].0);
        (!rungs.is_empty() && ordered).then_some(Self(rungs))
    }

    /// 2026-10-10: The drafts of the rung for `n_active` sequences.
    pub fn drafts(&self, n_active: usize) -> usize {
        self.0
            .iter()
            .rev()
            .find(|(lo, _)| *lo <= n_active)
            .map_or(self.0[0].1, |r| r.1)
    }

    /// 2026-10-10: The widest rung's drafts.
    pub fn max_drafts(&self) -> usize {
        self.0.iter().map(|r| r.1).max().unwrap_or(0)
    }

    /// 2026-10-10: The rungs as `(min_active, drafts)`.
    pub fn rungs(&self) -> &[(usize, usize)] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-10: The upper-bound form answers the upper-bound lookup at every width, for
    /// sorted steps with a duplicate `n_max`.
    #[test]
    fn upper_bounds_match_the_first_covering_step_lookup() {
        let steps = [(4, 3), (4, 2), (8, 1), (16, 2), (32, 1)];
        let t = RungTable::from_upper_bounds(&steps).unwrap();
        for n in 0..=64 {
            let want = steps
                .iter()
                .find(|&&(m, _)| n <= m)
                .or(steps.last())
                .unwrap()
                .1;
            assert_eq!(t.drafts(n), want, "n={n}");
        }
        assert!(RungTable::from_upper_bounds(&[]).is_none());
    }

    /// 2026-10-10: The GLM reference DFlash table: every concurrency maps to its rung, 0 to
    /// the first, and 6 and above widen again.
    #[test]
    fn the_reference_lower_bound_ladder_maps_every_concurrency() {
        let l = RungTable::parse_lower_bounds("1:7,2:5,3:4,4:3,5:2,6:7").unwrap();
        let got: Vec<usize> = (0..=8).map(|n| l.drafts(n)).collect();
        assert_eq!(got, vec![7, 7, 5, 4, 3, 2, 7, 7, 7]);
        assert_eq!(l.drafts(64), 7);
        assert_eq!(l.max_drafts(), 7);
    }

    #[test]
    fn a_malformed_lower_bound_ladder_is_refused() {
        for bad in ["", "2:5", "1:0", "1:7,1:5", "1:7,3:4,2:5", "1-7", "1:x"] {
            assert!(
                RungTable::parse_lower_bounds(bad).is_err(),
                "{bad:?} accepted"
            );
        }
        assert!(RungTable::from_lower_bounds(vec![]).is_none());
        assert!(RungTable::from_lower_bounds(vec![(4, 1), (2, 3)]).is_none());
    }
}
