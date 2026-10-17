// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: An explicit per-concurrency DFlash draft ladder (`--dflash-draft-ladder`): the
//! drafts per verify step by the number of active sequences, as `min_active:drafts` pairs.
//!
//! Owner: speculative.
//! Invariants:
//! - A parsed ladder starts at 1 active sequence, its thresholds strictly increase and every
//!   draft count is at least 1, so every concurrency has exactly one rung.
//!
//! On a routed-MoE target a wider verify reads the union of more rows' experts, so a ladder
//! that narrows the block at low concurrency and widens it again once the union saturates can
//! beat one fixed width. The GLM-5.3 Flash DFlash2 reference serve runs
//! `1:7,2:5,3:4,4:3,5:2,6:7` (drafts per step, before the anchor row).

use anyhow::{Result, bail};

/// 2026-10-09: `(min_active, drafts)` rungs, ascending by `min_active`, the first at 1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftLadder(Vec<(usize, usize)>);

impl DraftLadder {
    /// 2026-10-09: Parse `min_active:drafts[,min_active:drafts...]`. Errors on an empty or
    /// malformed pair, a ladder that does not start at 1, thresholds that do not strictly
    /// increase, or a draft count of 0.
    pub fn parse(spec: &str) -> Result<Self> {
        let mut rungs = Vec::new();
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

    /// 2026-10-09: The drafts of the rung for `n_active` sequences: the last rung whose
    /// threshold is at most `n_active` (the first rung for 0).
    pub fn drafts(&self, n_active: usize) -> usize {
        self.0
            .iter()
            .rev()
            .find(|(a, _)| *a <= n_active.max(1))
            .map_or(self.0[0].1, |r| r.1)
    }

    /// 2026-10-09: The widest rung's drafts, which the serve's γ must cover.
    pub fn max_drafts(&self) -> usize {
        self.0.iter().map(|r| r.1).max().unwrap_or(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-09: The reference table: every concurrency maps to its rung, and 6 and above
    /// widen again.
    #[test]
    fn the_reference_ladder_maps_every_concurrency() {
        let l = DraftLadder::parse("1:7,2:5,3:4,4:3,5:2,6:7").unwrap();
        let got: Vec<usize> = (0..=8).map(|n| l.drafts(n)).collect();
        assert_eq!(got, vec![7, 7, 5, 4, 3, 2, 7, 7, 7]);
        assert_eq!(l.drafts(64), 7);
        assert_eq!(l.max_drafts(), 7);
    }

    #[test]
    fn a_malformed_ladder_is_refused() {
        for bad in ["", "2:5", "1:0", "1:7,1:5", "1:7,3:4,2:5", "1-7", "1:x"] {
            assert!(DraftLadder::parse(bad).is_err(), "{bad:?} accepted");
        }
    }
}
