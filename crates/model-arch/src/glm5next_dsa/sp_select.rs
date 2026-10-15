// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The rank-local half of the sequence-parallel DSA selection
//! (`METRALE_GLM_DSA_SP_SELECT`): which query
//! rows of a prefill pass each rank selects, how the selected rows are all-gathered, and whether
//! a pass splits at all.
//!
//! Every rank runs scores, top-k and expand only on the rows it owns, then the rows are
//! all-gathered in place (`glm5next_sp_mhc::all_gather_rows`), so every rank holds the whole
//! `[k, out_width]` selection. The split is by query row: a row's whole top-k domain is computed
//! on one rank, so the result is the unsplit selection's bytes (`sp_select/tie_tests.rs`).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - [`slabs`] cuts `0..k` into `2 * world` contiguous slabs whose sizes differ by at most one
//!   row; rank `r` owns slabs `r` and `2 * world - 1 - r` (zigzag), so under causality, where a
//!   row's scores cost grows with its position, every rank gets one early and one late slab.
//! - Every row is owned by exactly one rank, and the two [`gather_shards`] lists put each rank's
//!   slab at its own index, the shape `all_gather_plan` takes.
//! - [`applies`] is a pure function of host integers every rank holds identically (the lever,
//!   the world size, the pass's rows and pools, the prefill flag, the startup-loaded table), so
//!   all ranks take the same branch and the gather can never be entered by one rank alone. It is
//!   false whenever `is_prefill` is false: no decode step reaches a collective through it.
//! - [`MinPoolsTable`] holds only measured points: a `k` outside its measured range never
//!   splits, and between two measured `k` the smaller one's threshold applies.

use anyhow::{Result, ensure};

use crate::glm5next_sp_mhc::shards;

/// 2026-10-14: Slabs each rank owns: one early, one late (zigzag).
pub const SLABS_PER_RANK: usize = 2;

/// 2026-10-15: The fewest rows any selection sub-pass of a split pass may have, unsplit or on any
/// rank: a 1-row sub-pass takes other scores kernels (`scores_mma_for` needs `q_rows > 1`) and
/// so other bits, so a pass whose plan holds one never splits (bit identity, not a tuning knob).
pub const MIN_SUBPASS_ROWS: usize = 2;

/// 2026-10-14: The `2 * world` contiguous row slabs of a `k`-row pass.
pub fn slabs(k: usize, world: usize) -> Vec<(usize, usize)> {
    shards(k, SLABS_PER_RANK * world)
}

/// 2026-10-14: The two slabs rank `rank` selects: slab `rank` and slab `2 * world - 1 - rank`.
/// Errors on an empty slab (fewer than `2 * world` rows), which the gather cannot send.
pub fn owned(k: usize, world: usize, rank: usize) -> Result<[(usize, usize); SLABS_PER_RANK]> {
    ensure!(
        rank < world,
        "DSA SP select: rank {rank} outside a world of {world}"
    );
    ensure!(
        k >= SLABS_PER_RANK * world,
        "DSA SP select: {k} rows cannot fill {} slabs",
        SLABS_PER_RANK * world
    );
    let s = slabs(k, world);
    Ok([s[rank], s[SLABS_PER_RANK * world - 1 - rank]])
}

/// 2026-10-14: The shard lists of the two in-place gathers: entry `r` of list `i` is rank `r`'s
/// `i`-th owned slab ([`owned`]).
pub fn gather_shards(k: usize, world: usize) -> [Vec<(usize, usize)>; SLABS_PER_RANK] {
    let s = slabs(k, world);
    let last = SLABS_PER_RANK * world - 1;
    [
        (0..world).map(|r| s[r]).collect(),
        (0..world).map(|r| s[last - r]).collect(),
    ]
}

/// 2026-10-14: The measured split threshold: for each measured pass width `k`, the fewest
/// candidate pools at which the split was measured faster than the unsplit selection by more
/// than its margin. 2026-10-15: Read once from the lever's value (`sp_select::lever`), the
/// table the single-GPU gate measured; there is no built-in table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MinPoolsTable {
    rows: Vec<(usize, usize)>,
}

impl MinPoolsTable {
    /// 2026-10-14: `rows` are `(k, min_pools)`, `k` strictly increasing; at least one row.
    pub fn new(rows: Vec<(usize, usize)>) -> Result<Self> {
        ensure!(
            !rows.is_empty(),
            "DSA SP select: the min-pools table has no measured row"
        );
        ensure!(
            rows.windows(2).all(|w| w[0].0 < w[1].0),
            "DSA SP select: the min-pools table's k must strictly increase: {rows:?}"
        );
        Ok(Self { rows })
    }

    /// 2026-10-14: Whether a pass of `k` rows over `pools` candidate pools splits: `k` within
    /// the measured range, and `pools` at least the threshold of the largest measured `k` not
    /// above it. The saving grows with `pools` while the gather's bytes do not depend on it, so
    /// a measured threshold also holds above it.
    pub fn split(&self, k: usize, pools: usize) -> bool {
        let last = self.rows[self.rows.len() - 1].0;
        if k > last {
            return false;
        }
        match self.rows.iter().rev().find(|&&(rk, _)| rk <= k) {
            Some(&(_, min_pools)) => pools >= min_pools,
            None => false,
        }
    }

    /// 2026-10-15: The table as the lever spells it, `k:min_pools` rows joined by `,`
    /// ([`lever::parse`] reads it back to the same table).
    pub fn canonical(&self) -> String {
        self.rows
            .iter()
            .map(|(k, p)| format!("{k}:{p}"))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// 2026-10-14: Whether a pass splits its selection over the ranks: the lever on, a prefill pass,
/// two or more ranks, every slab non-empty, and the measured table says so.
pub fn applies(
    on: bool,
    world: usize,
    k: usize,
    pools: usize,
    is_prefill: bool,
    table: &MinPoolsTable,
) -> bool {
    on && is_prefill && world >= 2 && k >= SLABS_PER_RANK * world && table.split(k, pools)
}

pub mod lever;
pub mod run;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod run_tests;

#[cfg(test)]
mod tie_tests;
