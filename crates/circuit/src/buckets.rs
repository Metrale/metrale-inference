// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The row buckets of a mode (LIFECYCLE-DESIGN.md sections 4.2 and 15.4): the ranges
//! of rows inside which no rule and no runtime route of the mode changes, so one plan, fused at
//! the bucket's top, is exact for every row count in it. The ladder is derived from the rule set,
//! never chosen, and the golden plans pin it.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - The buckets tile `1..=max_rows` exactly, in order, with no gap and no overlap.
//! - Every boundary is a rule's or a route's `rows` edge (`lo`, or `hi + 1`) in the mode.

use crate::rules::{Mode, Rule};
use crate::runtime::RuntimeRoute;

/// 2026-10-03: Row counts `lo..=hi` that one plan serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Bucket {
    pub lo: u64,
    pub hi: u64,
}

impl Bucket {
    /// 2026-10-03: Whether `rows` falls in it.
    pub fn holds(self, rows: u64) -> bool {
        (self.lo..=self.hi).contains(&rows)
    }
}

/// 2026-10-03: The buckets of `mode` over `1..=max_rows`: split at every row edge of a rule or
/// route that applies in `mode`. `max_rows` 0 gives no bucket.
pub fn bucket_ladder(
    rules: &[Rule],
    routes: &[RuntimeRoute],
    mode: Mode,
    max_rows: u64,
) -> Vec<Bucket> {
    if max_rows == 0 {
        return Vec::new();
    }
    let ranges = rules
        .iter()
        .filter(|r| r.modes.contains(&mode))
        .map(|r| r.rows)
        .chain(
            routes
                .iter()
                .filter(|r| r.modes.contains(&mode))
                .map(|r| r.rows),
        );
    let mut starts: Vec<u64> = std::iter::once(1)
        .chain(ranges.flat_map(|(lo, hi)| [lo, hi.saturating_add(1)]))
        .filter(|&s| (2..=max_rows).contains(&s) || s == 1)
        .collect();
    starts.sort_unstable();
    starts.dedup();
    starts
        .iter()
        .enumerate()
        .map(|(i, &lo)| Bucket {
            lo,
            hi: starts.get(i + 1).map_or(max_rows, |next| next - 1),
        })
        .collect()
}

/// 2026-10-03: The bucket of `ladder` holding `rows`; `None` above the ladder.
pub fn bucket_of(ladder: &[Bucket], rows: u64) -> Option<Bucket> {
    ladder.iter().copied().find(|b| b.holds(rows))
}

#[cfg(test)]
#[path = "buckets_tests.rs"]
mod buckets_tests;
