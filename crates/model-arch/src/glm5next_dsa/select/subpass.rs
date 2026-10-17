// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The row sub-pass plans of an exact selection pass whose scratch holds fewer rows
//! than the pass (moved out of `launch.rs`, which `select_tokens_rows` and
//! `select_tokens_passes` still launch them from): [`row_passes`], the unsplit pass's plan
//! (steps of `per` from each range's first row), and [`even_row_passes`], the same number of
//! sub-passes sized evenly, the sequence-parallel selection's plan for its slabs.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: both plans cover each range exactly, in order, in `ceil(len / per)` sub-passes of
//! 1 to `per` rows; they differ only in where the sub-pass boundaries fall.

use super::*;
use anyhow::ensure;

/// 2026-10-15: The row sub-passes `(first_row, rows)` that cover `ranges` (each `(lo, hi)`, in
/// order) at most `per` rows at a time: every range from its first row in steps of `per`, the
/// plan `select_tokens_rows` launches. Errors on `per == 0`.
pub fn row_passes(ranges: &[(usize, usize)], per: usize) -> Result<Vec<(usize, usize)>> {
    ensure!(per > 0, "DSA select: a sub-pass of no rows");
    Ok(ranges
        .iter()
        .flat_map(|&(lo, hi)| (lo..hi).step_by(per).map(move |r0| (r0, per.min(hi - r0))))
        .collect())
}

/// 2026-10-17: The row sub-passes that cover `ranges` in as many sub-passes as [`row_passes`]
/// plans (`ceil(len / per)` per range, each at most `per` rows), sized evenly: within a range
/// the sizes differ by at most one row, the larger first, so no range ends in a sub-pass
/// narrower than the rest (at `per >= 3` none has fewer than 2 rows). The sequence-parallel
/// selection runs its slabs in these (`sp_select::run::slab_passes`). Errors on `per == 0`.
pub fn even_row_passes(ranges: &[(usize, usize)], per: usize) -> Result<Vec<(usize, usize)>> {
    ensure!(per > 0, "DSA select: a sub-pass of no rows");
    let mut out = Vec::new();
    for &(lo, hi) in ranges {
        let len = hi.saturating_sub(lo);
        let n = len.div_ceil(per);
        let mut r0 = lo;
        for i in 0..n {
            let rows = len / n + usize::from(i < len % n);
            out.push((r0, rows));
            r0 += rows;
        }
    }
    Ok(out)
}

/// 2026-10-17: [`DsaSelectScratch::rows_per_pass`] of `geom`, an error when it is 0.
pub(super) fn sub_pass_rows(scratch: &DsaSelectScratch, geom: &DsaSelectGeometry) -> Result<usize> {
    let per = scratch.rows_per_pass(geom);
    if per == 0 {
        bail!(
            "DSA select: the scratch holds no query row at {} pools ({} tokens)",
            geom.n_pools,
            geom.seq
        );
    }
    Ok(per)
}

/// 2026-10-17: Errors unless every sub-pass of `passes` has 1 to `per` rows inside the pass's
/// `geom.q_rows` rows ([`DsaSelectScratch::rows_per_pass`] of `geom`, which must not be 0).
pub(super) fn check_passes(
    scratch: &DsaSelectScratch,
    geom: &DsaSelectGeometry,
    passes: &[(usize, usize)],
) -> Result<()> {
    let per = sub_pass_rows(scratch, geom)?;
    ensure!(
        passes
            .iter()
            .all(|&(r0, n)| (1..=per).contains(&n) && r0 + n <= geom.q_rows),
        "DSA select: sub-passes {passes:?} outside a {}-row pass of at most {per} rows each",
        geom.q_rows
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{even_row_passes, row_passes};

    /// 2026-10-17: Over every range of 1..=600 rows at `lo` 0 and 37 and every `per` 1..=40:
    /// the even plan covers the range exactly, contiguously and in order, in as many sub-passes
    /// as `row_passes`, each 1 to `per` rows, sizes within one row of each other and
    /// non-increasing; at `per >= 3` (and 2 or more rows) no sub-pass has fewer than 2 rows.
    /// Control: `row_passes` does leave a 1-row sub-pass at `per * m + 1` rows (`m >= 1`), the
    /// shape the even plan exists for.
    #[test]
    fn the_even_plan_covers_each_range_in_as_many_balanced_sub_passes() {
        for lo in [0usize, 37] {
            for len in 1..=600usize {
                let r = [(lo, lo + len)];
                for per in 1..=40usize {
                    let even = even_row_passes(&r, per).unwrap();
                    let step = row_passes(&r, per).unwrap();
                    assert_eq!(even.len(), step.len(), "len {len} per {per}");
                    assert_eq!(even[0].0, lo);
                    assert!(even.windows(2).all(|w| w[0].0 + w[0].1 == w[1].0));
                    let (last_r0, last_n) = even[even.len() - 1];
                    assert_eq!(last_r0 + last_n, lo + len, "len {len} per {per}");
                    let max = even.iter().map(|p| p.1).max().unwrap();
                    let min = even.iter().map(|p| p.1).min().unwrap();
                    assert!(
                        min >= 1 && max <= per && max - min <= 1,
                        "len {len} per {per}"
                    );
                    assert!(even.windows(2).all(|w| w[0].1 >= w[1].1));
                    if per >= 3 && len >= 2 {
                        assert!(min >= 2, "len {len} per {per}: {even:?}");
                    }
                    if len > per && len % per == 1 {
                        assert_eq!(step[step.len() - 1].1, 1, "control len {len} per {per}");
                    }
                }
            }
        }
    }

    /// 2026-10-17: Several ranges keep their own boundaries, and `per` 0 is an error.
    #[test]
    fn the_even_plan_keeps_each_range_and_refuses_an_empty_sub_pass() {
        let got = even_row_passes(&[(0, 7), (10, 13)], 3).unwrap();
        assert_eq!(got, vec![(0, 3), (3, 2), (5, 2), (10, 3)]);
        assert!(even_row_passes(&[(0, 7)], 0).is_err());
    }
}
