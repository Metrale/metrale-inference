// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: How a GLM prefill pass is cut on the model's prefill grid
//! (`ModelLevers::prefill_grid`, the scheduler's `prefill_plan::PREFILL_GRID`).
//!
//! A pass is cut at every absolute multiple of the grid G (counted from token 0) into cells, and
//! each cell into sub-chunks of `prefill_rows()` rows from the cell's first row: the units a
//! serve chunked at G runs, whatever this serve's budget ([`grid_units`]). Consecutive units
//! narrower than a sub-chunk together may share one call ([`plan_calls`]): the launches whose
//! arithmetic depends on the row count (the mixer, the router GEMM, the shared expert, a dense
//! MLP site, the non-SP all-reduces) still run once per unit, and the row-invariant ones (the mHC
//! and norm kernels, the top-k, the grouped routed experts, the combine) once over the call, so
//! a `--max-prefill-tokens 16384` serve reads the routed-expert weights once per 16384 rows and
//! computes every row with the bits of the 8192-chunk serve.
//!
//! Before this date the cut was the opt-in `METRALE_GLM_PREFILL_PIECE_ROWS=P`, taken from each
//! sub-chunk's first row; it is retired, and a serve that sets it is refused
//! ([`refuse_retired_env`]).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - [`grid_units`] covers the pass once, in order; no unit crosses a grid line or is wider
//!   than `rows`; without a grid the units are `rows` wide from row 0, the last shorter.
//! - The grid is a multiple of [`PIECE_ALIGN`], so a cell boundary is a boundary of the KDA
//!   chunked scan's 256-row slabs.
//! - [`plan_calls`] keeps the units in order and puts at most `cap` rows in a call.

use anyhow::Result;

/// 2026-10-14: The retired piece lever's name; setting it is refused at load.
pub const PIECE_ROWS_ENV: &str = "METRALE_GLM_PREFILL_PIECE_ROWS";

/// 2026-10-14: The grid must be a multiple of this: the KDA chunked scan's slab
/// (`glm5next_kda`).
pub const PIECE_ALIGN: usize = 256;

/// 2026-10-14: Refuses a serve that sets [`PIECE_ROWS_ENV`]: the prefill grid now cuts every
/// GLM pass, so the lever would be silently ignored.
pub fn refuse_retired_env() -> Result<()> {
    if let Some(v) = std::env::var_os(PIECE_ROWS_ENV) {
        anyhow::bail!(
            "{PIECE_ROWS_ENV}={v:?} is retired: a GLM prefill pass is cut on the prefill grid \
             whatever --max-prefill-tokens; unset it"
        );
    }
    Ok(())
}

/// 2026-10-14: The unit widths of the `n` rows of a pass from absolute position `start`: cut at
/// every multiple of `grid` (when set), then each cell into `rows`-wide sub-chunks from its
/// first row. A `rows` of 0 is treated as 1. Errors on a grid off [`PIECE_ALIGN`].
pub fn grid_units(start: usize, n: usize, rows: usize, grid: Option<usize>) -> Result<Vec<usize>> {
    let rows = rows.max(1);
    if let Some(g) = grid {
        anyhow::ensure!(
            g > 0 && g.is_multiple_of(PIECE_ALIGN),
            "prefill grid {g}: must be a positive multiple of {PIECE_ALIGN}"
        );
    }
    let mut out = Vec::new();
    let mut t = 0usize;
    for cell_end in grid_lines(start, n, grid).into_iter().chain([n]) {
        while t < cell_end {
            let k = rows.min(cell_end - t);
            out.push(k);
            t += k;
        }
    }
    Ok(out)
}

/// 2026-10-14: The rows strictly inside a sequence of `n` rows from absolute position `start`
/// at which a multiple of `grid` falls; none without a grid: the cell bounds of [`grid_units`].
pub fn grid_lines(start: usize, n: usize, grid: Option<usize>) -> Vec<usize> {
    let Some(g) = grid.filter(|&g| g > 0) else {
        return Vec::new();
    };
    (start / g + 1..)
        .map(|j| j * g - start)
        .take_while(|&r| r < n)
        .collect()
}

/// 2026-10-14: The calls the `units` of a pass run as, each a list of consecutive unit widths
/// totalling at most `cap` rows. A unit joins the open call only when `share` of the unit and of
/// the call total with it are both `Some` and equal to `share` of the call's first unit; `share`
/// names the row-count-dependent decisions a call makes (its routed path, SP-mHC) and is asked
/// only where a join fits in `cap`. An error from it is returned unchanged.
pub fn plan_calls<S: PartialEq>(
    units: &[usize],
    cap: usize,
    mut share: impl FnMut(usize) -> Result<Option<S>>,
) -> Result<Vec<Vec<usize>>> {
    let mut calls: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    // 2026-10-14: `share` of the open call's first unit, once asked.
    let mut first: Option<Option<S>> = None;
    for &n in units {
        anyhow::ensure!(n > 0, "prefill unit of 0 rows in {units:?}");
        let total = cur.iter().sum::<usize>() + n;
        if !cur.is_empty() && total <= cap {
            if first.is_none() {
                first = Some(share(cur[0])?);
            }
            if let Some(Some(f)) = &first
                && share(n)?.as_ref() == Some(f)
                && share(total)?.as_ref() == Some(f)
            {
                cur.push(n);
                continue;
            }
        }
        if !cur.is_empty() {
            calls.push(std::mem::take(&mut cur));
        }
        cur.push(n);
        first = None;
    }
    if !cur.is_empty() {
        calls.push(cur);
    }
    Ok(calls)
}

#[cfg(test)]
#[path = "glm5next_prefill_piece_tests.rs"]
mod tests;
