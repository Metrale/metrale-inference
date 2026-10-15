// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `METRALE_GLM_PREFILL_CELL_GRID=F,C` (unset: off): a GLM prefill pass runs its rows
//! as cells cut at the absolute sequence positions `F, F + C, F + 2C, ...`, each cell exactly the
//! rows a serve chunked on that grid runs as one pass, with each KDA layer's SSM state clamp
//! (`ssm_state_clamp_norm_fused`, the kernel the model runs between passes) between that layer's
//! cells. With `--max-prefill-tokens 16384`, `METRALE_GLM_PREFILL_ROWS=8208` and `F,C = 8208,8192`
//! a pass holds two cells of the `--max-prefill-tokens 8192` serve's passes (its idle first
//! chunk 8208 rows, the next ones 8192): the same launches with the same shapes and inputs, in
//! half the passes, so a later SP pipeline can overlap one cell's exchanges with the other's
//! compute.
//!
//! Why the bits are the 8192 serve's. Every GLM launch is per sub-chunk; a cell starts and ends
//! where that serve's pass does and is at most `PREFILL_ROWS` rows, so it is one sub-chunk with
//! that pass's start, length and SP partition. The state a later cell reads is the earlier
//! cells' with the clamp applied, as between passes: the clamp is per (head, layer) block, so
//! clamping one layer between its cells equals clamping all layers at the pass end. A capture
//! on a cell boundary is taken before that clamp, as the model's chunk-end snapshot is taken
//! inside the pass, before the scheduler's clamp.
//!
//! The clamp schedule is mirrored, not changed. The grid is anchored at absolute position 0,
//! the scheduler's chunk origin, also on a prefix hit (the scheduler still walks every chunk
//! from 0; fully cached ones return early and are still clamped). Two pre-existing asymmetries
//! of the 8192 serve are reproduced exactly (changing either is a numerics change, class N):
//! - Rank 0's chunk 0 is not clamped (`prefill_a_step` does not call `normalize_ssm_states`;
//!   every later chunk and every chunk of a worker rank is): rank 0 skips the cell-boundary
//!   clamp at `F` ([`plan`]'s `rank0`), and clamps at the end of its chunk 0 when that chunk
//!   ends past `F` ([`rank0_first_chunk_clamp`]), where the 8192 serve's chunk 1 ends.
//! - On a hit restoring the state at `u`, every fully cached chunk clamps the restored state
//!   once more (the clamp is not idempotent in floating point): the restored state takes the
//!   clamps of the 8192 serve's cached chunks the 16384 serve has no chunk end for
//!   ([`restored_clamps`]) before the first computed row.
//!
//! The clamp scales a (head, layer) block only when its Frobenius norm exceeds 200; below that
//! every extra or missing clamp is a no-op, so a short prompt cannot show a schedule defect.
//! Only the idle cold path's chunking is mirrored: chunk 0 of `max_batch_tokens`, then
//! `--max-prefill-tokens` each ([`check_cell_budget`] at load).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Unset, `prefill_segments` runs its pre-existing plan unchanged (the planner is not called)
//!   and the model runs no extra clamp.
//! - [`plan`] covers `0..n` once, in order; every segment lies inside one cell; a capture point
//!   strictly inside a cell keeps the pre-existing mode's handling; one on a cell boundary
//!   captures after the segment that ends there.
//! - At load ([`check_cell_budget`]) every chunk the scheduler forms on the idle cold path
//!   starts and ends on the grid, and no cell exceeds `PREFILL_ROWS`.

use std::sync::OnceLock;

use anyhow::Result;

/// 2026-10-15: The env lever's name.
pub const CELL_GRID_ENV: &str = "METRALE_GLM_PREFILL_CELL_GRID";

/// 2026-10-15: The absolute grid: points at `first + k * step`, `k >= 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellGrid {
    pub first: usize,
    pub step: usize,
}

/// 2026-10-15: [`CELL_GRID_ENV`], read once: unset is off (`None`); otherwise `F,C` with both
/// positive, anything else an error.
pub fn cell_grid() -> Result<Option<CellGrid>> {
    static V: OnceLock<Result<Option<CellGrid>, String>> = OnceLock::new();
    V.get_or_init(|| parse(std::env::var(CELL_GRID_ENV).ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)
}

/// 2026-10-15: [`cell_grid`]'s parse, pure.
pub(crate) fn parse(v: Option<&str>) -> Result<Option<CellGrid>, String> {
    let Some(s) = v else {
        return Ok(None);
    };
    let bad = || format!("{CELL_GRID_ENV}={s:?}: expected `first,step`, two positive row counts");
    let (f, c) = s.split_once(',').ok_or_else(bad)?;
    let first: usize = f.parse().map_err(|_| bad())?;
    let step: usize = c.parse().map_err(|_| bad())?;
    if first == 0 || step == 0 {
        return Err(bad());
    }
    Ok(Some(CellGrid { first, step }))
}

impl CellGrid {
    /// 2026-10-15: Whether absolute position `a` is a grid point.
    pub fn is_point(&self, a: usize) -> bool {
        a >= self.first && (a - self.first).is_multiple_of(self.step)
    }

    /// 2026-10-15: The smallest grid point strictly after absolute position `a`.
    pub fn next_after(&self, a: usize) -> usize {
        if a < self.first {
            self.first
        } else {
            self.first + self.step * ((a - self.first) / self.step + 1)
        }
    }

    /// 2026-10-15: The cells of a pass of `n` rows from absolute position `seq_start`, as
    /// `(start, end)` offsets into the pass, in order: cut at every grid point strictly inside.
    pub fn cells(&self, seq_start: usize, n: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut t = 0usize;
        while t < n {
            let end = (self.next_after(seq_start + t) - seq_start).min(n);
            out.push((t, end));
            t = end;
        }
        out
    }
}

/// 2026-10-15: One segment of a pass under the grid: rows `start..end` of the pass (inside one
/// cell), the capture after it (a point on its end), the KDA cuts inside it (merge mode, offsets
/// into the segment), and whether it ends a cell before the pass end (the clamp runs after it).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Seg<W> {
    pub start: usize,
    pub end: usize,
    pub after: Option<W>,
    pub cuts: Vec<(usize, W)>,
    pub clamp_after: bool,
    /// 2026-10-15: The segment is its whole cell (a pass of the grid serve).
    pub whole_cell: bool,
    /// 2026-10-15: Split mode cut its cell at a capture point (the grid serve's split pass).
    pub split_in_cell: bool,
}

/// 2026-10-15: The segments of a pass of `n` rows from absolute `seq_start` under `grid`, with
/// the capture points `points` (offsets into the pass, strictly inside it, ascending) handled as
/// the pre-existing mode does inside each cell: `merge` keeps them as KDA cuts of the cell's
/// sub-chunk (`glm5next_tail_merge::merge_plan` over the cell), split mode cuts the cell into
/// segments with a capture after each. A point on a cell boundary captures after the segment
/// ending there in either mode. A cell end before the pass end clamps
/// after it, except the end at the grid's first point on rank 0 (`rank0`), the 8192 serve's
/// unclamped chunk 0 there. Pure.
pub fn plan<W: Copy>(
    grid: CellGrid,
    seq_start: usize,
    n: usize,
    rows: usize,
    merge: bool,
    points: &[(usize, W)],
    rank0: bool,
) -> Vec<Seg<W>> {
    let cells = grid.cells(seq_start, n);
    let last = cells.len().saturating_sub(1);
    let mut out = Vec::new();
    for (ci, &(cs, ce)) in cells.iter().enumerate() {
        let inside: Vec<(usize, W)> = points
            .iter()
            .filter(|&&(p, _)| p > cs && p < ce)
            .map(|&(p, w)| (p - cs, w))
            .collect();
        let at_end = points.iter().find(|&&(p, _)| p == ce).map(|&(_, w)| w);
        let mut segs: Vec<Seg<W>> = if merge {
            crate::glm5next_tail_merge::merge_plan(ce - cs, rows, &inside)
                .into_iter()
                .map(|(s, e, cuts)| Seg {
                    start: cs + s,
                    end: cs + e,
                    after: None,
                    cuts,
                    clamp_after: false,
                    whole_cell: false,
                    split_in_cell: false,
                })
                .collect()
        } else {
            let mut v = Vec::with_capacity(inside.len() + 1);
            let mut s = cs;
            for &(p, w) in &inside {
                v.push(Seg {
                    start: s,
                    end: cs + p,
                    after: Some(w),
                    cuts: Vec::new(),
                    clamp_after: false,
                    whole_cell: false,
                    split_in_cell: false,
                });
                s = cs + p;
            }
            v.push(Seg {
                start: s,
                end: ce,
                after: None,
                cuts: Vec::new(),
                clamp_after: false,
                whole_cell: false,
                split_in_cell: false,
            });
            v
        };
        let k = segs.len();
        for g in &mut segs {
            g.whole_cell = k == 1;
            g.split_in_cell = !merge && k > 1;
        }
        if let Some(tail) = segs.last_mut() {
            tail.after = at_end;
            tail.clamp_after = ci < last && !(rank0 && seq_start + ce == grid.first);
        }
        out.append(&mut segs);
    }
    out
}

/// 2026-10-15: Whether segment `seg`, from absolute position `abs_start`, runs the layer's decode
/// path (`forward_one`): a whole cell takes the path the grid serve's pass of the same rows takes
/// (`takes_decode_path`, the model's one-row pass), a segment of a split cell the path the grid
/// serve's segment takes (`split_segment_takes_decode_path`). Pure.
pub fn seg_takes_decode<W>(
    seg: &Seg<W>,
    abs_start: usize,
    tail_decode: bool,
    one_row_prefill: bool,
) -> bool {
    let rows = seg.end - seg.start;
    if seg.whole_cell {
        metrale_model_layers::layer::takes_decode_path(rows, abs_start, one_row_prefill, true)
    } else {
        metrale_model_layers::layer::split_segment_takes_decode_path(
            rows,
            abs_start,
            seg.split_in_cell,
            tail_decode,
            one_row_prefill,
        )
    }
}

/// 2026-10-15: The load-time check of the grid against the serve's prefill sizing, so every
/// chunk the scheduler forms on an idle cold prompt starts and ends on the grid: the first
/// chunk (`max_batch_tokens`, the idle budget) ends `step`-aligned past `first`, each later
/// chunk (`prefill_budget`) is a multiple of `step`, and no cell exceeds `prefill_rows`, so each
/// cell is one sub-chunk.
pub fn check_cell_budget(
    grid: CellGrid,
    prefill_rows: usize,
    prefill_budget: usize,
    max_batch_tokens: usize,
) -> Result<(), String> {
    let CellGrid { first, step } = grid;
    if first > prefill_rows || step > prefill_rows {
        return Err(format!(
            "{CELL_GRID_ENV}={first},{step}: a cell wider than METRALE_GLM_PREFILL_ROWS \
             ({prefill_rows}) would be re-split by the layer"
        ));
    }
    if max_batch_tokens <= first || !(max_batch_tokens - first).is_multiple_of(step) {
        return Err(format!(
            "{CELL_GRID_ENV}={first},{step}: the idle first chunk ({max_batch_tokens} tokens, \
             max_batch_tokens) must end on the grid past its first point"
        ));
    }
    if !prefill_budget.is_multiple_of(step) {
        return Err(format!(
            "{CELL_GRID_ENV}={first},{step}: the prefill budget ({prefill_budget}, \
             --max-prefill-tokens) must be a multiple of the step"
        ));
    }
    Ok(())
}

/// 2026-10-15: The load-time check of the model: a GLM-5.3 text stack (the only layers that
/// plan cells), whose model-wide clamp dims (`ssm_state_norm_dims`) are the KDA layer's own
/// `(heads, head_dim, head_dim)`, which the layer's per-cell clamp launches with.
pub fn check_cell_model(
    model_type: &str,
    norm_dims: (usize, usize, usize),
    kda_heads: usize,
    kda_head_dim: usize,
) -> Result<(), String> {
    if !matches!(model_type, "glm5_next" | "glm5_next_text") {
        return Err(format!(
            "{CELL_GRID_ENV}: only a GLM-5.3 serve plans cells, not model_type {model_type:?}"
        ));
    }
    let own = (kda_heads, kda_head_dim, kda_head_dim);
    if norm_dims != own {
        return Err(format!(
            "{CELL_GRID_ENV}: the model clamps states of dims {norm_dims:?}, the KDA layer's \
             are {own:?}"
        ));
    }
    Ok(())
}

/// 2026-10-15: The load-time check of the KV block size: every grid point on a block boundary,
/// so the scheduler's block-aligned chunk ends (`prefill_plan::plan_chunk_len`) stay on it.
pub fn check_cell_blocks(grid: CellGrid, block_size: Option<usize>) -> Result<(), String> {
    match block_size {
        Some(bs) if bs > 0 && grid.first.is_multiple_of(bs) && grid.step.is_multiple_of(bs) => {
            Ok(())
        }
        other => Err(format!(
            "{CELL_GRID_ENV}={},{}: every grid point must be a KV block boundary (block size \
             {other:?})",
            grid.first, grid.step
        )),
    }
}

/// 2026-10-15: The scheduler's chunking of an idle cold prompt: chunk 0 of `first` tokens
/// (`max_batch_tokens`), then `budget` each (`--max-prefill-tokens`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunking {
    pub first: usize,
    pub budget: usize,
}

static CHUNKING: OnceLock<Chunking> = OnceLock::new();

/// 2026-10-15: Record the serve's chunking, once at load after [`check_cell_budget`]; a second
/// call with other values is an error.
pub fn set_chunking(c: Chunking) -> Result<()> {
    let got = *CHUNKING.get_or_init(|| c);
    anyhow::ensure!(
        got == c,
        "{CELL_GRID_ENV}: chunking already set to {got:?}, not {c:?}"
    );
    Ok(())
}

/// 2026-10-15: The chunking [`set_chunking`] recorded; an error before it was set.
pub fn chunking() -> Result<Chunking> {
    CHUNKING.get().copied().ok_or_else(|| {
        anyhow::anyhow!("{CELL_GRID_ENV} is set but the load never recorded the chunking")
    })
}

/// 2026-10-15: How many grid points (or chunk ends `first + j * step`) are at or below `u`.
fn ends_through(first: usize, step: usize, u: usize) -> usize {
    if u < first { 0 } else { (u - first) / step + 1 }
}

/// 2026-10-15: The clamps a state restored at `u` takes before the first computed row, beyond
/// those the scheduler already ran: the grid serve's fully cached chunks (every grid point at
/// or below `u`, less its unclamped chunk 0 on rank 0) less this serve's fully cached chunks
/// (every chunk end at or below `u`, each clamped, rank 0's chunk 0 by
/// [`rank0_first_chunk_clamp`]). Errors when the chunking has more cached chunk ends than the
/// grid, which [`check_cell_budget`] rules out. Pure.
pub fn restored_clamps(
    grid: CellGrid,
    chunking: Chunking,
    rank0: bool,
    u: usize,
) -> Result<usize, String> {
    let grid_ends = ends_through(grid.first, grid.step, u) - usize::from(rank0 && grid.first <= u);
    let own_ends = ends_through(chunking.first, chunking.budget, u);
    grid_ends.checked_sub(own_ends).ok_or_else(|| {
        format!(
            "{CELL_GRID_ENV}: {own_ends} cached chunk ends at or below {u} under {chunking:?}, \
             more than the grid's {grid_ends}"
        )
    })
}

/// 2026-10-15: Whether rank 0 clamps at the end of its chunk 0 (`chunk0_end`): when the chunk
/// ends past the grid's first point, where the grid serve's chunk 1, which it clamps, ends.
/// Pure.
pub fn rank0_first_chunk_clamp(grid: CellGrid, chunk0_end: usize) -> bool {
    chunk0_end > grid.first
}

#[cfg(test)]
#[path = "glm5next_prefill_cell_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "glm5next_prefill_cell_clamp_tests.rs"]
mod clamp_tests;
