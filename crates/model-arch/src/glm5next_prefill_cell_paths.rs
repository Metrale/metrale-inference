// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: Which scheduler paths `METRALE_GLM_PREFILL_CELL_GRID` admits
//! (`glm5next_prefill_cell`). The cell walk, the grid floor and every grid clamp live on the
//! single-sequence chunk path (`prefill_chunk`); a path that runs a prompt elsewhere is either
//! refused at load ([`check_cell_paths`]) or carries only the prompts it computes as that path
//! does ([`batched_pass_is_inert`]).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Pure: no I/O, no environment reads.

use super::{CELL_GRID_ENV, CellGrid};

/// 2026-10-16: The scheduler levers that send a prompt past `prefill_chunk`'s grid floor and
/// cell walk, as the serve resolved them: the tail-checkpoint pre-cut (`ssm_tail_ckpt` without
/// `ssm_tail_midchunk`) ends a chunk off the grid, and the single-GPU batched prefills
/// (`prefill_codispatch`, `prefill_varlen`) give streams a chunk geometry of their own.
/// `METRALE_EP_PREFILL_BATCH` is not here: the scheduler keeps its batched pass to the prompts
/// of [`batched_pass_is_inert`] under the grid.
#[derive(Clone, Copy, Debug)]
pub struct CellPathLevers {
    pub ssm_tail_ckpt: bool,
    pub ssm_tail_midchunk: bool,
    pub prefill_codispatch: bool,
    pub prefill_varlen: bool,
}

/// 2026-10-16: The load-time refusal of a grid with any lever of [`CellPathLevers`] that routes
/// a prompt around the floored chunk path; the error names every such lever. Pure.
pub fn check_cell_paths(levers: CellPathLevers) -> Result<(), String> {
    let off_path = [
        (
            levers.ssm_tail_ckpt && !levers.ssm_tail_midchunk,
            "METRALE_SSM_TAIL_CKPT=1 without METRALE_SSM_TAIL_MIDCHUNK",
        ),
        (levers.prefill_codispatch, "METRALE_PREFILL_CODISPATCH"),
        (levers.prefill_varlen, "METRALE_PREFILL_VARLEN"),
    ];
    let set: Vec<&str> = off_path
        .iter()
        .filter(|(on, _)| *on)
        .map(|(_, n)| *n)
        .collect();
    if set.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{CELL_GRID_ENV}: {} route prompts around the grid-floored chunk path",
        set.join(", ")
    ))
}

/// 2026-10-16: Whether a batched prefill pass (`prefill_spans`, the mixed step) computes a
/// stream's chunk `[chunk_start, chunk_start + chunk_len)` of a `prompt_len`-token prompt as
/// the single-sequence chunk path does under `grid`. The batched pass runs no cell plan, no
/// restored-state or rank-0 first-chunk clamp and records no chunk end, so it may carry only a
/// chunk none of those touches: the whole prompt from 0, ending at or before the first grid
/// point. Such a chunk has no grid point inside it, rank 0 takes no first-chunk clamp
/// (`rank0_first_chunk_clamp`), a restore point inside it is below the first point, so every
/// rank's `restored_clamps` is 0, and no later chunk reads its record. Pure.
pub fn batched_pass_is_inert(
    grid: CellGrid,
    chunk_start: usize,
    chunk_len: usize,
    prompt_len: usize,
) -> bool {
    chunk_start == 0 && chunk_len == prompt_len && prompt_len <= grid.first
}

#[cfg(test)]
#[path = "glm5next_prefill_cell_paths_tests.rs"]
mod tests;
