// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The selected-pools payload of the sequence-parallel DSA selection
//! (`sp_select`, `METRALE_GLM_DSA_SP_SELECT_SEL`): a split rank runs its rows' scores and top-k,
//! folds their candidacy into the selected pools (`dsa_sel_fold`) and leaves them in the
//! `[q_rows, select_k]` selected region at their own rows; after the ranks gather those rows,
//! every rank expands all of them (`dsa_expand_selection_folded`, `dsa_sel_fold.cu`).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - [`select_rows_folded`] runs the sub-passes `select_tokens_rows` runs for the same `ranges`
//!   (the same scores and top-k launches, each row's inputs at its offset), except that each
//!   sub-pass writes its selected pools at their own rows instead of region row 0 and ends with
//!   the fold instead of the expansion. Its rows' selected pools are the unsplit pass's, folded.
//!   2026-10-17: likewise [`select_passes_folded`] and `select_tokens_passes` for the same
//!   sub-passes.
//! - [`expand_folded`] over the folded rows of a whole pass writes the token rows
//!   `select_tokens_in_passes` writes (`dsa_sel_fold.cu`'s invariants); it reads no candidacy
//!   row, so it needs only the gathered selected pools and the compressed pools.
//! - Both are for an exact host-geometry pass (`geom_dev` NULL), the batched prefill's.

use super::subpass::{check_passes, row_passes};
use super::*;
use anyhow::ensure;

impl DsaSelectScratch {
    /// 2026-10-15: The `[out_rows, select_k]` i32 selected-pools region (the stride is the
    /// pass's `select_k`).
    pub fn selected(&self) -> DevicePtr {
        self.selected
    }

    /// 2026-10-15: [`Self::row`] with the selected-pools region also at row `row` of a
    /// `select_k`-wide pass, so a sub-pass keeps its selected pools at their own rows.
    fn row_sel(&self, row: usize, cfg: &Glm5NextDsaConfig, select_k: usize) -> Self {
        let mut s = self.row(row, cfg);
        s.selected = self.selected.offset(row * select_k * 4);
        s.capacity[5] = self.capacity[5] - row * select_k * 4;
        s
    }
}

/// 2026-10-15: The rows `ranges` of an exact pass, scored and ranked in the sub-passes
/// `select_tokens_rows` plans ([`select_passes_folded`]). Returns the number of sub-passes.
#[allow(clippy::too_many_arguments)]
pub fn select_rows_folded(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    ranges: &[(usize, usize)],
    compress: bool,
    stream: u64,
) -> Result<usize> {
    folded_fits(geom, inputs, scratch)?;
    let per = scratch.rows_per_pass(geom);
    ensure!(
        ranges.iter().all(|&(lo, hi)| lo < hi && hi <= geom.q_rows),
        "DSA select: row ranges {ranges:?} outside a {}-row pass",
        geom.q_rows
    );
    let passes = row_passes(ranges, per)?;
    select_passes_folded(
        gpu, kernels, cfg, geom, inputs, scratch, &passes, compress, stream,
    )
}

/// 2026-10-17 (was the body of `select_rows_folded`, 2026-10-15): The row sub-passes `passes`
/// of an exact pass (as `select_tokens_passes` takes them), scored and ranked, their selected
/// pools folded and left at their own rows of [`DsaSelectScratch::selected`]; no token row is
/// written. `compress` as in `select_tokens_passes`. Returns the number of sub-passes. Errors
/// before any launch when the selected region cannot hold the pass's rows, the pass has no
/// pools, `geom_dev` is set, or a sub-pass is empty, wider than the scratch or outside the pass.
#[allow(clippy::too_many_arguments)]
pub fn select_passes_folded(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    passes: &[(usize, usize)],
    compress: bool,
    stream: u64,
) -> Result<usize> {
    folded_fits(geom, inputs, scratch)?;
    check_passes(scratch, geom, passes)?;
    for (i, &(r0, rows)) in passes.iter().enumerate() {
        let sub = DsaSelectGeometry {
            q_rows: rows,
            ..*geom
        };
        launch::run(
            gpu,
            kernels,
            cfg,
            &sub,
            &inputs.from_row(r0, cfg),
            &scratch.row_sel(r0, cfg, geom.select_k),
            DsaSelectLaunch::Exact,
            compress && i == 0,
            true,
            stream,
        )?;
    }
    Ok(passes.len())
}

/// 2026-10-15: The token rows of a whole exact pass from its folded selected pools
/// ([`select_rows_folded`] on every row, here or on the rank that gathered them) and the
/// compressed pools, into [`DsaSelectScratch::tokens`].
pub fn expand_folded(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    stream: u64,
) -> Result<()> {
    folded_fits(geom, inputs, scratch)?;
    scratch.fits(cfg, &DsaSelectGeometry { q_rows: 1, ..*geom })?;
    ensure!(
        geom.q_rows * cfg.out_width() * 4 <= scratch.tokens_bytes,
        "DSA select: a folded expansion of {} rows exceeds the output",
        geom.q_rows
    );
    launch_expand(gpu, kernels, cfg, geom, inputs, scratch, stream)
}

/// 2026-10-17: The `dsa_expand_selection_folded` launch of [`expand_folded`], unchecked: also the
/// expansion of an unfolded fused pass (`fused`), whose selected rows are folded.
pub(super) fn launch_expand(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernels.rows.expand_folded)
        .grid([geom.q_rows as u32, 1, 1])
        .block([ROW_BLOCK, 1, 1])
        .arg_ptr(scratch.selected)
        .arg_ptr(scratch.pool_indices)
        .arg_ptr(inputs.rows.valid)
        .arg_ptr(inputs.q_pos)
        .arg_ptr(inputs.q_mask)
        .arg_ptr(scratch.tokens)
        .arg_u32(geom.q_rows as u32)
        .arg_u32(geom.n_pools as u32)
        .arg_u32(geom.index_kpool as u32)
        .arg_u32(geom.seq as u32)
        .arg_u32(geom.select_k as u32)
        .arg_u32(geom.out_width as u32)
        .arg_i32(inputs.first_key)
        .arg_i32(cfg.always_select_tail as i32)
        .launch(stream)
}

/// 2026-10-15: `dsa_sel_fold` over the `geom.q_rows` rows of one sub-pass, after its top-k
/// (`launch::run` with `fold`): `scratch.selected` is the sub-pass's first row, the candidacy
/// at its region row 0. Public for the gate's timing of the launch alone.
pub fn launch_fold(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernels.rows.sel_fold)
        .grid([geom.q_rows as u32, 1, 1])
        .block([ROW_BLOCK, 1, 1])
        .arg_ptr(scratch.selected)
        .arg_ptr(scratch.valid_cand)
        .arg_ptr(inputs.q_pos)
        .arg_ptr(inputs.q_mask)
        .arg_u32(geom.n_pools as u32)
        .arg_u32(geom.index_kpool as u32)
        .arg_u32(geom.select_k as u32)
        .launch(stream)
}

/// 2026-10-15: The folded payload's preconditions: an exact host-geometry pass with pools, its
/// kernels resolved, and every row's selected pools in the selected region at once.
fn folded_fits(
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
) -> Result<()> {
    ensure!(
        inputs.geom_dev.0 == 0 && geom.n_pools > 0,
        "DSA select: the folded payload needs host geometry and pools ({} pools)",
        geom.n_pools
    );
    ensure!(
        geom.q_rows * geom.select_k * 4 <= scratch.capacity[5],
        "DSA select: {} rows of {} selected pools exceed the selected region ({} B)",
        geom.q_rows,
        geom.select_k,
        scratch.capacity[5]
    );
    Ok(())
}

/// 2026-10-15: Whether a pass's folded payload can run: [`folded_fits`] and both kernels
/// resolved; the error names what is missing.
pub fn folded_check(
    kernels: &Glm5NextDsaKernels,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
) -> Result<()> {
    ensure!(
        kernels.rows.sel_fold.0 != 0 && kernels.rows.expand_folded.0 != 0,
        "DSA select: the folded payload's kernels (dsa_sel_fold.cu) are not in this target"
    );
    folded_fits(geom, inputs, scratch)
}
