// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `METRALE_GLM_DSA_SELECT_FUSED`: the streamed tensor-core scores
//! (`METRALE_GLM_DSA_SCORES_PIPE`), the radix top-k and the selected-pools fold in one kernel,
//! `dsa_select_fused_pipe` (`dsa_index_scores_pipe.cu`, `dsa_topk_stream.cuh`). Each warp keeps
//! its row's running top `select_k` in shared memory while the row is scored (in-warp
//! compaction, no cross-warp merge), so the `[rows, pools]` scores and candidacy are neither
//! written nor read back: at a long context those rows dominate the selection's memory traffic
//! and, through the scratch they need, cap a pass at a few hundred rows (a grid under one wave).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - A fused pass leaves in the selected region what `dsa_index_scores_mma_pipe` +
//!   `dsa_topk_pools_radix` + `dsa_sel_fold` leave, slot for slot, for NaN-free scores; an
//!   unfolded pass then expands with `dsa_expand_selection_folded`, whose token rows are
//!   `dsa_expand_selection`'s over the unfolded rows (`dsa_sel_fold.cu`). So its token rows are
//!   the streamed scores' bits: no numerics change against `METRALE_GLM_DSA_SCORES_PIPE`.
//! - It runs only where the scratch was allocated with it and the pass has its shape
//!   ([`fused_shape`]); there a missing kernel or a device geometry fails the pass, never falls
//!   back to a path the scratch was not sized for.

use super::*;
use anyhow::ensure;

/// 2026-10-17: The widest selection the kernel keeps (`DSA_STK_MAXK`), the radix select's.
pub const SELECT_FUSED_MAXK: usize = TOPK_RADIX_MAX;

/// 2026-10-17: Entries of one warp's buffer (`DSA_STK_CAP`).
const SELECT_FUSED_CAP: usize = SELECT_FUSED_MAXK + 192;

/// 2026-10-17: Dynamic shared memory of `dsa_select_fused_pipe`: the pipe's ring, then per
/// warp `SELECT_FUSED_CAP` u32 keys and i32 pools (98,048 B).
pub const SELECT_FUSED_SMEM: u32 =
    SCORES_PIPE_SMEM + (SCORES_MMA_ROWS * SELECT_FUSED_CAP * 8) as u32;

/// 2026-10-17: The passes the fused kernel takes: the tensor-core scores' shape
/// ([`scores_mma_shape`]), pools to score, and 1 ..= [`SELECT_FUSED_MAXK`] selected.
pub fn fused_shape(geom: &DsaSelectGeometry) -> bool {
    scores_mma_shape(geom) && geom.n_pools > 0 && (1..=SELECT_FUSED_MAXK).contains(&geom.select_k)
}

/// 2026-10-17: Whether this pass runs fused: an exact launch the scratch takes fused
/// ([`DsaSelectScratch::fused_on`]). Errors there when a kernel it needs is not in the target or
/// the launch carries a device geometry (the kernel reads host scalars only).
pub(super) fn takes_fused(
    kernels: &Glm5NextDsaKernels,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    launch: DsaSelectLaunch,
) -> Result<bool> {
    if !(matches!(launch, DsaSelectLaunch::Exact) && scratch.fused_on(geom)) {
        return Ok(false);
    }
    let k = &kernels.rows;
    ensure!(
        k.select_fused.0 != 0 && k.pool_prep.0 != 0 && k.expand_folded.0 != 0,
        "DSA select: METRALE_GLM_DSA_SELECT_FUSED is set but dsa_select_fused_pipe, \
         dsa_pool_prep_bf16 or dsa_expand_selection_folded is not in this target"
    );
    ensure!(
        inputs.geom_dev.0 == 0,
        "DSA select: the fused selection reads host geometry, but the pass carries geom_dev"
    );
    Ok(true)
}

/// 2026-10-17: `dsa_select_fused_pipe` over the pass's rows: their folded selected pools into
/// `scratch.selected` (the pass's first row), from the regions `pipe::launch_prep` wrote.
pub(super) fn launch(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    stream: u64,
) -> Result<()> {
    scratch.pipe.check(geom.n_pools)?;
    let d = geom.index_head_dim;
    KernelLaunch::new(gpu, kernels.rows.select_fused)
        .grid([geom.q_rows.div_ceil(SCORES_MMA_ROWS) as u32, 1, 1])
        .block([256, 1, 1])
        .shared_mem(SELECT_FUSED_SMEM)
        .arg_ptr(inputs.q)
        .arg_ptr(scratch.pipe.keys_bf16)
        .arg_ptr(inputs.weights)
        .arg_ptr(scratch.pipe.vis)
        .arg_ptr(inputs.q_pos)
        .arg_ptr(inputs.q_mask)
        .arg_ptr(scratch.selected)
        .arg_u32(geom.q_rows as u32)
        .arg_u32(geom.n_pools as u32)
        .arg_u32(geom.index_heads as u32)
        .arg_u32(d as u32)
        .arg_u32(geom.index_kpool as u32)
        .arg_u32(geom.select_k as u32)
        .arg_f32((d as f32).powf(-0.5))
        .launch(stream)?;
    Ok(())
}
