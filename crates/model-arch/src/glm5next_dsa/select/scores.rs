// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The per-block and warp-per-pool indexer scores of a selection pass, and the
//! persisted-pool-key pass of a captured decode row (`IndexerRowFormat::Bf16Pooled`).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - [`launch_scores`] writes the bytes `dsa_index_scores` writes for the pass, whichever
//!   kernel it launches (`dsa_index_scores_warp` byte-gated against it; the persisted keys are
//!   the bytes `dsa_kpool_compress` writes, so `dsa_index_scores_warp_paged`'s are too).
//! - A persisted pass ([`persisted_for`]) launches no `dsa_kpool_compress` and expands through
//!   the geometry (`dsa_expand_selection_rows`, slots recomputed), so nothing reads the
//!   scratch's `pool_keys`, `pool_indices` or `pool_valid`.

use super::*;

/// 2026-10-21: Whether the warp-per-pool kernels take this shape (`DSA_SW_HMAX`,
/// `DSA_SW_DMAX`, whole warps of channels).
fn warp_shape(geom: &DsaSelectGeometry) -> bool {
    geom.index_heads <= 32 && geom.index_head_dim <= 128 && geom.index_head_dim.is_multiple_of(32)
}

/// 2026-10-21: Whether a pass scores the persisted pool keys: a ceiling launch (a captured
/// decode row) over a paged cache that keeps them, pooled from the first key, with the warp
/// shape and the paged scores and geometry expansion resolved.
pub(super) fn persisted_for(
    kernels: &Glm5NextDsaKernels,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    ceiling: bool,
) -> bool {
    ceiling
        && inputs.rows.pool_keys.0 != 0
        && inputs.first_key == 0
        && warp_shape(geom)
        && kernels.rows.scores_warp_paged.0 != 0
        && kernels.rows.expand_selection.0 != 0
}

/// 2026-10-21: The scores of a pass that takes neither the tiled nor the tensor-core kernels:
/// the persisted keys (`persisted`), else `dsa_index_scores_warp` under
/// `METRALE_GLM_DSA_SCORES_WARP` when the shape fits, else `dsa_index_scores`. `npools_a`,
/// `seq_a` and `gd` are the pass's launch arguments (`launch::run`).
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_scores(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    ceiling: Option<usize>,
    persisted: bool,
    (npools_a, seq_a): (usize, usize),
    stream: u64,
) -> Result<()> {
    let (d, kp, gd) = (geom.index_head_dim, geom.index_kpool, inputs.geom_dev);
    let scale = (d as f32).powf(-0.5);
    if persisted {
        return KernelLaunch::new(gpu, kernels.rows.scores_warp_paged)
            .grid([scores_warp_grid_x(npools_a), geom.q_rows as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(inputs.q)
            .arg_ptr(inputs.rows.pool_keys)
            .arg_ptr(inputs.weights)
            .arg_ptr(inputs.rows.bt)
            .arg_ptr(inputs.q_pos)
            .arg_ptr(scratch.scores)
            .arg_ptr(scratch.valid_cand)
            .arg_u32(geom.q_rows as u32)
            .arg_u32(npools_a as u32)
            .arg_u32(geom.index_heads as u32)
            .arg_u32(d as u32)
            .arg_u32(kp as u32)
            .arg_u32(seq_a as u32)
            .arg_u32(inputs.rows.block_size)
            .arg_u32(inputs.rows.pk_blk)
            .arg_f32(scale)
            .arg_ptr(gd)
            .launch(stream);
    }
    // 2026-10-21: `METRALE_GLM_DSA_SCORES_WARP` (default on): the same scores, one warp per
    // pool, when the shape fits the kernel.
    let warp = scratch.levers.scores_warp && kernels.rows.scores_warp.0 != 0 && warp_shape(geom);
    let (k, grid, block, smem) = if warp {
        (
            kernels.rows.scores_warp,
            [scores_warp_grid_x(npools_a), geom.q_rows as u32, 1],
            256,
            0,
        )
    } else {
        (
            kernels.index_scores,
            [
                launch::index_scores_grid_x(geom, ceiling),
                geom.q_rows as u32,
                1,
            ],
            SCORES_BLOCK,
            // 2026-09-25: `dsa_index_scores` keeps one f32 per index head in shared memory
            // and sums them in head order.
            SCORES_BLOCK.max((geom.index_heads * 4) as u32),
        )
    };
    KernelLaunch::new(gpu, k)
        .grid(grid)
        .block([block, 1, 1])
        .shared_mem(smem)
        .arg_ptr(inputs.q)
        .arg_ptr(scratch.pool_keys)
        .arg_ptr(inputs.weights)
        .arg_ptr(scratch.pool_indices)
        .arg_ptr(scratch.pool_valid)
        .arg_ptr(inputs.rows.valid)
        .arg_ptr(inputs.q_pos)
        .arg_ptr(scratch.scores)
        .arg_ptr(scratch.valid_cand)
        .arg_u32(geom.q_rows as u32)
        .arg_u32(npools_a as u32)
        .arg_u32(geom.index_heads as u32)
        .arg_u32(d as u32)
        .arg_u32(kp as u32)
        .arg_u32(seq_a as u32)
        .arg_f32(scale)
        .arg_ptr(gd)
        .launch(stream)
}

/// 2026-10-21: A persisted pass's expansion: `dsa_expand_selection_rows` for its one row,
/// which recomputes the pools' token slots and S from the device geometry instead of reading
/// `pool_indices` (the same tokens `dsa_expand_selection` writes on a paged cache).
pub(super) fn launch_expand_geom(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernels.rows.expand_selection)
        .grid([1, 1, 1])
        .block([ROW_BLOCK, 1, 1])
        .arg_ptr(scratch.selected)
        .arg_ptr(scratch.valid_cand)
        .arg_ptr(inputs.q_pos)
        .arg_ptr(inputs.q_mask)
        .arg_ptr(scratch.tokens)
        .arg_u32(geom.index_kpool as u32)
        .arg_u32(geom.out_width as u32)
        .arg_i32(inputs.first_key)
        .arg_i32(cfg.always_select_tail as i32)
        .arg_ptr(inputs.geom_dev)
        .arg_u32(0)
        .arg_u32(0)
        .launch(stream)
}
