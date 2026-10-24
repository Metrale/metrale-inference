// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The per-block and warp-per-pool indexer scores of a selection pass, and the
//! persisted-pool-key pass of a captured decode row (`IndexerRowFormat::Bf16Pooled`).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - [`launch_scores`] writes the bytes `dsa_index_scores` writes for the pass, whichever
//!   kernel it launches (`dsa_index_scores_warp` byte-gated against it; the persisted keys are
//!   the bytes `dsa_kpool_compress` writes, so `dsa_index_scores_warp_paged`'s are too).
//! - A ceiling pass of several rows ([`rows_ceiling_for`]) reads row `r`'s geometry at slots
//!   `r * DSA_GEOM_SLOTS` and indexes rows at the ceiling's strides in every kernel it launches.
//! - A persisted pass ([`persisted_for`]) launches no `dsa_kpool_compress` and expands through
//!   the geometry (`dsa_expand_selection_rows`, slots recomputed), so nothing reads the
//!   scratch's `pool_keys`, `pool_indices` or `pool_valid`.

use super::*;
use metrale_gpu_runtime::gpu::KernelHandle;

/// 2026-10-21: Whether the warp-per-pool kernels take this shape (`DSA_SW_HMAX`,
/// `DSA_SW_DMAX`, whole warps of channels).
fn warp_shape(geom: &DsaSelectGeometry) -> bool {
    geom.index_heads <= 32 && geom.index_head_dim <= 128 && geom.index_head_dim.is_multiple_of(32)
}

/// 2026-10-21: `DSA_SH_D` and the most rows of `dsa_index_scores_heads_paged` (`DSA_SW_WARPS`).
pub(crate) const SCORES_HEADS_D: usize = 128;
pub(crate) const SCORES_HEADS_MAX_ROWS: usize = 8;

/// 2026-10-22: The fewest rows `dsa_index_scores_heads_window` takes: a verify's. Its walk
/// saves the head math's bookkeeping, which a 2-row pass is bound by (gate
/// examples/glm5next_dsa_ceiling_scores_gate.rs, 32,768 pools: 92 -> 82 us); a 1-row pass is
/// bound by the key reads (72 us either way, the window walk sometimes 80 us in a replayed graph).
pub(crate) const SCORES_WINDOW_MIN_ROWS: usize = 2;

/// 2026-10-21: Whether a persisted pass's scores run as `dsa_index_scores_heads_paged`: the
/// lever on, the kernel and the SM count resolved, its shape (at most 32 heads of exactly 128,
/// at most 8 rows), and the 16-byte alignment its key copies need (the key region's base and
/// its block stride in f32).
pub(crate) fn heads_for(
    kernels: &Glm5NextDsaKernels,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
) -> bool {
    scratch.levers.scores_heads
        && kernels.rows.scores_heads_paged.0 != 0
        && kernels.rows.sm_count > 0
        && geom.index_heads <= 32
        && geom.index_head_dim == SCORES_HEADS_D
        && geom.q_rows <= SCORES_HEADS_MAX_ROWS
        && inputs.rows.pool_keys.0.is_multiple_of(16)
        && inputs.rows.pk_blk.is_multiple_of(4)
}

/// 2026-10-22: The lane-per-head kernel a persisted pass's scores run as ([`heads_for`]), or
/// `None` for `dsa_index_scores_warp_paged`: `dsa_index_scores_heads_window` for a pass of at
/// least [`SCORES_WINDOW_MIN_ROWS`] rows under `METRALE_GLM_DSA_SCORES_HEADS_WINDOW` when it
/// resolved, else `dsa_index_scores_heads_paged` (the same bytes, arguments and grid).
pub(crate) fn heads_kernel(
    kernels: &Glm5NextDsaKernels,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
) -> Option<KernelHandle> {
    let window = kernels.rows.scores_heads_window;
    heads_for(kernels, geom, inputs, scratch).then_some(
        if scratch.levers.scores_heads_window
            && window.0 != 0
            && geom.q_rows >= SCORES_WINDOW_MIN_ROWS
        {
            window
        } else {
            kernels.rows.scores_heads_paged
        },
    )
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

/// 2026-10-21: Whether a ceiling launch of `geom.q_rows` rows (a captured verify's rows of one
/// sequence, each with its geometry at slots `row * DSA_GEOM_SLOTS` of `inputs.geom_dev`) can
/// run as one pass: it scores the persisted keys and selects by radix, the kernels that read
/// each row's own geometry and index rows at the ceiling's strides, and the scratch holds that
/// many rows at `max_pools` pools.
pub(crate) fn rows_ceiling_for(
    kernels: &Glm5NextDsaKernels,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    max_pools: usize,
) -> bool {
    let select_k = cfg.select_k(max_pools);
    persisted_for(kernels, geom, inputs, true)
        && topk_radix_for(
            true,
            select_k,
            kernels.rows.topk_radix,
            scratch.levers.topk_radix,
            scratch.levers.decode_topk_radix,
        )
        && scratch
            .rows_regions(cfg, geom.q_rows, max_pools, select_k)
            .is_some()
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
        // 2026-10-21: `METRALE_GLM_DSA_SCORES_HEADS` (default on): the same bytes, one lane per
        // index head, one block per SM with every row in it.
        let (k, grid) = match heads_kernel(kernels, geom, inputs, scratch) {
            Some(k) => (k, [kernels.rows.sm_count, 1, 1]),
            None if kernels.rows.pool_keys_bf16 => bail!(
                "METRALE_GLM_DSA_POOL_KEYS_BF16: a persisted pass of {} rows the lane-per-head \
                 kernels do not take (heads {}, head dim {}); dsa_index_scores_warp_paged reads \
                 FP32 keys",
                geom.q_rows,
                geom.index_heads,
                d
            ),
            None => (
                kernels.rows.scores_warp_paged,
                [scores_warp_grid_x(npools_a), geom.q_rows as u32, 1],
            ),
        };
        return launch_persisted(
            gpu,
            k,
            grid,
            geom,
            inputs,
            scratch,
            (npools_a, seq_a),
            stream,
        );
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

/// 2026-10-23: The persisted keys' scores on `k` over `grid` (`launch_scores`'s pick, or a plan's
/// lane-per-head kernel over one block per SM).
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_persisted(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    grid: [u32; 3],
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    (npools_a, seq_a): (usize, usize),
    stream: u64,
) -> Result<()> {
    let d = geom.index_head_dim;
    KernelLaunch::new(gpu, k)
        .grid(grid)
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
        .arg_u32(geom.index_kpool as u32)
        .arg_u32(seq_a as u32)
        .arg_u32(inputs.rows.block_size)
        .arg_u32(inputs.rows.pk_blk)
        .arg_f32((d as f32).powf(-0.5))
        .arg_ptr(inputs.geom_dev)
        .launch(stream)
}

/// 2026-10-21: A persisted pass's expansion: `dsa_expand_selection_rows` for its rows, which
/// recomputes the pools' token slots and S from each row's device geometry instead of reading
/// `pool_indices` (the same tokens `dsa_expand_selection` writes on a paged cache). `strides`
/// are the scores' and selected pools' row strides, the pass's `(npools_a, selk_a)`.
/// 2026-10-23: `expand` is `rows.expand_selection` (the layer's, or a plan's).
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_expand_geom(
    gpu: &dyn GpuBackend,
    expand: KernelHandle,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    (sc_stride, sel_stride): (usize, usize),
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, expand)
        .grid([1, geom.q_rows as u32, 1])
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
        .arg_u32(sc_stride as u32)
        .arg_u32(sel_stride as u32)
        .launch(stream)
}
