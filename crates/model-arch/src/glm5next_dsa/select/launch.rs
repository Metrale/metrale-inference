// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `select_tokens`, the launcher for one DSA selection pass.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - 2026-10-11: `select_tokens_in_passes` launches the kernels `select_tokens` would for the
//!   same pass when the scratch holds all its rows; otherwise one `dsa_kpool_compress`, then per
//!   row sub-pass the scores, top-k and expansion of those rows, each row's inputs and output
//!   at the offsets the one-pass launch reads and writes.
//! - 2026-10-14: A ceiling launch's `dsa_kpool_compress` and `dsa_index_scores` grids stop at
//!   [`POOL_GRID_MAX`] pools (the kernels stride over the live pools), so a single-row decode's
//!   launches do not grow with `--max-seq-len`; an exact launch's grids are unchanged.
//! - 2026-10-17: A pass the fused selection takes (`fused::takes_fused`) launches the pool prep
//!   (when it compresses), `dsa_select_fused_pipe`, then the folded expansion unless the pass
//!   is folded; it writes no scores or candidacy row. Every other pass launches as before.

use super::subpass::{check_passes, row_passes, sub_pass_rows};
use super::*;
use anyhow::ensure;

/// 2026-09-25: Run the selection kernels, leaving `[q_rows, out_width]` token ids in
/// [`DsaSelectScratch::tokens`].
///
/// Fails before any launch when `scratch` is too small for `geom`, or when a ceiling launch
/// has no `geom_dev` or has `q_rows > 1`. The kernels are enqueued on `stream`; nothing here
/// synchronises.
#[allow(clippy::too_many_arguments)]
pub fn select_tokens(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    launch: DsaSelectLaunch,
    stream: u64,
) -> Result<()> {
    run(
        gpu, kernels, cfg, geom, inputs, scratch, launch, true, false, stream,
    )
}

/// 2026-10-11: [`select_tokens`] for an exact pass of `geom.q_rows` rows over a scratch whose
/// per-row regions may hold fewer rows ([`DsaSelectScratch::rows_per_pass`]): the pools are
/// compressed once, then the rows are scored, ranked and expanded in sub-passes of at most that
/// many rows. Each row's selection is the one-pass selection, bit for bit: every kernel after
/// the compression reads and writes only its own row (`r * H * D` of the query, `r * H` of the
/// head weights, `q_pos[r]`, `q_mask[r]`, the scores / candidacy row `r * P`, the selected row
/// `r * select_k`, the output row `r * out_width`), the compressed pools are a pure function
/// of the cache, and a sub-pass shifts every per-row pointer by its first row while the
/// temporaries start over at row 0. Returns the number of sub-passes. Errors when the output
/// cannot hold `geom.q_rows` rows or the pool regions are too small.
#[allow(clippy::too_many_arguments)]
pub fn select_tokens_in_passes(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    stream: u64,
) -> Result<usize> {
    let per = scratch.rows_per_pass(geom);
    if per >= geom.q_rows {
        run(
            gpu,
            kernels,
            cfg,
            geom,
            inputs,
            scratch,
            DsaSelectLaunch::Exact,
            true,
            false,
            stream,
        )?;
        return Ok(1);
    }
    select_tokens_rows(
        gpu,
        kernels,
        cfg,
        geom,
        inputs,
        scratch,
        &[(0, geom.q_rows)],
        true,
        stream,
    )
}

/// 2026-10-15: The rows `ranges` of an exact pass of `geom.q_rows` rows, in the sub-passes
/// [`row_passes`] plans at [`DsaSelectScratch::rows_per_pass`] ([`select_tokens_passes`]).
/// Returns the number of sub-passes.
#[allow(clippy::too_many_arguments)]
pub fn select_tokens_rows(
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
    let per = sub_pass_rows(scratch, geom)?;
    ensure!(
        ranges.iter().all(|&(lo, hi)| lo < hi && hi <= geom.q_rows),
        "DSA select: row ranges {ranges:?} outside a {}-row pass",
        geom.q_rows
    );
    let passes = row_passes(ranges, per)?;
    select_tokens_passes(
        gpu, kernels, cfg, geom, inputs, scratch, &passes, compress, stream,
    )
}

/// 2026-10-17 (was the body of `select_tokens_rows`, 2026-10-15): The row sub-passes `passes`
/// (`(first_row, rows)`, each at most [`DsaSelectScratch::rows_per_pass`] rows) of an exact
/// pass of `geom.q_rows` rows: each row's inputs and output at its offset, so each row's
/// selection is the one-pass selection's bits (see [`select_tokens_in_passes`]); rows outside
/// `passes` are not written. `compress` runs the pool compression on the first sub-pass; false
/// reads pools an earlier launch on `stream` compressed. Returns the number of sub-passes.
/// 2026-10-15: the sequence-parallel selection (`sp_select`) runs each rank's owned slabs
/// through it.
#[allow(clippy::too_many_arguments)]
pub fn select_tokens_passes(
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
    check_passes(scratch, geom, passes)?;
    let want_tokens = geom.q_rows * cfg.out_width() * 4;
    if want_tokens > scratch.tokens_bytes {
        bail!(
            "DSA select: selection output needs {want_tokens} B but only {} B was reserved",
            scratch.tokens_bytes
        );
    }
    for (i, &(r0, rows)) in passes.iter().enumerate() {
        let sub = DsaSelectGeometry {
            q_rows: rows,
            ..*geom
        };
        run(
            gpu,
            kernels,
            cfg,
            &sub,
            &inputs.from_row(r0, cfg),
            &scratch.row(r0, cfg),
            DsaSelectLaunch::Exact,
            compress && i == 0,
            false,
            stream,
        )?;
    }
    Ok(passes.len())
}

/// 2026-10-14: Grid x of `dsa_kpool_compress`: the full pool count of an exact pass, or under a
/// ceiling of `max_pools` the old one-block-per-pool grid `max_pools + 1` capped at
/// [`POOL_GRID_MAX`]. The kernel strides over pools up to the live count clamped to
/// `max_pools + 1` (`S / KP + 1` of the ceiling's `S`), so the capped grid writes the same
/// pools, each from one block running the same code.
pub(crate) fn kpool_compress_grid_x(geom: &DsaSelectGeometry, ceiling: Option<usize>) -> u32 {
    ceiling.map_or(geom.n_pools_full as u32, |m| pool_grid_x(m + 1))
}

/// 2026-10-14: Grid x of `dsa_index_scores`: the complete pool count of an exact pass, or the
/// ceiling's `max_pools` capped at [`POOL_GRID_MAX`]; the kernel strides as
/// [`kpool_compress_grid_x`] says, clamped to `P` = `max_pools`.
pub(crate) fn index_scores_grid_x(geom: &DsaSelectGeometry, ceiling: Option<usize>) -> u32 {
    ceiling.map_or(geom.n_pools as u32, pool_grid_x)
}

/// 2026-10-11: The body of [`select_tokens`]; `compress` false skips `dsa_kpool_compress`, for
/// a later row sub-pass over pools an earlier one on `stream` already compressed. 2026-10-15:
/// `fold` ends the pass with `dsa_sel_fold` on the selected pools instead of the expansion
/// (`folded`); it needs an exact launch with pools.
#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    launch: DsaSelectLaunch,
    compress: bool,
    fold: bool,
    stream: u64,
) -> Result<()> {
    // 2026-10-17: `METRALE_GLM_DSA_SELECT_FUSED`: decided first, as it sets which regions the
    // pass needs (`fused`).
    let fused = fused::takes_fused(kernels, geom, inputs, scratch, launch)?;
    scratch.fits_for(cfg, geom, fused)?;
    ensure!(
        !fold || (matches!(launch, DsaSelectLaunch::Exact) && geom.n_pools > 0),
        "DSA select: a folded pass needs an exact launch with pools ({} pools)",
        geom.n_pools
    );

    let d = geom.index_head_dim;
    let kp = geom.index_kpool;

    let ceiling = match launch {
        DsaSelectLaunch::Exact => None,
        DsaSelectLaunch::Ceiling { max_pools } => {
            if inputs.geom_dev.0 == 0 {
                bail!(
                    "DSA select: a ceiling launch has no host geometry to fall back on and \
                     needs `geom_dev`; passing NULL would run the frozen scalars."
                );
            }
            if geom.q_rows != 1 {
                bail!(
                    "DSA select: ceiling launch is decode-only (q_rows must be 1, got {}). \
                     At q_rows > 1 the row strides vary with the context and a replayed \
                     graph would index the wrong rows.",
                    geom.q_rows
                );
            }
            Some(max_pools)
        }
    };
    let gd = inputs.geom_dev;

    // 2026-09-25: Under a ceiling launch the kernels read S, the pool counts, the tile and
    // select_k from `geom_dev`, so the scalar arguments are unused. They are set from the
    // ceiling, so every capture at the same ceiling passes the same values.
    let (seq_a, npools_a, np2_a, selk_a) = match ceiling {
        Some(m) => (
            m * kp,
            m,
            m.next_power_of_two().max(2).min(topk_tile()),
            cfg.select_k(m),
        ),
        None => (geom.seq, geom.n_pools, geom.topk_np2, geom.select_k),
    };

    // 2026-09-25: Below `index_kpool` tokens there are no pools to score or sort, and a
    // zero-extent grid is not a valid launch, so `dsa_index_scores` and `dsa_topk_pools` are
    // skipped. `dsa_kpool_compress` still runs (`n_pools_full >= 1`), and
    // `dsa_expand_selection` writes the row to -1 and appends the visible tail, which is the
    // whole selection here. Under a ceiling launch both always run; with no live pools
    // `dsa_index_scores` blocks return at once and `dsa_topk_pools` selects nothing.
    let has_pools = geom.n_pools > 0 || ceiling.is_some();

    // 2026-09-25: Pool compression over the full pool count; the trailing partial pool is
    // written and marked invalid, and no later kernel reads it.
    // 2026-10-16: An FP8 cache's pool keys come from its pooled entries (`dsa_kpool_compress_fp8`).
    if compress && let Some(f) = inputs.rows.fp8 {
        ensure!(
            inputs.first_key == 0 && d == super::super::fp8_rows::FP8_INDEX_HEAD_DIM,
            "DSA select: the FP8 indexer cache needs first_key 0 and D 128 (first_key {}, D {d})",
            inputs.first_key
        );
        KernelLaunch::new(gpu, kernels.fp8.compress)
            .grid([kpool_compress_grid_x(geom, ceiling), 1, 1])
            .block([d as u32, 1, 1])
            .arg_ptr(inputs.rows.k)
            .arg_ptr(scratch.pool_keys)
            .arg_ptr(scratch.pool_indices)
            .arg_ptr(scratch.pool_valid)
            .arg_u32(seq_a as u32)
            .arg_u32(kp as u32)
            .arg_u32(f.tail_sets)
            .arg_ptr(gd)
            .arg_ptr(inputs.rows.bt)
            .arg_u32(inputs.rows.block_size)
            .arg_u32(f.blk_bytes)
            .launch(stream)?;
    } else if compress {
        KernelLaunch::new(gpu, kernels.kpool_compress)
            .grid([kpool_compress_grid_x(geom, ceiling), 1, 1])
            .block([d.min(1024) as u32, 1, 1])
            .arg_ptr(inputs.rows.k)
            .arg_ptr(inputs.rows.gate)
            .arg_ptr(inputs.rows.valid)
            .arg_ptr(inputs.ape)
            .arg_ptr(scratch.pool_keys)
            .arg_ptr(scratch.pool_indices)
            .arg_ptr(scratch.pool_valid)
            .arg_u32(seq_a as u32)
            .arg_u32(d as u32)
            .arg_u32(kp as u32)
            .arg_i32(inputs.first_key)
            .arg_ptr(gd)
            // 2026-10-09: The paged address path; NULL / 0 / 0 for a flat cache.
            .arg_ptr(inputs.rows.bt)
            .arg_u32(inputs.rows.block_size)
            .arg_u32(inputs.rows.blk_elems)
            .launch(stream)?;
    }

    // 2026-10-10: `METRALE_GLM_DSA_SCORES_TILED=1`: the same scores, bit for bit, in query x
    // pool tiles for an exact launch of several query rows (`scores_tiled_for`).
    // 2026-10-12: `METRALE_GLM_DSA_SCORES_MMA=1`: the scores on BF16 tensor cores, ahead of
    // the tiled path (`scores_mma_for`).
    let mma = scores_mma_for(
        ceiling.is_some(),
        geom,
        kernels.rows.scores_mma,
        scratch.levers.scores_mma,
    );
    // 2026-10-14: `METRALE_GLM_DSA_SCORES_PIPE=1`: those scores streamed (`pipe`), from BF16
    // keys and visibility prepared once per compression.
    let pipe = scores_pipe_for(mma, &kernels.rows, scratch.levers.scores_pipe);
    if ceiling.is_none() && geom.q_rows > 1 {
        pipe::readback_select_levers(mma, pipe, fused, scratch.levers.topk_radix);
    }
    // 2026-10-17: The fused pass: the pipe's prepared pools, then scores, top-k and fold in one
    // kernel; `fused` implies an exact launch with pools.
    if fused {
        ensure!(
            pipe,
            "DSA select: the fused selection needs the streamed scores"
        );
        if compress {
            pipe::launch_prep(gpu, kernels, geom, inputs, scratch, npools_a, seq_a, stream)?;
        }
        fused::launch(gpu, kernels, geom, inputs, scratch, stream)?;
        if fold {
            return Ok(());
        }
        return folded::launch_expand(gpu, kernels, cfg, geom, inputs, scratch, stream);
    }
    let tiled = !mma
        && scores_tiled_for(
            ceiling.is_some(),
            geom,
            kernels.rows.scores_tiled,
            scratch.levers.scores_tiled,
        );
    if has_pools && pipe {
        if compress {
            pipe::launch_prep(gpu, kernels, geom, inputs, scratch, npools_a, seq_a, stream)?;
        }
        pipe::launch_scores(gpu, kernels, geom, inputs, scratch, npools_a, stream)?;
    }
    if has_pools && mma && !pipe {
        KernelLaunch::new(gpu, kernels.rows.scores_mma)
            .grid([geom.q_rows.div_ceil(SCORES_MMA_ROWS) as u32, 1, 1])
            .block([256, 1, 1])
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
            .arg_f32((d as f32).powf(-0.5))
            .launch(stream)?;
    }
    if has_pools && tiled {
        KernelLaunch::new(gpu, kernels.rows.scores_tiled)
            .grid([
                geom.n_pools.div_ceil(SCORES_TILE_POOLS) as u32,
                geom.q_rows.div_ceil(SCORES_TILE_ROWS) as u32,
                1,
            ])
            .block([256, 1, 1])
            .shared_mem(SCORES_TILE_SMEM)
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
            .arg_f32((d as f32).powf(-0.5))
            .launch(stream)?;
    }
    if has_pools {
        if !tiled && !mma {
            KernelLaunch::new(gpu, kernels.index_scores)
                .grid([index_scores_grid_x(geom, ceiling), geom.q_rows as u32, 1])
                .block([SCORES_BLOCK, 1, 1])
                // 2026-09-25: `dsa_index_scores` keeps one f32 per index head in shared memory and
                // sums them in head order.
                .shared_mem(SCORES_BLOCK.max((geom.index_heads * 4) as u32))
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
                .arg_f32((d as f32).powf(-0.5))
                .arg_ptr(gd)
                .launch(stream)?;
        }

        // 2026-10-12: `METRALE_GLM_DSA_TOPK_RADIX=1`: the same selection by radix select.
        // 2026-10-21: a ceiling launch by `METRALE_GLM_DSA_DECODE_TOPK_RADIX` (default on), over
        // the live pools in `gd`.
        if topk_radix_for(
            ceiling.is_some(),
            selk_a,
            kernels.rows.topk_radix,
            scratch.levers.topk_radix,
            scratch.levers.decode_topk_radix,
        ) {
            KernelLaunch::new(gpu, kernels.rows.topk_radix)
                .grid([geom.q_rows as u32, 1, 1])
                .block([ROW_BLOCK, 1, 1])
                .arg_ptr(scratch.scores)
                .arg_ptr(scratch.selected)
                .arg_u32(geom.q_rows as u32)
                .arg_u32(npools_a as u32)
                .arg_u32(selk_a as u32)
                .arg_ptr(gd)
                .launch(stream)?;
        } else {
            // 2026-09-25: `np2_a` is at most `topk_tile()`, so the request is at most
            // `topk_smem_for_tile(topk_tile())`, within `TOPK_SMEM_CEILING`.
            KernelLaunch::new(gpu, kernels.topk_pools)
                .grid([geom.q_rows as u32, 1, 1])
                .block([ROW_BLOCK, 1, 1])
                .shared_mem(topk_smem_for_tile(np2_a) as u32)
                .arg_ptr(scratch.scores)
                .arg_ptr(scratch.selected)
                .arg_u32(geom.q_rows as u32)
                .arg_u32(npools_a as u32)
                .arg_u32(np2_a as u32)
                .arg_u32(selk_a as u32)
                .arg_ptr(gd)
                .launch(stream)?;
        }
    }

    if fold {
        return folded::launch_fold(gpu, kernels, geom, inputs, scratch, stream);
    }

    KernelLaunch::new(gpu, kernels.expand_selection)
        .grid([geom.q_rows as u32, 1, 1])
        .block([ROW_BLOCK, 1, 1])
        .arg_ptr(scratch.selected)
        .arg_ptr(scratch.pool_indices)
        .arg_ptr(scratch.valid_cand)
        .arg_ptr(inputs.rows.valid)
        .arg_ptr(inputs.q_pos)
        .arg_ptr(inputs.q_mask)
        .arg_ptr(scratch.tokens)
        .arg_u32(geom.q_rows as u32)
        .arg_u32(npools_a as u32)
        .arg_u32(kp as u32)
        .arg_u32(seq_a as u32)
        .arg_u32(selk_a as u32)
        .arg_u32(geom.out_width as u32)
        .arg_i32(inputs.first_key)
        .arg_i32(cfg.always_select_tail as i32)
        .arg_ptr(gd)
        .launch(stream)?;

    Ok(())
}
