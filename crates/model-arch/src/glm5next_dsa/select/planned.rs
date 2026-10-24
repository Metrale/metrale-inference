// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: A plan's selection pass: the ceiling pass over persisted pool keys a captured
//! decode row or a captured verify's rows run (`launch::run` with a ceiling and
//! `scores::persisted_for`), its kernels and ceiling taken from the plan instead of the levers:
//! the lane-per-head scores over one block per SM, the multi-CTA radix top-k, the expansion from
//! each row's device geometry.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - Reads no lever. A shape, cache or scratch the plan's kernels cannot take is an error naming
//!   it, never another kernel: the conditions `launch::run` tests before it takes this path.
//! - The launches are the shared launchers `launch::run` calls (`scores::launch_persisted`,
//!   `topk::launch_mc`, `scores::launch_expand_geom`) with the ceiling's arguments
//!   (`launch::ceiling_args`).

use super::launch::ceiling_args;
use super::scores::{SCORES_HEADS_D, SCORES_HEADS_MAX_ROWS};
use super::*;
use anyhow::ensure;
use metrale_gpu_runtime::gpu::KernelHandle;

/// 2026-10-23: A plan's selection kernels bound to this device, and the ceiling they launch at.
#[derive(Clone, Copy, Debug)]
pub struct PlannedSelect {
    /// 2026-10-23: `dsa_index_scores_heads_paged` or `dsa_index_scores_heads_window`.
    pub scores: KernelHandle,
    /// 2026-10-23: The scores' grid x: one block per SM.
    pub sm_count: u32,
    pub topk_pass: KernelHandle,
    pub topk_collect: KernelHandle,
    /// 2026-10-23: CTAs per row of the top-k (`topk::topk_mc_ctas` of `max_pools`).
    pub topk_ctas: usize,
    pub expand: KernelHandle,
    /// 2026-10-23: The ceiling's pool count (`contiguous_pool_count` of the context ceiling).
    pub max_pools: usize,
}

/// 2026-10-23: Select `geom.q_rows` rows with `p` into `scratch`, each row's geometry at its slot
/// of `inputs.geom_dev`.
pub fn select_planned(
    gpu: &dyn GpuBackend,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    p: &PlannedSelect,
    stream: u64,
) -> Result<()> {
    let rows = geom.q_rows;
    ensure!(
        inputs.geom_dev.0 != 0,
        "DSA plan select: a ceiling pass reads its geometry from the device and has none"
    );
    scratch.fits_for(cfg, geom, false)?;
    let (seq_a, npools_a, _, selk_a) = ceiling_args(cfg, geom.index_kpool, p.max_pools);
    ensure!(
        rows == 1
            || scratch
                .rows_regions(cfg, rows, p.max_pools, selk_a)
                .is_some(),
        "DSA plan select: the scratch does not hold {rows} rows at {} pools",
        p.max_pools
    );
    ensure!(
        inputs.rows.pool_keys.0 != 0 && inputs.first_key == 0 && inputs.rows.fp8.is_none(),
        "DSA plan select: the plan scores persisted BF16-cache pool keys from the first key; \
         this cache keeps none"
    );
    ensure!(
        geom.index_heads <= 32
            && geom.index_head_dim == SCORES_HEADS_D
            && rows <= SCORES_HEADS_MAX_ROWS
            && inputs.rows.pool_keys.0.is_multiple_of(16)
            && inputs.rows.pk_blk.is_multiple_of(4)
            && p.sm_count > 0,
        "DSA plan select: the lane-per-head scores take at most 32 heads of {SCORES_HEADS_D}, \
         {SCORES_HEADS_MAX_ROWS} rows, 16-byte aligned keys and a resolved SM count ({} heads of \
         {}, {rows} rows, {} SMs)",
        geom.index_heads,
        geom.index_head_dim,
        p.sm_count
    );
    ensure!(
        selk_a <= TOPK_RADIX_MAX,
        "DSA plan select: the radix top-k takes at most {TOPK_RADIX_MAX} pools, the ceiling \
         selects {selk_a}"
    );
    let grid = [p.sm_count, 1, 1];
    let a = (npools_a, seq_a);
    scores::launch_persisted(gpu, p.scores, grid, geom, inputs, scratch, a, stream)?;
    let mc = (p.topk_pass, p.topk_collect);
    let gd = inputs.geom_dev;
    topk::launch_mc(
        gpu,
        mc,
        p.topk_ctas,
        rows,
        scratch,
        (npools_a, selk_a),
        gd,
        stream,
    )?;
    let strides = (npools_a, selk_a);
    scores::launch_expand_geom(gpu, p.expand, cfg, geom, inputs, scratch, strides, stream)
}
