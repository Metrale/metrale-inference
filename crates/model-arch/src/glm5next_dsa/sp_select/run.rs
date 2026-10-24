// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The sequence-parallel DSA selection's forward half: the split decision for one
//! batched prefill pass, and the split run (each rank's two owned slabs, then two in-place
//! all-gathers of the selected token rows).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - [`plan`] decides from host integers every rank holds alike (the table, the world size,
//!   the pass's rows and pools, the prefill flag, the scratch's rows per sub-pass), so every
//!   rank enqueues the same gathers or none. It splits only when every sub-pass of the unsplit
//!   pass and of every rank's slabs has at least 2 rows: a 1-row sub-pass takes other scores
//!   kernels (`scores_mma_for` needs `q_rows > 1`) and so other bits, and a split that changed
//!   the sub-pass plan that way would not select the unsplit pass's bytes.
//! - 2026-10-17: a rank runs each slab in [`slab_passes`] (`even_row_passes`: as many
//!   sub-passes as the unsplit plan's steps of `per` would take, sized evenly), not in steps of
//!   `per`. A slab of `per * m + 1` rows stepped ends in a 1-row sub-pass, so [`plan`] refused
//!   the split at every chunk whose slabs had that length (at 8,192-row chunks on 3 ranks under a
//!   1,048,576-token ceiling: the chunks ending at 196,608 * m tokens) and every rank selected
//!   all the chunk's rows. Even sizes leave every slab sub-pass at least 2 rows wide whenever
//!   `per >= 3`. It is the default because it cannot change a row's bits: every kernel after the
//!   compression reads and writes only its own row at its offset, and which kernels a sub-pass
//!   launches depends on its row count only through `q_rows > 1` (`scores_mma_shape`,
//!   `scores_tiled_for`, `fused_shape`; the top-k, the fold and the expansion not at all). The
//!   split already relied on that: a slab's sub-passes start at the slab's first row, not at the
//!   unsplit plan's boundaries, and the gate (`glm5next_dsa_sp_select_gate`) checks the bytes.
//!   The unsplit pass keeps `row_passes`.
//! - [`run_split`] writes this rank's owned rows exactly as the unsplit pass writes them (the
//!   row sub-passes of `select_tokens_passes`), sends only those rows and receives every peer's
//!   rows into their own places, so after it every rank's `[k, out_width]` tokens are the
//!   unsplit pass's. The caller's stream waits for both gathers before it returns, and no
//!   gather reads or writes a row a kernel of the pass is writing at the time (slab A is sent
//!   while slab B, a disjoint row range, is selected).
//! - 2026-10-15: under [`Payload::Selected`] [`run_split`] runs the same scores and top-k on
//!   the owned rows, folds them (`select::folded`), gathers the folded `[rows, select_k]`
//!   selected pools instead of the token rows, and then every rank expands all `k` rows from
//!   them on the caller's stream: the token rows of the unsplit pass (`dsa_sel_fold.cu`).
//! - [`gathers`] counts the gathers enqueued, process-wide; a decode step never moves it
//!   ([`plan`] is `None` off prefill).

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use metrale_comm::CommBackend;
use metrale_gpu_runtime::gpu::GpuBackend;

use super::super::select::{
    DsaSelectGeometry, DsaSelectInputs, DsaSelectScratch, even_row_passes, expand_folded,
    folded_check, row_passes, select_passes_folded, select_tokens_in_passes, select_tokens_passes,
};
use super::super::{Glm5NextDsaConfig, Glm5NextDsaKernels};
use super::lever::Payload;
use super::{MIN_SUBPASS_ROWS, MinPoolsTable, SLABS_PER_RANK, applies, gather_shards, owned};
use crate::glm5next_sp_mhc::all_gather_rows;

static GATHERS: AtomicU64 = AtomicU64::new(0);

/// 2026-10-15: In-place gathers of selected rows enqueued by this process so far (two per split
/// pass).
pub fn gathers() -> u64 {
    GATHERS.load(Ordering::Relaxed)
}

/// 2026-10-15: The comm stream the gathers run on and the three events that order them against
/// the caller's stream: slab A selected, slab B selected, both gathers done.
#[derive(Clone, Copy, Debug)]
pub struct SplitStreams {
    pub comm: u64,
    pub slab_a: u64,
    pub slab_b: u64,
    pub done: u64,
}

/// 2026-10-15: The split's comm stream (greatest priority, as the SP overlap's) and events,
/// created at the first split pass, never under capture (a batched selection is never
/// captured).
fn split_streams(gpu: &dyn GpuBackend) -> Result<SplitStreams> {
    static S: Mutex<Option<SplitStreams>> = Mutex::new(None);
    let mut g = S.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = *g {
        return Ok(s);
    }
    let ev = || gpu.create_event().context("DSA SP select: event");
    let s = SplitStreams {
        comm: gpu
            .create_stream_greatest_priority()
            .context("DSA SP select: comm stream")?,
        slab_a: ev()?,
        slab_b: ev()?,
        done: ev()?,
    };
    *g = Some(s);
    Ok(s)
}

/// 2026-10-17: A slab's row sub-passes in a split pass: `even_row_passes` at `per` rows per
/// sub-pass (the module's invariants say why the bits are the unsplit pass's).
pub fn slab_passes(slab: (usize, usize), per: usize) -> Result<Vec<(usize, usize)>> {
    even_row_passes(&[slab], per)
}

/// 2026-10-17: A planner of one slab's row sub-passes at `per` rows per sub-pass.
pub type SlabPlan = fn((usize, usize), usize) -> Result<Vec<(usize, usize)>>;

/// 2026-10-15: This rank's two owned slabs when a pass of `k` rows over `pools` pools splits,
/// `None` when it runs unsplit: [`applies`] with the lever on (`table` given), and every
/// sub-pass of at least [`MIN_SUBPASS_ROWS`] rows at `per` rows per sub-pass, the unsplit pass's
/// (`row_passes`) and every rank's slabs' ([`slab_passes`], 2026-10-17).
pub fn plan(
    table: Option<&MinPoolsTable>,
    world: usize,
    rank: usize,
    k: usize,
    pools: usize,
    is_prefill: bool,
    per: usize,
) -> Result<Option<[(usize, usize); SLABS_PER_RANK]>> {
    plan_with(slab_passes, table, world, rank, k, pools, is_prefill, per)
}

/// 2026-10-17: [`plan`] with the slabs' sub-passes from `slab_plan`; the tests' negative control
/// passes the steps of `per` the split ran before 2026-10-17.
#[allow(clippy::too_many_arguments)]
pub fn plan_with(
    slab_plan: SlabPlan,
    table: Option<&MinPoolsTable>,
    world: usize,
    rank: usize,
    k: usize,
    pools: usize,
    is_prefill: bool,
    per: usize,
) -> Result<Option<[(usize, usize); SLABS_PER_RANK]>> {
    let Some(table) = table else {
        return Ok(None);
    };
    if per == 0 || !applies(true, world, k, pools, is_prefill, table) {
        return Ok(None);
    }
    let wide = |passes: Vec<(usize, usize)>| passes.iter().all(|&(_, n)| n >= MIN_SUBPASS_ROWS);
    if !wide(row_passes(&[(0, k)], per)?) {
        return Ok(None);
    }
    for r in 0..world {
        for slab in owned(k, world, r)? {
            if !wide(slab_plan(slab, per)?) {
                return Ok(None);
            }
        }
    }
    owned(k, world, rank).map(Some)
}

/// 2026-10-15: The batched prefill selection of one pass: split over the ranks of `comm` when
/// [`plan`] says so under `table` (`None` = lever off), else `select_tokens_in_passes`, the
/// split gathering `payload`. Returns the sub-passes this rank ran.
#[allow(clippy::too_many_arguments)]
pub fn select_with(
    table: Option<&MinPoolsTable>,
    payload: Payload,
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    comm: Option<&dyn CommBackend>,
    is_prefill: bool,
    stream: u64,
) -> Result<usize> {
    if let Some(c) = comm {
        let per = scratch.rows_per_pass(geom);
        let split = plan(
            table,
            c.world_size(),
            c.rank(),
            geom.q_rows,
            geom.n_pools,
            is_prefill,
            per,
        )?;
        if let Some(own) = split {
            let s = split_streams(gpu)?;
            return run_split(
                gpu, kernels, cfg, geom, inputs, scratch, c, &s, own, payload, stream,
            );
        }
    }
    select_tokens_in_passes(gpu, kernels, cfg, geom, inputs, scratch, stream)
}

/// 2026-10-15: The split pass on `main`: slab A (compressing the pools), its gather on the comm
/// stream while slab B is selected on `main`, slab B's gather, and `main` waiting for both;
/// under [`Payload::Selected`] the slabs are folded selected pools and `main` then expands all
/// rows. Errors before any launch when the selected-pools payload cannot run on this pass.
#[allow(clippy::too_many_arguments)]
pub fn run_split(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    comm: &dyn CommBackend,
    s: &SplitStreams,
    own: [(usize, usize); SLABS_PER_RANK],
    payload: Payload,
    main: u64,
) -> Result<usize> {
    let shards = gather_shards(geom.q_rows, comm.world_size());
    let sel = payload == Payload::Selected;
    let (buf, row_bytes) = if sel {
        folded_check(kernels, geom, inputs, scratch)?;
        (scratch.selected(), geom.select_k * 4)
    } else {
        (scratch.tokens(), cfg.out_width() * 4)
    };
    let per = scratch.rows_per_pass(geom);
    let mut passes = 0;
    for (i, (slab, event)) in own.into_iter().zip([s.slab_a, s.slab_b]).enumerate() {
        let (sub, compress) = (slab_passes(slab, per)?, i == 0);
        passes += if sel {
            select_passes_folded(
                gpu, kernels, cfg, geom, inputs, scratch, &sub, compress, main,
            )?
        } else {
            select_tokens_passes(
                gpu, kernels, cfg, geom, inputs, scratch, &sub, compress, main,
            )?
        };
        gpu.record_event(event, main)?;
        gpu.stream_wait_event(s.comm, event)?;
        all_gather_rows(comm, buf, &shards[i], row_bytes, s.comm)?;
        GATHERS.fetch_add(1, Ordering::Relaxed);
    }
    gpu.record_event(s.done, s.comm)?;
    gpu.stream_wait_event(main, s.done)?;
    if sel {
        expand_folded(gpu, kernels, cfg, geom, inputs, scratch, main)?;
    }
    Ok(passes)
}
