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
//! - [`run_split`] writes this rank's owned rows exactly as the unsplit pass writes them (the
//!   row sub-passes of `select_tokens_rows`), sends only those rows and receives every peer's
//!   rows into their own places, so after it every rank's `[k, out_width]` tokens are the
//!   unsplit pass's. The caller's stream waits for both gathers before it returns, and no
//!   gather reads or writes a row a kernel of the pass is writing at the time (slab A is sent
//!   while slab B, a disjoint row range, is selected).
//! - [`gathers`] counts the gathers enqueued, process-wide; a decode step never moves it
//!   ([`plan`] is `None` off prefill).

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use metrale_comm::CommBackend;
use metrale_gpu_runtime::gpu::GpuBackend;

use super::super::select::{
    DsaSelectGeometry, DsaSelectInputs, DsaSelectScratch, row_passes, select_tokens_in_passes,
    select_tokens_rows,
};
use super::super::{Glm5NextDsaConfig, Glm5NextDsaKernels};
use super::{
    MIN_SUBPASS_ROWS, MinPoolsTable, SLABS_PER_RANK, applies, gather_shards, lever, owned,
};
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

/// 2026-10-15: This rank's two owned slabs when a pass of `k` rows over `pools` pools splits,
/// `None` when it runs unsplit: [`applies`] with the lever on (`table` given), and every
/// sub-pass at `per` rows per sub-pass of at least [`MIN_SUBPASS_ROWS`] rows, unsplit and on every rank.
pub fn plan(
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
    let wide = |ranges: &[(usize, usize)]| -> Result<bool> {
        Ok(row_passes(ranges, per)?
            .iter()
            .all(|&(_, n)| n >= MIN_SUBPASS_ROWS))
    };
    if !wide(&[(0, k)])? {
        return Ok(None);
    }
    for r in 0..world {
        for slab in owned(k, world, r)? {
            if !wide(&[slab])? {
                return Ok(None);
            }
        }
    }
    owned(k, world, rank).map(Some)
}

/// 2026-10-15: The batched prefill selection of one pass: split over the ranks of `comm` when
/// [`plan`] says so under the lever's table, else `select_tokens_in_passes`. Returns the
/// sub-passes this rank ran.
#[allow(clippy::too_many_arguments)]
pub fn select(
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
    select_with(
        lever::table()?,
        gpu,
        kernels,
        cfg,
        geom,
        inputs,
        scratch,
        comm,
        is_prefill,
        stream,
    )
}

/// 2026-10-15: [`select`] under an explicit `table` (`None` = lever off).
#[allow(clippy::too_many_arguments)]
pub fn select_with(
    table: Option<&MinPoolsTable>,
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
            return run_split(gpu, kernels, cfg, geom, inputs, scratch, c, &s, own, stream);
        }
    }
    select_tokens_in_passes(gpu, kernels, cfg, geom, inputs, scratch, stream)
}

/// 2026-10-15: The split pass on `main`: slab A (compressing the pools), its gather on the comm
/// stream while slab B is selected on `main`, slab B's gather, and `main` waiting for both.
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
    main: u64,
) -> Result<usize> {
    let shards = gather_shards(geom.q_rows, comm.world_size());
    let row_bytes = cfg.out_width() * 4;
    let mut passes = 0;
    for (i, (slab, event)) in own.into_iter().zip([s.slab_a, s.slab_b]).enumerate() {
        passes += select_tokens_rows(
            gpu,
            kernels,
            cfg,
            geom,
            inputs,
            scratch,
            &[slab],
            i == 0,
            main,
        )?;
        gpu.record_event(event, main)?;
        gpu.stream_wait_event(s.comm, event)?;
        all_gather_rows(comm, scratch.tokens(), &shards[i], row_bytes, s.comm)?;
        GATHERS.fetch_add(1, Ordering::Relaxed);
    }
    gpu.record_event(s.done, s.comm)?;
    gpu.stream_wait_event(main, s.done)?;
    Ok(passes)
}
