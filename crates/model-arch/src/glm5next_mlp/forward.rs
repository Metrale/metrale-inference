// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The GLM-5.3 MLP forward over `rows` rows: the dense SwiGLU FFN (also used for the
//! shared expert) and the routed MoE site.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - `forward_dense` and `forward_moe` leave a partial sum in `out` when the site is split over
//!   TP or EP ranks; the caller all-reduces.
//! - `forward_moe` zeroes `expert_out` before any expert writes it, so an expert another EP rank
//!   owns adds zero.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::forward_prefill_gemm;
use super::weights::Glm5NextMoeWeights;
use super::{Glm5NextMlpConfig, Glm5NextMlpKernels};

mod dense;
mod launch;
mod levers;
mod moe_experts;
mod pieces;
mod route;
mod router;
mod shared_fork;
mod topk_combine;
mod w4a4;
mod w4a4_dense;
mod w4a4_dense_grouped;
mod w4a4_grouped;
mod w4a4_grouped_gu;
mod w4a4_grouped_nc;
mod w4a4_grouped_raster;
mod w4a4_slots;
mod workspace;

pub use dense::{DenseScratch, forward_dense, forward_dense_site};
use launch::swiglu;
pub use levers::MlpLevers;
pub(crate) use levers::count;
pub(crate) use pieces::check_pieces;
pub use route::{ExpertRoute, expert_route, grouped_row_floor};
pub use shared_fork::SHARED_FORK_MAX_ROWS;
pub use w4a4_dense::forward_dense_w4a4;
pub use w4a4_dense_grouped::DenseGroupedTables;
pub use w4a4_grouped::{
    smem_bytes as w4a4_grouped_smem_bytes, takes_grouped as w4a4_takes_grouped,
    tile_index as w4a4_grouped_tile_index,
};
pub use w4a4_grouped_gu::{W4a4GuCall, check_at_load as check_gu_fused_at_load, launch_w4a4_gu};
pub use w4a4_grouped_nc::check_at_load as check_nc_at_load;
pub use w4a4_grouped_raster::{GroupedEntry, RasterGroups, check_at_load as check_raster_at_load};
pub use w4a4_slots::{CUDA_MAX_GRID_Y, W4a4SlotsCall, launch_w4a4_slots, slot_pieces};
pub use workspace::{mlp_ws_bytes, mlp_ws_total_bytes};

const ACT_BLOCK: u32 = 256;

/// 2026-09-25: `out = silu(min(gate, limit)) * clamp(up, -limit, limit)` over `n` elements
/// (`glm5next_swiglu_clamp`), for the grouped-prefill module.
pub(super) fn swiglu_rows(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    out: DevicePtr,
    n: usize,
    limit: f32,
    stream: u64,
) -> Result<()> {
    swiglu(gpu, k, gate, up, out, n, limit, stream)
}

/// 2026-09-25: MLP scratch for up to `max_rows` rows, allocated once at load. The buffer sizes
/// are listed in `mlp_ws_bytes`.
pub struct Glm5NextMlpWorkspace {
    /// 2026-09-25: Gate, up and activated values, BF16, each
    /// `max(rows * max_inter, rows * top_k * moe_intermediate)` elements.
    a_gate: DevicePtr,
    a_up: DevicePtr,
    a_act: DevicePtr,
    /// 2026-09-25: `[rows, num_experts]` F32 router logits.
    logits: DevicePtr,
    /// 2026-09-25: `[rows, top_k]` selected ids (I32); `wts` holds their F32 weights.
    ids: DevicePtr,
    wts: DevicePtr,
    /// 2026-09-25: `[rows, top_k, hidden]` BF16 routed outputs; `forward_moe` zeroes it before
    /// the experts run.
    expert_out: DevicePtr,
    /// 2026-09-25: `[rows, hidden]` BF16 shared-expert output.
    shared_out: DevicePtr,
    /// 2026-09-25: `[rows * top_k]` I32 union expert ids, `-1` for an unused entry. Row-batched
    /// path only.
    u_eid: DevicePtr,
    /// 2026-09-25: `[rows * top_k, rows]` I32: the slot row `r` gave union entry `u`, `-1` when
    /// row `r` did not select it.
    u_slot: DevicePtr,
    /// 2026-09-25: `[rows * top_k]` I32: sorted row to original token. Grouped GEMM only.
    sorted_token_ids: DevicePtr,
    /// 2026-09-25: `[rows * top_k]` I32: sorted row to expert id. Grouped GEMM only.
    sorted_expert_ids: DevicePtr,
    /// 2026-09-25: `[num_experts + 1]` I32 prefix sum over the expert-sorted rows.
    expert_offsets: DevicePtr,
    /// 2026-09-25: `[rows, top_k]` I32: a slot's row in the expert-sorted output, read by
    /// `glm5next_moe_combine_indexed`.
    token_to_perm: DevicePtr,
    /// 2026-10-08: W4A4 activation scratch (`w4a4_scratch_dims`): `[w4a4_rows, k / 2]` E2M1
    /// codes, `[w4a4_rows, k / 16]` E4M3 scales and `[w4a4_rows]` F32 global scales, in the
    /// layout `w4a4_quant_rows_static` writes.
    w4a4_aq: DevicePtr,
    w4a4_as: DevicePtr,
    w4a4_ag: DevicePtr,
    /// 2026-10-10: `[num_experts + 1]` I32: the first 32- or 64-slot tile of each expert in the
    /// grouped W4A4 GEMM (`w4a4_grouped_tiles`), the last entry the tile count.
    w4a4_tile_off: DevicePtr,
    /// 2026-10-08: Rows, and the widest K, the W4A4 scratch holds.
    w4a4_rows: usize,
    w4a4_k: usize,
    max_inter: usize,
    /// 2026-09-25: Widest row group this scratch serves, at least 1.
    max_rows: usize,
    /// 2026-09-25: `max_rows * top_k`, the routed-slot count the grouped-path buffers hold.
    max_total_expanded: usize,
    /// 2026-10-11: The shared expert's side stream and scratch (`METRALE_GLM_SHARED_FORK`).
    shared_fork: Option<shared_fork::SharedFork>,
}

impl Glm5NextMlpWorkspace {
    /// 2026-10-11: The gate, up and activated buffers of a dense pass (`forward_dense`).
    pub(super) fn dense_scratch(&self) -> DenseScratch {
        DenseScratch {
            gate: self.a_gate,
            up: self.a_up,
            act: self.a_act,
            max_inter: self.max_inter,
            max_rows: self.max_rows,
        }
    }
    /// 2026-09-25: Widest row group this scratch serves.
    pub(super) fn max_rows(&self) -> usize {
        self.max_rows
    }
    /// 2026-09-25: `max_rows * top_k`, the routed-slot count the grouped buffers hold.
    pub(super) fn max_total_expanded(&self) -> usize {
        self.max_total_expanded
    }
    pub(super) fn ids(&self) -> DevicePtr {
        self.ids
    }
    pub(super) fn a_gate(&self) -> DevicePtr {
        self.a_gate
    }
    pub(super) fn a_up(&self) -> DevicePtr {
        self.a_up
    }
    pub(super) fn a_act(&self) -> DevicePtr {
        self.a_act
    }
    pub(super) fn expert_out(&self) -> DevicePtr {
        self.expert_out
    }
    pub(super) fn sorted_token_ids(&self) -> DevicePtr {
        self.sorted_token_ids
    }
    pub(super) fn sorted_expert_ids(&self) -> DevicePtr {
        self.sorted_expert_ids
    }
    pub(super) fn expert_offsets(&self) -> DevicePtr {
        self.expert_offsets
    }
    pub(super) fn token_to_perm(&self) -> DevicePtr {
        self.token_to_perm
    }
}

/// 2026-09-25: Widest compiled `w4a16_gemv_sw_moe_batchm_mR` tier: the
/// `METRALE_MOE_BATCHM_ENTRY(2..=16)` instances in `kernels/gb10/common/w4a16_gemv.cu`, held at
/// index `R - 2` of `Glm5NextMlpKernels::w4a16_gemv_sw_moe_batchm`. `forward_moe` splits a wider
/// row group into sub-groups no wider than this. 2026-10-09: 16 (was 8), so a 16-row decode
/// group reads each union expert once instead of once per 8-row half;
/// `METRALE_GLM_MOE_ROW_BATCH_MAX=8` restores the halves.
pub const MOE_ROW_BATCH_MAX_ROWS: usize = 16;

/// 2026-09-25: Split `rows` into consecutive `(start, width)` sub-groups of at most `cap`, as
/// even as the count allows. There is no width-1 tier; at `cap = MOE_ROW_BATCH_MAX_ROWS` no
/// group of a split of 2 or more rows is one row wide. At a smaller `cap` one can be (3 rows at
/// cap 2), and `forward_moe` then does not take the row-batched path.
///
/// The split does not change a row's result: its routed output is the sum over its own top-k
/// slots, and each slot's GEMV accumulates in per-row registers, whatever other rows share the
/// sweep.
fn moe_row_groups(rows: usize, cap: usize) -> Vec<(usize, usize)> {
    let n = rows.div_ceil(cap.max(1)).max(1);
    let mut out = Vec::with_capacity(n);
    let mut start = 0usize;
    for i in 0..n {
        let w = (rows - start).div_ceil(n - i);
        out.push((start, w));
        start += w;
    }
    out
}

/// 2026-09-25: Most ids `glm5next_moe_row_union` resolves: it runs as one block of
/// `rows * top_k` threads, and `forward_moe` does not take the row-batched path above this.
/// 2026-10-09: 128 (was 64): `MOE_ROW_BATCH_MAX_ROWS` rows at GLM-5.3's top-8.
pub const MOE_ROW_UNION_MAX_IDS: usize = 128;

/// 2026-09-25: Log once per row count (counts from 15 up share one bit) whether the routed
/// experts took the row-batched path.
fn announce_row_batch(batched: bool, rows: usize) {
    use std::sync::atomic::{AtomicU16, Ordering};
    static SEEN: AtomicU16 = AtomicU16::new(0);
    let bit = 1u16 << rows.min(15);
    if SEEN.fetch_or(bit, Ordering::Relaxed) & bit != 0 {
        return;
    }
    if batched {
        tracing::info!("GLM MoE: row-batched expert union ({rows} rows, one sweep each)");
    } else {
        tracing::info!("GLM MoE: per-row expert sweeps ({rows} rows)");
    }
}

/// 2026-09-25: Log once per row count (counts from 31 up share one bit) whether a row group
/// wider than `MOE_ROW_BATCH_MAX_ROWS` took the grouped GEMM; narrower groups log nothing.
fn announce_grouped_prefill(on: bool, rows: usize, t: super::forward_prefill_gemm::GemmTile) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEEN: AtomicU32 = AtomicU32::new(0);
    if rows <= MOE_ROW_BATCH_MAX_ROWS {
        return;
    }
    let bit = 1u32 << (rows.min(31));
    if SEEN.fetch_or(bit, Ordering::Relaxed) & bit != 0 {
        return;
    }
    if on {
        tracing::info!(
            "GLM MoE prefill: grouped tensor-core W4A16 GEMM, {rows} rows in ONE launch \
             per projection, tile `{}` (M_TILE {}, N_TILE {}, {} threads) \
             (METRALE_GLM_MOE_PREFILL_GEMM=0 to restore the GEMV path; \
             METRALE_GLM_MOE_GEMM_TILE=base for the prior tile)",
            t.name,
            t.m_tile,
            t.n_tile,
            t.threads
        );
    } else {
        tracing::info!(
            "GLM MoE prefill: row-batched GEMV, {rows} rows split into \
             {MOE_ROW_BATCH_MAX_ROWS}-row sweeps"
        );
    }
}

/// 2026-09-25: Log, once per process, whether the per-row expert path is device-dispatched
/// (`w4a16_gemv_sw_moe`) or host-dispatched.
fn announce_dispatch(grouped: bool) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if grouped {
            tracing::info!("GLM MoE: grouped device dispatch (no per-layer D2H)");
        } else {
            tracing::info!("GLM MoE: host dispatch (per-layer stream sync + D2H)");
        }
    });
}

/// 2026-09-25: One routed MoE site over `rows` rows of `x` into `out`: router, top-k, routed
/// experts, shared expert, and the weighted combine.
///
/// The routed experts take one of three paths:
/// - the grouped GEMM (`forward_prefill_gemm`), for `rows > MOE_ROW_BATCH_MAX_ROWS` and
///   `rows >= MlpLevers::prefill_gemm_min_rows`;
/// - the row-batched union GEMV over the sub-groups of `moe_row_groups`, for `rows >= 2`;
/// - per row otherwise, device-dispatched (`w4a16_gemv_sw_moe`) or host-dispatched.
///
/// Route tracing (`METRALE_GLM_ROUTE_TRACE=1`) forces the per-row path, the one that reads `ids`
/// back. Errors when `rows` does not fit the workspace or the bound expert count is not this
/// rank's share.
#[allow(clippy::too_many_arguments)]
pub fn forward_moe(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextMoeWeights,
    x: DevicePtr,
    out: DevicePtr,
    rows: usize,
    ws: &Glm5NextMlpWorkspace,
    // 2026-10-09: True while a CUDA graph is captured: the grouped GEMM then sizes its grid
    // from the worst case instead of reading the expert histogram back to the host.
    capturing: bool,
    stream: u64,
) -> Result<()> {
    forward_moe_pieces(gpu, k, cfg, w, x, out, rows, &[rows], ws, capturing, stream)
}

/// 2026-10-09: [`forward_moe`] with `row_pieces` splitting the `rows` rows into consecutive
/// pieces: the router and the shared expert, whose arithmetic depends on the row count, run
/// once per piece as a `forward_moe` of that piece's row count runs them (`pieces.rs`); the
/// top-k, the routed experts and the combine run once over all rows. The multi-sequence
/// prefill passes one piece per sequence; `forward_moe` passes `[rows]`.
#[allow(clippy::too_many_arguments)]
pub fn forward_moe_pieces(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextMoeWeights,
    x: DevicePtr,
    out: DevicePtr,
    rows: usize,
    row_pieces: &[usize],
    ws: &Glm5NextMlpWorkspace,
    capturing: bool,
    stream: u64,
) -> Result<()> {
    if rows == 0 || rows > ws.max_rows {
        bail!(
            "GLM MoE: {rows} rows do not fit a workspace built for {}",
            ws.max_rows
        );
    }
    if w.experts.len() != cfg.local_experts {
        bail!(
            "GLM MoE: {} bound experts but this rank owns {} of {}",
            w.experts.len(),
            cfg.local_experts,
            cfg.num_experts
        );
    }
    pieces::check_pieces(rows, row_pieces)?;

    use crate::glm5next_layer::profile;

    let groups = moe_row_groups(rows, k.levers.mlp.row_batch_max);
    // 2026-10-08: The precision plan picks W4A4 or the W4A16 paths below for this row count.
    // 2026-10-09: The path comes from `route::expert_route`, which the prefill grouping reads too.
    let path = route::expert_route(k, cfg, w, ws, rows);
    let w4a4 = match path {
        route::ExpertRoute::W4a4 => Some(w.act_scales.ok_or_else(|| {
            anyhow::anyhow!("GLM MoE: the plan runs W4A4 at {rows} rows but no scales were bound")
        })?),
        _ => None,
    };
    let grouped_prefill = path == route::ExpertRoute::Grouped;
    // 2026-10-10: A W4A4 launch wider than the slot GEMV serves runs the grouped W4A4 GEMM,
    // which leaves `expert_out` in expert-sorted order as the grouped W4A16 GEMM does.
    let grouped_w4a4 = w4a4.is_some() && w4a4_grouped::takes_grouped(rows);
    let sorted_out = grouped_prefill || grouped_w4a4;
    let batched = path == route::ExpertRoute::RowBatched;
    if w4a4.is_none() {
        announce_row_batch(batched, rows);
        announce_grouped_prefill(grouped_prefill, rows, k.levers.mlp.gemm_tile);
    }

    // 2026-10-11: Under `METRALE_GLM_SHARED_FORK` the shared expert starts here on its side
    // stream (it reads only `x`) and joins before the combine.
    let fork = ws.shared_fork.as_ref().filter(|f| f.takes(rows));
    if let Some(f) = fork {
        f.fork(gpu, stream, |side, scratch| {
            pieces::shared_expert(gpu, k, cfg, w, x, ws, scratch, row_pieces, side)
        })?;
    }

    let t = profile::start();
    pieces::router_logits(gpu, k, cfg, w, x, ws, row_pieces, stream)?;
    // 2026-09-25: One top-k launch for all rows, one block per row (`topk_combine`).
    topk_combine::router_topk(gpu, k, cfg, w, ws, rows, stream)?;
    profile::end(profile::MOE_ROUTER, t, gpu, stream);

    // 2026-09-25: Zero every routed output first: an expert this rank does not own writes
    // nothing, and its slots must add zero to the sum. 2026-10-10: Not before the grouped W4A4
    // GEMM when this rank holds every expert: its down launch writes every sorted row.
    // 2026-10-10: Nor before the one-row own-slots sweep, which zeroes the rows it computes
    // nothing for (`w4a4::writes_every_expert_row`).
    let own_rows = w4a4.is_some() && w4a4::writes_every_expert_row(k, cfg, rows);
    if w4a4_grouped::zeroes_expert_out(grouped_w4a4, cfg) && !own_rows {
        gpu.memset_async(ws.expert_out, 0, rows * cfg.top_k * cfg.hidden * 2, stream)?;
    }

    let site = moe_experts::MoeSite {
        gpu,
        k,
        cfg,
        w,
        x,
        rows,
        ws,
        stream,
    };
    if let Some(scales) = w4a4 {
        w4a4::w4a4_experts(&site, scales)?;
    } else {
        moe_experts::per_row_experts(&site, batched, grouped_prefill)?;
    }

    if grouped_prefill {
        // 2026-09-25: Leaves the routed outputs in expert-sorted order; the combine below reads
        // them through `token_to_perm`.
        let t = profile::start();
        forward_prefill_gemm::forward_moe_grouped_prefill(
            gpu, k, cfg, w, x, rows, ws, !capturing, stream,
        )?;
        profile::end(profile::MOE_EXPERTS, t, gpu, stream);
    }

    if batched {
        moe_experts::row_batched_experts(&site, &groups)?;
    }

    let t = profile::start();
    match fork {
        Some(f) => f.join(gpu, stream)?,
        None => pieces::shared_expert(
            gpu,
            k,
            cfg,
            w,
            x,
            ws,
            ws.dense_scratch(),
            row_pieces,
            stream,
        )?,
    }

    // 2026-09-25: The shared expert joins the routed sum in the combine, before the caller's
    // all-reduce, so one collective reduces both partial sums.
    profile::end(profile::MOE_SHARED, t, gpu, stream);
    let t = profile::start();
    // 2026-09-25: One combine launch for all rows (row = `blockIdx.x`), reading the routed rows
    // in expert-sorted order after the grouped GEMM (`topk_combine`).
    topk_combine::combine(gpu, k, cfg, ws, out, rows, sorted_out, stream)?;
    profile::end(profile::MOE_COMBINE, t, gpu, stream);
    Ok(())
}

#[cfg(test)]
mod tests;
