// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: `METRALE_GLM_MOE_W4A4_GROUPED=1`: routed-expert launches wider than
//! `MOE_W4A4_MAX_ROWS` (prefill sub-chunks, wide batched decode) at the checkpoint's declared
//! W4A4 on the grouped mx GEMM (`kernels/gb10/common/w4a4_grouped_mx.cu`) instead of the
//! grouped W4A16 GEMM. Each routed slot's gate, up and down output is byte for byte the slot
//! GEMV's (`w4a4_gemv_mx8_moe_slots`), the kernel the declared W4A4 decode runs, so a prompt row
//! gets the bits a decode row of the same input gets.
//!
//! Sequence: sort the slots by expert, the tile pass, quantize the rows under the gate/up
//! `input_scale`, gate and up over all slots (gathered through `sorted_token_ids`), the clamped
//! SwiGLU, quantize every slot's product under the down `input_scale`, down into `expert_out` in
//! expert-sorted order (the caller combines through `token_to_perm`). No host read, so a CUDA
//! graph may capture it.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Launches only after every shape check passes; never falls back to another format.
//! - [`takes_grouped`] is the one row test `forward_moe_pieces` and `w4a4_experts` both read.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::launch::swiglu;
use super::moe_experts::MoeSite;
use super::w4a4::quant_static;
use super::w4a4_grouped_gu::GuPlan;
use super::w4a4_grouped_nc::down_entry as nc_down;
use super::w4a4_grouped_raster::{GroupedEntry, RasterGroups, down_entry as raster_down, gu_entry};
use crate::glm5next_layer::profile;
use crate::glm5next_mlp::build_w4a4::{check_w4a4_k, check_w4a4_k64};
use crate::glm5next_mlp::weights::{Glm5NextExpertPtrTable, W4a4ActScales};
use crate::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels, MOE_W4A4_MAX_ROWS};

/// 2026-10-10: Weight rows (output columns) one CTA of each entry covers (`NT`).
pub const W4A4_GROUPED_COLS: [usize; 3] = [128, 128, 64];
/// 2026-10-10: The widest K the kernel's chunk-order table holds (`W4G_MAX_CHUNKS` x 128).
pub const W4A4_GROUPED_MAX_K: usize = 8192;
/// 2026-10-10: Slots per tile of the three entries (`m32`, `m64`, `m128`).
pub const W4A4_GROUPED_MT: [usize; 3] = [32, 64, 128];
/// 2026-10-10: Pipeline stages of each entry (`S`).
pub const W4A4_GROUPED_STAGES: [usize; 3] = [3, 3, 3];

/// 2026-10-10: Dynamic shared memory of an entry: `sizeof(W4gSmem<MT, NT, S>)` in
/// `w4a4_grouped_mx.cu` (S stages of NT weight rows and MT slots at 72 bytes a row, or the BF16
/// output tile when larger; then the slot rows, their globals and the 64-chunk order), plus 16.
pub fn smem_bytes(mt: usize, nt: usize, stages: usize) -> u32 {
    let stage = nt * 72 + mt * 72;
    let union = (stages * stage).max(mt * (nt + 8) * 2);
    (union.next_multiple_of(4) + mt * 8 + 64 + 16) as u32
}

/// 2026-10-10: The 128-slot tile from this many routed slots per expert on average (each
/// expert's weights then stream once for all its slots); below it the 32-slot tile wastes less
/// on experts with few slots. GB10 microbench (`examples/glm5next_moe_prefill_w4a4_bench.rs`)
/// at 2048 rows; the bits do not depend on the tile.
pub const W4A4_GROUPED_M128_MIN_AVG: usize = 48;
const BLOCK: u32 = 256;

/// 2026-10-10: Whether a W4A4 routed launch of `rows` rows runs the grouped GEMM: wider than
/// the slot GEMV and sweeps serve.
pub fn takes_grouped(rows: usize) -> bool {
    rows > MOE_W4A4_MAX_ROWS
}

/// 2026-10-10: Whether `forward_moe` zeroes `expert_out` before the routed experts: always,
/// except before the grouped W4A4 GEMM on a rank that holds every expert, whose down launch
/// writes every sorted row (a null weight pointer is the only row it skips).
pub fn zeroes_expert_out(grouped_w4a4: bool, cfg: &Glm5NextMlpConfig) -> bool {
    !(grouped_w4a4 && cfg.local_experts == cfg.num_experts)
}

/// 2026-10-10: The tile index (into `W4A4_GROUPED_MT`) for `slots` routed slots over
/// `experts` experts.
pub fn tile_index(slots: usize, experts: usize) -> usize {
    if slots >= W4A4_GROUPED_M128_MIN_AVG * experts.max(1) {
        2
    } else {
        0
    }
}

/// 2026-10-10: One grouped launch: `c` rows of every expert's sorted slots, `n` columns, from
/// the quantized scratch over `k`; `gather` is `sorted_token_ids` (gate/up) or null (down).
#[allow(clippy::too_many_arguments)]
fn grouped_mx(
    site: &MoeSite<'_>,
    entry: GroupedEntry,
    ti: usize,
    table: &Glm5NextExpertPtrTable,
    c: DevicePtr,
    gather: DevicePtr,
    n: usize,
    k: usize,
) -> Result<()> {
    let (cfg, ws) = (site.cfg, site.ws);
    grouped_launch(
        site.gpu,
        entry,
        ti,
        smem_bytes_of(ti),
        [ws.w4a4_aq, ws.w4a4_as, ws.w4a4_ag],
        table,
        c,
        ws.expert_offsets(),
        ws.w4a4_tile_off,
        gather,
        cfg.num_experts,
        site.rows * cfg.top_k,
        n,
        k,
        site.stream,
    )
}

/// 2026-10-12: The dynamic shared memory of tile entry `ti`.
pub(super) fn smem_bytes_of(ti: usize) -> u32 {
    smem_bytes(
        W4A4_GROUPED_MT[ti],
        W4A4_GROUPED_COLS[ti],
        W4A4_GROUPED_STAGES[ti],
    )
}

/// 2026-10-12: The grouped GEMM launch with its inputs spelled out (`grouped_mx` for a routed
/// site, `w4a4_dense_grouped::proj` for a dense site as one group of `num_experts` = 1):
/// `act` = the quantized rows, scales and globals; grid (ceil(n / cols), ceil(slots / mt) +
/// num_experts). 2026-10-14: an `_r` entry takes its raster group as one more argument.
/// 2026-10-15: an `_n` entry its raster group and `nc`, over a grid `nc` times narrower.
#[allow(clippy::too_many_arguments)]
pub(super) fn grouped_launch(
    gpu: &dyn GpuBackend,
    entry: GroupedEntry,
    ti: usize,
    smem: u32,
    act: [DevicePtr; 3],
    table: &Glm5NextExpertPtrTable,
    c: DevicePtr,
    offsets: DevicePtr,
    tile_off: DevicePtr,
    gather: DevicePtr,
    num_experts: usize,
    slots: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    let l = KernelLaunch::new(gpu, entry.kern)
        .grid([
            entry.grid_x(n, W4A4_GROUPED_COLS[ti]),
            (slots.div_ceil(W4A4_GROUPED_MT[ti]) + num_experts) as u32,
            1,
        ])
        .block([BLOCK, 1, 1])
        .shared_mem(smem)
        .arg_ptr(act[0])
        .arg_ptr(act[1])
        .arg_ptr(act[2])
        .arg_ptr(table.packed_ptrs)
        .arg_ptr(table.scale_ptrs)
        .arg_ptr(table.scale2_vals)
        .arg_ptr(c)
        .arg_ptr(offsets)
        .arg_ptr(tile_off)
        .arg_ptr(gather)
        .arg_u32(num_experts as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32);
    entry.tail_args(l).launch(stream)
}

/// 2026-10-10: The down projection's entry and stored activation width for expert width `mi`:
/// the plain entry and quantizer at `mi % 128 == 0`, the `_k64` twins at `mi % 128 == 64`.
fn down_entry(
    k: &Glm5NextMlpKernels,
    mi: usize,
    ti: usize,
) -> Result<(KernelHandle, KernelHandle, usize)> {
    if mi.is_multiple_of(128) {
        check_w4a4_k(mi, "routed down")?;
        return Ok((k.w4a4_grouped.mx[ti], k.w4a4_quant_static, mi));
    }
    check_w4a4_k64(mi, "routed down")?;
    Ok((
        k.w4a4_grouped.mx_k64[ti],
        k.w4a4_moe_k64.quant,
        mi.next_multiple_of(128),
    ))
}

/// 2026-10-10: The routed experts of a W4A4 `forward_moe` wider than `MOE_W4A4_MAX_ROWS`, every
/// slot of every expert in one launch per projection; `expert_out` ends in expert-sorted order.
/// 2026-10-14: Gate, up, SwiGLU and the down quantization as one launch under
/// `METRALE_GLM_MOE_W4A4_GU_FUSED=1` (`w4a4_grouped_gu`), same bytes; the m128 launches'
/// raster under `METRALE_GLM_MOE_W4A4_RASTER` (`w4a4_grouped_raster`), same bytes.
/// 2026-10-15: the m128 down's column tiles per CTA under `METRALE_GLM_MOE_W4A4_NC`
/// (`w4a4_grouped_nc`), same bytes.
pub(super) fn w4a4_grouped_experts(site: &MoeSite<'_>, scales: W4a4ActScales) -> Result<()> {
    w4a4_grouped_experts_with(
        site,
        scales,
        site.k.levers.mlp.gu_fused,
        site.k.levers.mlp.raster,
        site.k.levers.mlp.nc,
    )
}

/// 2026-10-14: [`w4a4_grouped_experts`] with the fused gate/up and raster levers' values given.
/// 2026-10-15: and the column tiles per m128 down CTA (`METRALE_GLM_MOE_W4A4_NC`).
pub(super) fn w4a4_grouped_experts_with(
    site: &MoeSite<'_>,
    scales: W4a4ActScales,
    fused: bool,
    raster: Option<RasterGroups>,
    nc: Option<u32>,
) -> Result<()> {
    let MoeSite {
        gpu,
        k,
        cfg,
        w,
        x,
        rows,
        ws,
        stream,
    } = *site;
    let (mi, h) = (cfg.moe_intermediate, cfg.hidden);
    let slots = rows * cfg.top_k;
    let ti = tile_index(slots, cfg.num_experts);
    let mt = W4A4_GROUPED_MT[ti];
    check_w4a4_k(h, "routed gate/up")?;
    let (down_plain, down_quant, k_store) = down_entry(k, mi, ti)?;
    let k64 = k_store != mi;
    let down = nc_down(k, nc, ti, k64, raster_down(k, raster, ti, k64, down_plain));
    let gate_up = GroupedEntry::plain(k.w4a4_grouped.mx[ti]);
    ensure!(
        h <= W4A4_GROUPED_MAX_K && k_store <= W4A4_GROUPED_MAX_K,
        "GLM routed W4A4 grouped: K {h} / {k_store} past the kernel's {W4A4_GROUPED_MAX_K}"
    );
    ensure!(
        h.is_multiple_of(8) && mi.is_multiple_of(8),
        "GLM routed W4A4 grouped: N {mi} / {h} must be multiples of 8 (16-byte row stores)"
    );
    ensure!(
        [
            gate_up.kern,
            down.kern,
            down_quant,
            k.w4a4_grouped.tiles,
            k.moe_sort_by_expert
        ]
        .iter()
        .all(|h| h.0 != 0)
            && k.combine_indexed.0 != 0,
        "GLM routed W4A4 grouped: a kernel of the path is absent from this target's PTX"
    );
    ensure!(
        slots <= ws.max_total_expanded() && slots <= ws.w4a4_rows,
        "GLM routed W4A4 grouped: {slots} slots do not fit the workspace ({} sorted, {} \
         quantized rows); it was sized without METRALE_GLM_MOE_W4A4_GROUPED=1",
        ws.max_total_expanded(),
        ws.w4a4_rows
    );
    let gu = fused
        .then(|| GuPlan::new(gu_entry(k, raster, ti, k.w4a4_grouped.gu[ti]), ti, k_store))
        .transpose()?;
    let t = profile::start();
    KernelLaunch::new(gpu, k.moe_sort_by_expert)
        .grid([1, 1, 1])
        .block([BLOCK, 1, 1])
        .arg_ptr(ws.ids)
        .arg_ptr(ws.sorted_token_ids())
        .arg_ptr(ws.sorted_expert_ids())
        .arg_ptr(ws.expert_offsets())
        .arg_ptr(ws.token_to_perm())
        .arg_u32(slots as u32)
        .arg_u32(cfg.num_experts as u32)
        .arg_u32(cfg.top_k as u32)
        .launch(stream)?;
    KernelLaunch::new(gpu, k.w4a4_grouped.tiles)
        .grid([1, 1, 1])
        .block([32, 1, 1])
        .arg_ptr(ws.expert_offsets())
        .arg_ptr(ws.w4a4_tile_off)
        .arg_u32(cfg.num_experts as u32)
        .arg_u32(mt as u32)
        .launch(stream)?;
    quant_static(
        gpu,
        k.w4a4_quant_static,
        ws,
        x,
        rows,
        h,
        h,
        scales.gate_up,
        stream,
    )?;
    if let Some(gu) = gu {
        gu.launch(site, down, scales.down)?;
        profile::end(profile::MOE_EXPERTS, t, gpu, stream);
        return Ok(());
    }
    let gather = ws.sorted_token_ids();
    grouped_mx(site, gate_up, ti, &w.ptrs.gate, ws.a_gate, gather, mi, h)?;
    grouped_mx(site, gate_up, ti, &w.ptrs.up, ws.a_up, gather, mi, h)?;
    swiglu(
        gpu,
        k.swiglu,
        ws.a_gate,
        ws.a_up,
        ws.a_act,
        slots * mi,
        cfg.swiglu_limit,
        stream,
    )?;
    quant_static(
        gpu,
        down_quant,
        ws,
        ws.a_act,
        slots,
        mi,
        k_store,
        scales.down,
        stream,
    )?;
    grouped_mx(
        site,
        down,
        ti,
        &w.ptrs.down,
        ws.expert_out,
        DevicePtr::NULL,
        h,
        mi,
    )?;
    profile::end(profile::MOE_EXPERTS, t, gpu, stream);
    Ok(())
}

#[cfg(test)]
#[path = "w4a4_grouped_tests.rs"]
mod tests;
