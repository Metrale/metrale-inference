// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The W4A4 MLP forward of GLM-5.3, at the checkpoint's declared precision: NVFP4
//! weights times NVFP4 activations quantized under the static `input_scale`s, on the FP4
//! block-scale MMA. The routed experts run the persistent sweep, gate and up in one launch: at
//! one row over the row's own slots (`w4a4_gemv_mx8_moe_slots_sweep`, 2026-10-09), at 2..=16
//! over the rows' union (`w4a4_gemv_mx{8,16}_moe_union_sweep`, each union expert read once); the
//! slot GEMV (`w4a4_gemv_mx8_moe_slots`) where those kernels did not resolve;
//! the dense MLP the mx GEMVs in chunks of `DENSE_W4A4_CHUNK_ROWS` rows (2026-10-10: in
//! `w4a4_dense.rs`).
//! 2026-10-10: A routed down projection whose K (the expert width) is a multiple of 64 but not
//! of 128, a 64-unit expert slice of the `tp` layout, runs the `_k64` twins of the quantizer,
//! the slot GEMV and the sweep ([`down_kernels`]).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - A row's output does not depend on the row count or on the other rows: the quantizer is per
//!   row under a fixed global scale, the slot and union GEMVs compute each (row, slot) in its own MMA column, and every mx entry
//!   sums a row the same way.
//! - Launches only after the shape checks pass; never falls back to another format.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::Glm5NextMlpWorkspace;
use super::MOE_ROW_UNION_MAX_IDS;
use super::launch::swiglu;
use super::moe_experts::MoeSite;
use super::w4a4_slots::{W4a4SlotsCall, launch_w4a4_slots};
use crate::glm5next_layer::profile;
use crate::glm5next_mlp::build_w4a4::{check_w4a4_k, check_w4a4_k64};
use crate::glm5next_mlp::weights::{Glm5NextExpertPtrTable, W4a4ActScales};
use crate::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};

/// 2026-10-08: Rows of output one mx or slot block covers (`w4a4_gemv_mx.cu`, n0 = 16 * x).
pub(super) const W4A4_ROWS_PER_CTA: u32 = 16;
pub(super) const W4A4_BLOCK: u32 = 256;

/// 2026-10-08: Quantize `rows` BF16 rows of width `k` at `input` into the W4A4 scratch under the
/// static global scale `gs`. 2026-10-10: `k_store` is the row width `kern` writes: `k`, or
/// `k.next_multiple_of(128)` for the `_k64` quantizer.
#[allow(clippy::too_many_arguments)]
pub(super) fn quant_static(
    gpu: &dyn GpuBackend,
    kern: KernelHandle,
    ws: &Glm5NextMlpWorkspace,
    input: DevicePtr,
    rows: usize,
    k: usize,
    k_store: usize,
    gs: f32,
    stream: u64,
) -> Result<()> {
    ensure!(
        rows >= 1 && rows <= ws.w4a4_rows && k <= k_store && k_store <= ws.w4a4_k,
        "GLM W4A4: {rows} rows of {k} (stored {k_store} wide) do not fit the activation scratch \
         of {} x {}",
        ws.w4a4_rows,
        ws.w4a4_k
    );
    KernelLaunch::new(gpu, kern)
        .grid([rows as u32, 1, 1])
        .block([W4A4_BLOCK, 1, 1])
        .arg_ptr(input)
        .arg_ptr(ws.w4a4_aq)
        .arg_ptr(ws.w4a4_as)
        .arg_ptr(ws.w4a4_ag)
        .arg_u32(k as u32)
        .arg_f32(gs)
        .launch(stream)
}

/// 2026-10-08: `out[s, n]` for every slot `s < slots` over the quantized scratch, the weights of
/// expert `ids[s]` from `table`; slot `s` reads activation row `s / act_div`. 2026-10-14: Through
/// [`launch_w4a4_slots`], which splits past CUDA's grid-Y limit.
#[allow(clippy::too_many_arguments)]
fn moe_slots(
    gpu: &dyn GpuBackend,
    kern: KernelHandle,
    ws: &Glm5NextMlpWorkspace,
    table: &Glm5NextExpertPtrTable,
    out: DevicePtr,
    n: usize,
    k: usize,
    slots: usize,
    act_div: usize,
    num_experts: usize,
    stream: u64,
) -> Result<()> {
    let call = W4a4SlotsCall {
        aq: ws.w4a4_aq,
        as_: ws.w4a4_as,
        ag: ws.w4a4_ag,
        ids: ws.ids,
        packed_ptrs: table.packed_ptrs,
        scale_ptrs: table.scale_ptrs,
        scale2_vals: table.scale2_vals,
        out,
        n,
        k,
        slots,
        act_div,
        num_experts,
    };
    launch_w4a4_slots(gpu, kern, &call, stream)
}

/// 2026-10-09: Which entry list the persistent sweep reads.
#[derive(Clone, Copy)]
enum SweepEntries {
    /// 2026-10-09: One row's own slots: the router's ids row (`w4a4_gemv_mx8_moe_slots_sweep`).
    OwnSlots,
    /// 2026-10-09: The rows' union, which `glm5next_moe_row_union` builds first.
    Union,
}

/// 2026-10-10: The kernels of one routed W4A4 projection's input: its quantizer, the row width
/// that quantizer writes, the slot GEMV, the one-row own-slots sweep and the 8- and 16-token
/// union sweeps.
#[derive(Clone, Copy)]
struct ProjKernels {
    quant: KernelHandle,
    k_store: usize,
    slots: KernelHandle,
    own: KernelHandle,
    sweep: [KernelHandle; 2],
}

/// 2026-10-10: The routed down projection's kernels for expert width `mi`: the plain entries at
/// `mi % 128 == 0`, the `_k64` twins at `mi % 128 == 64`; an error for any other width, or when
/// the twins' quantizer or slot GEMV is absent from the target's PTX (an absent sweep runs the
/// slot GEMV).
fn down_kernels(k: &Glm5NextMlpKernels, mi: usize) -> Result<ProjKernels> {
    if mi.is_multiple_of(128) {
        check_w4a4_k(mi, "routed down")?;
        return Ok(ProjKernels {
            quant: k.w4a4_quant_static,
            k_store: mi,
            slots: k.w4a4_moe_slots,
            own: k.w4a4_moe_slots_sweep,
            sweep: k.w4a4_moe_sweep,
        });
    }
    check_w4a4_k64(mi, "routed down")?;
    let t = k.w4a4_moe_k64;
    ensure!(
        t.quant.0 != 0 && t.slots.0 != 0,
        "GLM routed W4A4: the expert width {mi} needs the _k64 quantizer and slot GEMV \
         (w4a4_gemv_mx_moe.cu), which this target's PTX lacks"
    );
    Ok(ProjKernels {
        quant: t.quant,
        k_store: mi.next_multiple_of(128),
        slots: t.slots,
        own: t.own,
        sweep: t.sweep,
    })
}

/// 2026-10-09: The persistent sweep for `rows` routed rows, when its kernels resolved: at one
/// row over the row's own slots (`own`); at 2..=16 rows (the 8- or 16-token entry of `sweep`)
/// over the union of rows whose `rows * top_k` ids the union builder covers. `None` runs the
/// slot GEMV. 2026-10-10: `own` and `sweep` are the projection's (plain or `_k64`).
fn sweep_kernel(
    k: &Glm5NextMlpKernels,
    own: KernelHandle,
    sweep: [KernelHandle; 2],
    rows: usize,
    top_k: usize,
) -> Option<(KernelHandle, SweepEntries)> {
    let (h, entries) = match rows {
        1 => (own, SweepEntries::OwnSlots),
        2..=8 => (sweep[0], SweepEntries::Union),
        9..=16 => (sweep[1], SweepEntries::Union),
        _ => return None,
    };
    let union_ok = match entries {
        SweepEntries::OwnSlots => true,
        SweepEntries::Union => k.moe_row_union.0 != 0 && rows * top_k <= MOE_ROW_UNION_MAX_IDS,
    };
    (h.0 != 0 && k.w4a4_sweep_ctas != 0 && union_ok).then_some((h, entries))
}

/// 2026-10-10: Whether `w4a4_experts` at `rows` rows writes every row of `expert_out` itself:
/// at one row when the down projection runs the own-slots sweep, which writes +0.0 over the
/// rows of a slot it computes nothing for (invalid or another rank's expert). `forward_moe` then
/// skips its zeroing of `expert_out`. A width the down kernels refuse reads false here and fails
/// in `w4a4_experts`.
pub(super) fn writes_every_expert_row(
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    rows: usize,
) -> bool {
    rows == 1
        && down_kernels(k, cfg.moe_intermediate).is_ok_and(|d| {
            matches!(
                sweep_kernel(k, d.own, d.sweep, rows, cfg.top_k),
                Some((_, SweepEntries::OwnSlots))
            )
        })
}

/// 2026-10-09: `out[r * top_k + s, n]` of every projection in `projs` (one or two, which read the
/// same activations) for every (row, slot) through the union tables (`glm5next_moe_row_union`),
/// each live union expert's weights read once, or at one row through the ids row itself, in one
/// launch of the persistent sweep.
fn moe_sweep(
    site: &MoeSite<'_>,
    (kern, entries): (KernelHandle, SweepEntries),
    projs: &[(&Glm5NextExpertPtrTable, DevicePtr)],
    n: usize,
    k: usize,
    act_div: usize,
) -> Result<()> {
    ensure!(
        matches!(projs.len(), 1 | 2),
        "GLM W4A4 union sweep: {} projections, the kernel takes one or two",
        projs.len()
    );
    let (ws, cfg) = (site.ws, site.cfg);
    // 2026-10-09: The own-slots entry reads the ids row as its entry list and not u_slot.
    let u_eid = match entries {
        SweepEntries::OwnSlots => ws.ids,
        SweepEntries::Union => ws.u_eid,
    };
    let mut launch = KernelLaunch::new(site.gpu, kern)
        .grid([site.k.w4a4_sweep_ctas, 1, 1])
        .block([W4A4_BLOCK, 1, 1])
        .arg_ptr(ws.w4a4_aq)
        .arg_ptr(ws.w4a4_as)
        .arg_ptr(ws.w4a4_ag)
        .arg_ptr(u_eid)
        .arg_ptr(ws.u_slot);
    // 2026-10-09: The kernel reads the second table only when nproj is 2; a single projection
    // passes its own table twice.
    for (table, out) in [projs[0], *projs.last().expect("checked non-empty")] {
        launch = launch
            .arg_ptr(table.packed_ptrs)
            .arg_ptr(table.scale_ptrs)
            .arg_ptr(table.scale2_vals)
            .arg_ptr(out);
    }
    launch
        .arg_u32(projs.len() as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(site.rows as u32)
        .arg_u32(cfg.top_k as u32)
        .arg_u32(act_div as u32)
        .arg_u32(cfg.num_experts as u32)
        .launch(site.stream)
}

/// 2026-10-08: The routed experts of `forward_moe` at W4A4, every (row, slot) in one launch per
/// projection: quantize the rows once, gate and up over all slots, the clamped SwiGLU, quantize
/// every slot's product, down into `expert_out`. A slot of another rank's expert writes nothing
/// (`forward_moe` zeroed `expert_out`).
pub(super) fn w4a4_experts(site: &MoeSite<'_>, scales: W4a4ActScales) -> Result<()> {
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
    // 2026-10-10: Wider than the slot GEMV serves: the grouped W4A4 GEMM.
    if super::w4a4_grouped::takes_grouped(rows) {
        return super::w4a4_grouped::w4a4_grouped_experts(site, scales);
    }
    let (mi, h) = (cfg.moe_intermediate, cfg.hidden);
    check_w4a4_k(h, "routed gate/up")?;
    let down = down_kernels(k, mi)?;
    let slots = rows * cfg.top_k;
    // 2026-10-08: Checked before the first launch: the down input holds one row per slot.
    ensure!(
        slots <= ws.w4a4_rows,
        "GLM routed W4A4: {rows} rows x top_k {} = {slots} slots do not fit the activation \
         scratch of {} rows",
        cfg.top_k,
        ws.w4a4_rows
    );
    let sweep = sweep_kernel(k, k.w4a4_moe_slots_sweep, k.w4a4_moe_sweep, rows, cfg.top_k);
    let down_sweep = sweep_kernel(k, down.own, down.sweep, rows, cfg.top_k);
    let t = profile::start();
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
    let union =
        |s: Option<(KernelHandle, SweepEntries)>| matches!(s, Some((_, SweepEntries::Union)));
    if union(sweep) || union(down_sweep) {
        // 2026-10-09: The union of the rows' experts, which every union launch below reads.
        KernelLaunch::new(gpu, k.moe_row_union)
            .grid([1, 1, 1])
            .block([slots as u32, 1, 1])
            .arg_ptr(ws.ids)
            .arg_ptr(ws.u_eid)
            .arg_ptr(ws.u_slot)
            .arg_u32(rows as u32)
            .arg_u32(cfg.top_k as u32)
            .launch(stream)?;
    }
    let slots_gemv = |kern, table: &Glm5NextExpertPtrTable, out, n, kk, act_div| {
        moe_slots(
            gpu,
            kern,
            ws,
            table,
            out,
            n,
            kk,
            slots,
            act_div,
            cfg.num_experts,
            stream,
        )
    };
    match sweep {
        Some(kern) => moe_sweep(
            site,
            kern,
            &[(&w.ptrs.gate, ws.a_gate), (&w.ptrs.up, ws.a_up)],
            mi,
            h,
            cfg.top_k,
        )?,
        None => {
            let kern = k.w4a4_moe_slots;
            slots_gemv(kern, &w.ptrs.gate, ws.a_gate, mi, h, cfg.top_k)?;
            slots_gemv(kern, &w.ptrs.up, ws.a_up, mi, h, cfg.top_k)?;
        }
    }
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
        down.quant,
        ws,
        ws.a_act,
        slots,
        mi,
        down.k_store,
        scales.down,
        stream,
    )?;
    match down_sweep {
        Some(kern) => moe_sweep(site, kern, &[(&w.ptrs.down, ws.expert_out)], h, mi, 1)?,
        None => slots_gemv(down.slots, &w.ptrs.down, ws.expert_out, h, mi, 1)?,
    }
    profile::end(profile::MOE_EXPERTS, t, gpu, stream);
    Ok(())
}

#[cfg(test)]
#[path = "w4a4_tests.rs"]
pub(super) mod tests;
