// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The W4A4 MLP forward of GLM-5.3, at the checkpoint's declared precision: NVFP4
//! weights times NVFP4 activations quantized under the static `input_scale`s, on the FP4
//! block-scale MMA. The routed experts run the slot GEMV (`w4a4_gemv_mx8_moe_slots`) at one row
//! and the persistent union sweep (`w4a4_gemv_mx{8,16}_moe_union_sweep`, each union expert read
//! once, gate and up in one launch) at 2..=16;
//! the dense MLP the mx GEMVs in chunks of [`DENSE_W4A4_CHUNK_ROWS`] rows.
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
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::Glm5NextMlpWorkspace;
use super::MOE_ROW_UNION_MAX_IDS;
use super::launch::swiglu;
use super::moe_experts::MoeSite;
use crate::glm5next_layer::profile;
use crate::glm5next_mlp::build_w4a4::{check_w4a4_k, check_w4a4_k64};
use crate::glm5next_mlp::weights::{
    Glm5NextDenseNvfp4Weights, Glm5NextExpertPtrTable, Nvfp4Proj, W4a4ActScales,
};
use crate::glm5next_mlp::{DENSE_W4A4_CHUNK_ROWS, Glm5NextMlpConfig, Glm5NextMlpKernels};

/// 2026-10-08: Rows of output one mx or slot block covers (`w4a4_gemv_mx.cu`, n0 = 16 * x).
const W4A4_ROWS_PER_CTA: u32 = 16;
const W4A4_BLOCK: u32 = 256;

/// 2026-10-08: Quantize `rows` BF16 rows of width `k` at `input` into the W4A4 scratch under the
/// static global scale `gs`. 2026-10-10: `k_store` is the row width `kern` writes: `k`, or
/// `k.next_multiple_of(128)` for the `_k64` quantizer.
#[allow(clippy::too_many_arguments)]
fn quant_static(
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
/// expert `ids[s]` from `table`; slot `s` reads activation row `s / act_div`.
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
    KernelLaunch::new(gpu, kern)
        .grid([div_ceil(n as u32, W4A4_ROWS_PER_CTA), slots as u32, 1])
        .block([W4A4_BLOCK, 1, 1])
        .arg_ptr(ws.w4a4_aq)
        .arg_ptr(ws.w4a4_as)
        .arg_ptr(ws.w4a4_ag)
        .arg_ptr(ws.ids)
        .arg_ptr(table.packed_ptrs)
        .arg_ptr(table.scale_ptrs)
        .arg_ptr(table.scale2_vals)
        .arg_ptr(out)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(act_div as u32)
        .arg_u32(num_experts as u32)
        .launch(stream)
}

/// 2026-10-10: The kernels of one routed W4A4 projection's input: its quantizer, the row width
/// that quantizer writes, the slot GEMV and the 8- and 16-token sweeps.
#[derive(Clone, Copy)]
struct ProjKernels {
    quant: KernelHandle,
    k_store: usize,
    slots: KernelHandle,
    sweep: [KernelHandle; 2],
}

/// 2026-10-10: The routed down projection's kernels for expert width `mi`: the plain entries at
/// `mi % 128 == 0`, the `_k64` twins at `mi % 128 == 64`; an error for any other width, or when
/// the twins are absent from the target's PTX.
fn down_kernels(k: &Glm5NextMlpKernels, mi: usize) -> Result<ProjKernels> {
    if mi.is_multiple_of(128) {
        check_w4a4_k(mi, "routed down")?;
        return Ok(ProjKernels {
            quant: k.w4a4_quant_static,
            k_store: mi,
            slots: k.w4a4_moe_slots,
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
        sweep: t.sweep,
    })
}

/// 2026-10-09: The union sweep for `rows` routed rows: the 8- or 16-token entry for 2..=16 rows
/// whose `rows * top_k` ids the union builder covers, when those kernels resolved; `None` (the
/// slot GEMV) for one row, where the union is the row's own slots, or otherwise.
/// 2026-10-10: `sweep` is the projection's pair (plain or `_k64`).
fn sweep_kernel(
    k: &Glm5NextMlpKernels,
    sweep: [KernelHandle; 2],
    rows: usize,
    top_k: usize,
) -> Option<KernelHandle> {
    let h = match rows {
        2..=8 => sweep[0],
        9..=16 => sweep[1],
        _ => return None,
    };
    (h.0 != 0
        && k.moe_row_union.0 != 0
        && k.w4a4_sweep_ctas != 0
        && rows * top_k <= MOE_ROW_UNION_MAX_IDS)
        .then_some(h)
}

/// 2026-10-09: `out[r * top_k + s, n]` of every projection in `projs` (one or two, which read the
/// same activations) for every (row, slot) through the union tables (`glm5next_moe_row_union`),
/// each live union expert's weights read once, in one launch of the persistent sweep.
fn moe_sweep(
    site: &MoeSite<'_>,
    kern: KernelHandle,
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
    let mut launch = KernelLaunch::new(site.gpu, kern)
        .grid([site.k.w4a4_sweep_ctas, 1, 1])
        .block([W4A4_BLOCK, 1, 1])
        .arg_ptr(ws.w4a4_aq)
        .arg_ptr(ws.w4a4_as)
        .arg_ptr(ws.w4a4_ag)
        .arg_ptr(ws.u_eid)
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

/// 2026-10-08: The mx entry for `rows` (1..=32): mx8, mx16 or mx32.
fn mx_kernel(k: &Glm5NextMlpKernels, rows: usize) -> KernelHandle {
    match rows {
        0..=8 => k.w4a4_mx[0],
        9..=16 => k.w4a4_mx[1],
        _ => k.w4a4_mx[2],
    }
}

/// 2026-10-08: `out[rows, n]` = the quantized scratch times `w` (`[n, k]` NVFP4).
#[allow(clippy::too_many_arguments)]
fn mx_proj(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    ws: &Glm5NextMlpWorkspace,
    w: &Nvfp4Proj,
    out: DevicePtr,
    rows: usize,
    n: usize,
    kk: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, mx_kernel(k, rows))
        .grid([div_ceil(n as u32, W4A4_ROWS_PER_CTA), 1, 1])
        .block([W4A4_BLOCK, 1, 1])
        .arg_ptr(ws.w4a4_aq)
        .arg_ptr(ws.w4a4_as)
        .arg_ptr(ws.w4a4_ag)
        .arg_ptr(w.packed)
        .arg_ptr(w.scale)
        .arg_f32(w.scale_2)
        .arg_ptr(out)
        .arg_u32(rows as u32)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .launch(stream)
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
    let union = sweep_kernel(k, k.w4a4_moe_sweep, rows, cfg.top_k);
    let down_union = sweep_kernel(k, down.sweep, rows, cfg.top_k);
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
    if union.is_some() || down_union.is_some() {
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
    match union {
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
    match down_union {
        Some(kern) => moe_sweep(site, kern, &[(&w.ptrs.down, ws.expert_out)], h, mi, 1)?,
        None => slots_gemv(down.slots, &w.ptrs.down, ws.expert_out, h, mi, 1)?,
    }
    profile::end(profile::MOE_EXPERTS, t, gpu, stream);
    Ok(())
}

/// 2026-10-08: The dense SwiGLU MLP of width `inter` at W4A4, `m` rows in chunks of
/// [`DENSE_W4A4_CHUNK_ROWS`]: gate and up per chunk from one quantization, the clamped SwiGLU
/// over all rows, then down per chunk. `out` is a partial sum under TP, as in `forward_dense`.
#[allow(clippy::too_many_arguments)]
pub fn forward_dense_w4a4(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextDenseNvfp4Weights,
    scales: W4a4ActScales,
    inter: usize,
    x: DevicePtr,
    out: DevicePtr,
    m: usize,
    ws: &Glm5NextMlpWorkspace,
    stream: u64,
) -> Result<()> {
    let h = cfg.hidden;
    ensure!(
        m >= 1 && m <= ws.max_rows() && inter <= ws.max_inter,
        "GLM dense W4A4: {m} rows x {inter} do not fit the workspace"
    );
    check_w4a4_k(h, "dense gate/up")?;
    check_w4a4_k(inter, "dense down")?;
    let chunks = (0..m)
        .step_by(DENSE_W4A4_CHUNK_ROWS)
        .map(|r0| (r0, (m - r0).min(DENSE_W4A4_CHUNK_ROWS)));
    for (r0, n) in chunks.clone() {
        quant_static(
            gpu,
            k.w4a4_quant_static,
            ws,
            x.offset(r0 * h * 2),
            n,
            h,
            h,
            scales.gate_up,
            stream,
        )?;
        mx_proj(
            gpu,
            k,
            ws,
            &w.gate_proj,
            ws.a_gate.offset(r0 * inter * 2),
            n,
            inter,
            h,
            stream,
        )?;
        mx_proj(
            gpu,
            k,
            ws,
            &w.up_proj,
            ws.a_up.offset(r0 * inter * 2),
            n,
            inter,
            h,
            stream,
        )?;
    }
    swiglu(
        gpu,
        k.swiglu,
        ws.a_gate,
        ws.a_up,
        ws.a_act,
        m * inter,
        cfg.swiglu_limit,
        stream,
    )?;
    for (r0, n) in chunks {
        quant_static(
            gpu,
            k.w4a4_quant_static,
            ws,
            ws.a_act.offset(r0 * inter * 2),
            n,
            inter,
            inter,
            scales.down,
            stream,
        )?;
        mx_proj(
            gpu,
            k,
            ws,
            &w.down_proj,
            out.offset(r0 * h * 2),
            n,
            h,
            inter,
            stream,
        )?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "w4a4_tests.rs"]
mod tests;
