// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `METRALE_GLM_MOE_W4A4_GU_FUSED=1`: the grouped W4A4 routed experts
//! (`w4a4_grouped`) run gate, up, the clamped SwiGLU and the down input's static NVFP4
//! quantization as ONE launch (`w4a4_grouped_gu_m{32,64,128}`, the fused epilogue in
//! `kernels/gb10/common/w4a4_grouped_core.cuh`) instead of two grouped GEMMs, the SwiGLU and the
//! quantizer. The BF16 gate, up and SwiGLU values never reach global memory; the quantized down
//! input (codes, block scales, globals) on every live slot row of a local expert is byte for byte
//! what the separate launches write, so the down GEMM and everything after it see the same bytes
//! (the GPU gate is the `glm5next_moe_epi_bench` example).
//!
//! The down input lands in the gate, up and activated scratch (`a_gate`, `a_up`, `a_act`), not
//! in the W4A4 scratch the fused launch still reads its own input from.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Every check runs before the routed experts' first launch ([`GuPlan::new`]).
//! - Off unless the lever is on (`MlpLevers::gu_fused`); the lever needs `METRALE_GLM_MOE_W4A4_GROUPED=1` and the three
//!   fused entries, else the load is refused ([`check_at_load`]).

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::moe_experts::MoeSite;
use super::w4a4_grouped::{W4A4_GROUPED_COLS, W4A4_GROUPED_MT, grouped_launch, smem_bytes_of};
use super::w4a4_grouped_raster::GroupedEntry;
use crate::glm5next_mlp::Glm5NextMlpKernels;
use crate::glm5next_mlp::weights::Glm5NextExpertPtrTable;

const BLOCK: u32 = 256;
const LEVER: &str = "METRALE_GLM_MOE_W4A4_GU_FUSED";

/// 2026-10-14: Whether the lever can run on this load: `grouped` is
/// `METRALE_GLM_MOE_W4A4_GROUPED`, `ready` whether every fused entry resolved.
pub(crate) fn check_ready(grouped: bool, ready: bool) -> Result<(), String> {
    if !grouped {
        return Err(format!("{LEVER}=1 needs METRALE_GLM_MOE_W4A4_GROUPED=1"));
    }
    if !ready {
        return Err(format!(
            "{LEVER}=1: w4a4_grouped_gu_m32/_m64/_m128 is absent from this target's PTX"
        ));
    }
    Ok(())
}

/// 2026-10-14: The load-time refusal of a lever this load cannot run, from the routed experts'
/// precision plan.
pub fn check_at_load(k: &Glm5NextMlpKernels) -> Result<()> {
    if !k.levers.mlp.gu_fused {
        return Ok(());
    }
    check_ready(
        k.levers.mlp.w4a4_grouped,
        k.w4a4_grouped.gu.iter().all(|h| h.0 != 0),
    )
    .map_err(anyhow::Error::msg)
}

/// 2026-10-14: One routed site's fused launch, checked: the entry of tile `ti` and the down
/// input's stored width `k_store` (`mi`, or `mi` rounded up to 128 for the `_k64` down).
pub(super) struct GuPlan {
    entry: GroupedEntry,
    ti: usize,
    k_store: usize,
}

impl GuPlan {
    /// 2026-10-14: Refuses a site whose fused entry is absent, before anything launches. The
    /// shapes need no check of their own: `w4a4_grouped_experts` has already required
    /// `mi % 64 == 0` (the entry needs `mi % 16 == 0`) and `slots <= max_total_expanded`, and the
    /// down input (`slots * k_store / 2` code bytes, `k_store <= mi + 64`) then fits the activated
    /// scratch's `max_total_expanded * mi` BF16 values.
    /// 2026-10-14: `entry` is the tile's fused entry or its `_r` twin.
    pub(super) fn new(entry: GroupedEntry, ti: usize, k_store: usize) -> Result<Self> {
        ensure!(
            entry.kern.0 != 0,
            "{LEVER}=1: w4a4_grouped_gu (tile {ti}) is absent from this target's PTX"
        );
        Ok(Self { entry, ti, k_store })
    }

    /// 2026-10-14: The fused gate/up launch, then `down` (the unchanged down entry) over its
    /// output into `expert_out`; `gs` is the down projection's `input_scale`.
    pub(super) fn launch(&self, site: &MoeSite<'_>, down: GroupedEntry, gs: f32) -> Result<()> {
        let MoeSite {
            gpu,
            k,
            cfg,
            w,
            rows,
            ws,
            stream,
            ..
        } = *site;
        let (mi, h) = (cfg.moe_intermediate, cfg.hidden);
        let slots = rows * cfg.top_k;
        let call = W4a4GuCall {
            ti: self.ti,
            act: [ws.w4a4_aq, ws.w4a4_as, ws.w4a4_ag],
            gate: w.ptrs.gate,
            up: w.ptrs.up,
            out: [ws.a_gate, ws.a_up, ws.a_act],
            offsets: ws.expert_offsets(),
            tile_off: ws.w4a4_tile_off,
            gather: ws.sorted_token_ids(),
            num_experts: cfg.num_experts,
            slots,
            n: mi,
            k: h,
            k_store: self.k_store,
            gs,
            limit: cfg.swiglu_limit,
        };
        let entry = self.entry.over(&k.scale_dict, w.ptrs.gate.scale_bases)?;
        launch_w4a4_gu(gpu, entry, &call, stream)?;
        grouped_launch(
            gpu,
            down.over(&k.scale_dict, w.ptrs.down.scale_bases)?,
            self.ti,
            smem_bytes_of(self.ti),
            call.out,
            &w.ptrs.down,
            ws.expert_out,
            call.offsets,
            call.tile_off,
            DevicePtr::NULL,
            cfg.num_experts,
            slots,
            h,
            mi,
            stream,
        )
    }
}

/// 2026-10-14: One fused gate/up call on tile entry `ti` (`W4A4_GROUPED_MT`): the gate/up input
/// `act` (codes, scales, globals of the token rows), the gate and up tables, the down input `out`
/// (codes, scales and globals of every sorted slot row at width `k_store`), the sort's offsets,
/// the tile offsets, the gather map, `n` = the expert width, `k` = hidden, the down
/// `input_scale` `gs` and the SwiGLU `limit`.
#[derive(Clone, Copy, Debug)]
pub struct W4a4GuCall {
    pub ti: usize,
    pub act: [DevicePtr; 3],
    pub gate: Glm5NextExpertPtrTable,
    pub up: Glm5NextExpertPtrTable,
    pub out: [DevicePtr; 3],
    pub offsets: DevicePtr,
    pub tile_off: DevicePtr,
    pub gather: DevicePtr,
    pub num_experts: usize,
    pub slots: usize,
    pub n: usize,
    pub k: usize,
    pub k_store: usize,
    pub gs: f32,
    pub limit: f32,
}

/// 2026-10-14: The one launcher of `w4a4_grouped_gu_m*` (the forward and the byte-gate bench
/// both go through it): grid (ceil(n / (cols / 2)), ceil(slots / mt) + num_experts), the
/// one-projection entry's shared memory; 2026-10-14: an `_r` entry's raster group last.
/// 2026-10-21: a `_cs` entry (`GroupedEntry::over`) the gate's and up's base tables after it;
/// refused when the entry and either table disagree on them.
pub fn launch_w4a4_gu(
    gpu: &dyn GpuBackend,
    entry: GroupedEntry,
    c: &W4a4GuCall,
    stream: u64,
) -> Result<()> {
    ensure!(
        c.ti < W4A4_GROUPED_MT.len(),
        "w4a4_grouped_gu: tile entry {} out of range",
        c.ti
    );
    ensure!(
        entry.nc.is_none(),
        "w4a4_grouped_gu: the fused entry has no `_n` twin"
    );
    let bases = [c.gate.scale_bases, c.up.scale_bases];
    ensure!(
        bases.iter().all(|b| entry.cs != b.is_null()),
        "w4a4_grouped_gu: entry `_cs` = {} over gate / up base tables {bases:?}",
        entry.cs
    );
    let l = KernelLaunch::new(gpu, entry.kern)
        .grid([
            c.n.div_ceil(W4A4_GROUPED_COLS[c.ti] / 2) as u32,
            (c.slots.div_ceil(W4A4_GROUPED_MT[c.ti]) + c.num_experts) as u32,
            1,
        ])
        .block([BLOCK, 1, 1])
        .shared_mem(smem_bytes_of(c.ti))
        .arg_ptr(c.act[0])
        .arg_ptr(c.act[1])
        .arg_ptr(c.act[2])
        .arg_ptr(c.gate.packed_ptrs)
        .arg_ptr(c.gate.scale_ptrs)
        .arg_ptr(c.gate.scale2_vals)
        .arg_ptr(c.up.packed_ptrs)
        .arg_ptr(c.up.scale_ptrs)
        .arg_ptr(c.up.scale2_vals)
        .arg_ptr(c.out[0])
        .arg_ptr(c.out[1])
        .arg_ptr(c.out[2])
        .arg_ptr(c.offsets)
        .arg_ptr(c.tile_off)
        .arg_ptr(c.gather)
        .arg_u32(c.num_experts as u32)
        .arg_u32(c.n as u32)
        .arg_u32(c.k as u32)
        .arg_u32(c.k_store as u32)
        .arg_f32(c.gs)
        .arg_f32(c.limit);
    let l = match entry.raster {
        Some(g) => l.arg_u32(g),
        None => l,
    };
    if entry.cs {
        l.arg_ptr(bases[0]).arg_ptr(bases[1])
    } else {
        l
    }
    .launch(stream)
}
