// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Shapes, seeded weights and routings, and one launcher per kernel of the
//! `glm5next_moe_prefill_w4a4_bench` example.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::device::{Rng, routing, up, up_bf16};

pub const HIDDEN: usize = 4096;
pub const MI: usize = 704;
pub const KP: usize = 768;
pub const EXPERTS: usize = 288;
pub const TOP_K: usize = 8;
pub const GS_GATE_UP: f32 = 3.0 / 2688.0;
pub const GS_DOWN: f32 = 100.0 / 2688.0;
/// 2026-10-10: Slots and weight rows per tile of the three entries.
pub const MT: [usize; 4] = [32, 64, 128, 128];
pub const COLS: [u32; 4] = [128, 128, 64, 64];
pub const STAGES: [usize; 4] = [3, 3, 3, 6];

pub struct Kern {
    pub sort: KernelHandle,
    pub tiles: KernelHandle,
    pub quant: KernelHandle,
    pub quant_k64: KernelHandle,
    pub swiglu: KernelHandle,
    pub slots: KernelHandle,
    pub slots_k64: KernelHandle,
    pub mx: [KernelHandle; 4],
    pub mx_k64: [KernelHandle; 4],
    pub w4a16: KernelHandle,
}

/// 2026-10-10: One projection of all 288 experts and its global-id pointer table.
pub struct Table {
    pub packed: DevicePtr,
    pub scale: DevicePtr,
    pub scale2: DevicePtr,
}

pub fn table(gpu: &dyn GpuBackend, rng: &mut Rng, n: usize, k: usize) -> Result<Table> {
    let (pb, sb) = (n * k / 2, n * k / 16);
    let (packed_all, scale_all) = (gpu.alloc(pb * EXPERTS)?, gpu.alloc(sb * EXPERTS)?);
    let mut pp = Vec::with_capacity(EXPERTS * 8);
    let mut sp = Vec::with_capacity(EXPERTS * 8);
    for e in 0..EXPERTS {
        let packed: Vec<u8> = (0..pb).map(|_| rng.next() as u8).collect();
        let scales: Vec<u8> = (0..sb).map(|_| 0x30 + (rng.next() % 9) as u8).collect();
        let (p, s) = (packed_all.offset(e * pb), scale_all.offset(e * sb));
        gpu.copy_h2d(&packed, p)?;
        gpu.copy_h2d(&scales, s)?;
        pp.extend(p.0.to_le_bytes());
        sp.extend(s.0.to_le_bytes());
    }
    let s2: Vec<u8> = (0..EXPERTS).flat_map(|_| 0.01f32.to_le_bytes()).collect();
    Ok(Table {
        packed: up(gpu, &pp)?,
        scale: up(gpu, &sp)?,
        scale2: up(gpu, &s2)?,
    })
}

/// 2026-10-10: The buffers of one routing of `rows` rows.
pub struct Bufs {
    pub rows: usize,
    pub ids: DevicePtr,
    pub sorted: DevicePtr,
    pub sorted_e: DevicePtr,
    pub offsets: DevicePtr,
    pub perm: DevicePtr,
    pub tile_off: DevicePtr,
    pub x: DevicePtr,
    pub aq: DevicePtr,
    pub as_: DevicePtr,
    pub ag: DevicePtr,
    pub gate: DevicePtr,
    pub upo: DevicePtr,
    pub act: DevicePtr,
    pub out: DevicePtr,
    pub out_ref: DevicePtr,
}

pub fn bufs(gpu: &dyn GpuBackend, rng: &mut Rng, rows: usize) -> Result<Bufs> {
    let slots = rows * TOP_K;
    let ids: Vec<u8> = routing(rng, rows)
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let x: Vec<f32> = (0..rows * HIDDEN).map(|_| rng.unit() * 3.0).collect();
    let qrows = slots.max(rows);
    let qk = HIDDEN.max(KP);
    Ok(Bufs {
        rows,
        ids: up(gpu, &ids)?,
        sorted: gpu.alloc(slots * 4)?,
        sorted_e: gpu.alloc(slots * 4)?,
        offsets: gpu.alloc((EXPERTS + 1) * 4)?,
        perm: gpu.alloc(slots * 4)?,
        tile_off: gpu.alloc((EXPERTS + 1) * 4)?,
        x: up_bf16(gpu, &x)?,
        aq: gpu.alloc(qrows * qk / 2)?,
        as_: gpu.alloc(qrows * qk / 16)?,
        ag: gpu.alloc(qrows * 4)?,
        gate: gpu.alloc(slots * MI * 2)?,
        upo: gpu.alloc(slots * MI * 2)?,
        act: gpu.alloc(slots * MI * 2)?,
        out: gpu.alloc(slots * HIDDEN * 2)?,
        out_ref: gpu.alloc(slots * HIDDEN * 2)?,
    })
}

pub fn sort(gpu: &dyn GpuBackend, kn: &Kern, b: &Bufs, s: u64) -> Result<()> {
    KernelLaunch::new(gpu, kn.sort)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(b.ids)
        .arg_ptr(b.sorted)
        .arg_ptr(b.sorted_e)
        .arg_ptr(b.offsets)
        .arg_ptr(b.perm)
        .arg_u32((b.rows * TOP_K) as u32)
        .arg_u32(EXPERTS as u32)
        .arg_u32(TOP_K as u32)
        .launch(s)
}

pub fn tiles(gpu: &dyn GpuBackend, kn: &Kern, b: &Bufs, mt: usize, s: u64) -> Result<()> {
    KernelLaunch::new(gpu, kn.tiles)
        .grid([1, 1, 1])
        .block([32, 1, 1])
        .arg_ptr(b.offsets)
        .arg_ptr(b.tile_off)
        .arg_u32(EXPERTS as u32)
        .arg_u32(mt as u32)
        .launch(s)
}

#[allow(clippy::too_many_arguments)]
pub fn quant(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    b: &Bufs,
    x: DevicePtr,
    rows: usize,
    kk: usize,
    gs: f32,
    s: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([rows as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(b.aq)
        .arg_ptr(b.as_)
        .arg_ptr(b.ag)
        .arg_u32(kk as u32)
        .arg_f32(gs)
        .launch(s)
}

#[allow(clippy::too_many_arguments)]
pub fn grouped(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    ti: usize,
    b: &Bufs,
    t: &Table,
    c: DevicePtr,
    gather: DevicePtr,
    (n, kk): (usize, usize),
    s: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([
            div_ceil(n as u32, COLS[ti]),
            ((b.rows * TOP_K).div_ceil(MT[ti]) + EXPERTS) as u32,
            1,
        ])
        .shared_mem(
            metrale_model_arch::glm5next_mlp::forward::w4a4_grouped_smem_bytes(
                MT[ti],
                COLS[ti] as usize,
                STAGES[ti],
            ),
        )
        .block([256, 1, 1])
        .arg_ptr(b.aq)
        .arg_ptr(b.as_)
        .arg_ptr(b.ag)
        .arg_ptr(t.packed)
        .arg_ptr(t.scale)
        .arg_ptr(t.scale2)
        .arg_ptr(c)
        .arg_ptr(b.offsets)
        .arg_ptr(b.tile_off)
        .arg_ptr(gather)
        .arg_u32(EXPERTS as u32)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .launch(s)
}

/// 2026-10-10: The slot GEMV over every slot: `ids` gives each slot's expert, act_div the
/// activation row divisor.
#[allow(clippy::too_many_arguments)]
pub fn slots_gemv(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    b: &Bufs,
    t: &Table,
    c: DevicePtr,
    (n, kk, act_div): (usize, usize, usize),
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([div_ceil(n as u32, 16), (b.rows * TOP_K) as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(b.aq)
        .arg_ptr(b.as_)
        .arg_ptr(b.ag)
        .arg_ptr(b.ids)
        .arg_ptr(t.packed)
        .arg_ptr(t.scale)
        .arg_ptr(t.scale2)
        .arg_ptr(c)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .arg_u32(act_div as u32)
        .arg_u32(EXPERTS as u32)
        .launch(0)
}

/// 2026-10-10: The grouped W4A16 GEMM of the prefill before this lever, grid as
/// `forward_prefill_gemm::dispatch` sizes it from the real histogram.
#[allow(clippy::too_many_arguments)]
pub fn w4a16(
    gpu: &dyn GpuBackend,
    kn: &Kern,
    b: &Bufs,
    t: &Table,
    a: DevicePtr,
    c: DevicePtr,
    gather: DevicePtr,
    (n, kk): (usize, usize),
    m_tiles: u32,
    s: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kn.w4a16)
        .grid([div_ceil(n as u32, 64), m_tiles, EXPERTS as u32])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(t.packed)
        .arg_ptr(t.scale)
        .arg_ptr(t.scale2)
        .arg_ptr(c)
        .arg_ptr(b.offsets)
        .arg_ptr(gather)
        .arg_u32(EXPERTS as u32)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .launch(s)
}
