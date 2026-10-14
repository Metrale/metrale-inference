// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The arms of `glm5next_kda_chunk64_conv_parity.rs`,
//! `glm5next_kda_gate_in_prepare_parity.rs` and `glm5next_kda_onorm_in_scan_parity.rs`: the unfused chunk64 path (the conv of every row, the
//! window advance, a prepare and a scan per slab) and the fused one (the conv inside each slab's
//! prepare, the window advance last), launched on one arm's buffers, with each slab's factors
//! recorded on request; the prepares read the FP32 gate rows or (the `_graw` entries) compute the
//! gate from f_b's BF16 rows; the scan is `Kernels::scan`, or (`Kernels::norm` set)
//! `kda_chunk64_scan_onorm` with its o_norm arguments.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: the launches take each kernel's arguments in its signature's order
//! (`kda_conv_tokens.cu`, `kda_chunk64_prepare.cu`, `kda_chunk64.cu`).
#![allow(dead_code)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_kda::chunk64::{
    CHUNK64 as C, CHUNK64_AD_STRIDE, CHUNK64_D as D, CHUNK64_PREP_SMEM, CHUNK64_PREP_THREADS,
    CHUNK64_SCAN_SMEM, CHUNK64_SCAN_THREADS, CHUNK64_SLAB as SLAB, CHUNK64_VB,
};

/// 2026-10-14: Per-rank KDA heads at TP=3, conv width 4, the config's L2 eps.
pub const H: usize = 22;
pub const QKV: usize = H * D;
pub const CD: usize = 3 * QKV;
pub const KC: usize = 4;
const EPS: f32 = 1e-6;
/// 2026-10-14: Row spans of the parts conv; its bytes do not depend on them (its own gate).
const SPANS: u32 = 48;

#[derive(Clone, Copy)]
pub struct Kernels {
    pub conv: KernelHandle,
    pub parts: KernelHandle,
    pub window: KernelHandle,
    pub window_parts: KernelHandle,
    pub prep: KernelHandle,
    pub prep_conv: KernelHandle,
    pub scan: KernelHandle,
    /// 2026-10-14: When set, each slab's scan is this entry instead of `scan`.
    pub norm: Option<ScanNorm>,
}

/// 2026-10-14: `kda_chunk64_scan_onorm` and its six arguments after the scan's own
/// (`kda_chunk64.cu`).
#[derive(Clone, Copy)]
pub struct ScanNorm {
    pub scan: KernelHandle,
    pub gate: DevicePtr,
    pub weight: DevicePtr,
    pub out: DevicePtr,
    pub part: DevicePtr,
    pub count: DevicePtr,
    pub eps: f32,
}

impl Kernels {
    /// 2026-10-14: This tree's kernels, from the embedded modules.
    pub fn load(g: &dyn GpuBackend) -> Result<Self> {
        Ok(Kernels {
            conv: g.kernel("kda_conv_tokens", "causal_conv1d_update_l2norm_tokens")?,
            parts: g.kernel(
                "kda_conv_tokens",
                "causal_conv1d_update_l2norm_tokens_parts",
            )?,
            window: g.kernel("kda_conv_tokens", "causal_conv1d_window_advance")?,
            window_parts: g.kernel("kda_conv_tokens", "causal_conv1d_window_advance_parts")?,
            prep: g.kernel("kda_chunk64_prepare", "kda_chunk64_prepare")?,
            prep_conv: g.kernel("kda_chunk64_prepare", "kda_chunk64_prepare_conv")?,
            scan: g.kernel("kda_chunk64", "kda_chunk64_scan")?,
            norm: None,
        })
    }
}

/// 2026-10-14: `kda_gate_bf16`'s inputs: f_b's BF16 rows `[T, QKV]`, `dt_bias` `[QKV]`, `A_log`
/// `[H]` and the lower bound.
#[derive(Clone, Copy)]
pub struct GateRaw {
    pub g_raw: DevicePtr,
    pub dt_bias: DevicePtr,
    pub a_log: DevicePtr,
    pub lower_bound: f32,
}

/// 2026-10-14: What a prepare reads for the gate: the FP32 rows `In::gate`, or (a `_graw`
/// prepare) the gate's inputs.
#[derive(Clone, Copy)]
pub enum Gate {
    Rows,
    Raw(GateRaw),
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Layout {
    /// 2026-10-14: The packed rows `[T, CD]` (q | k | v per row).
    Packed,
    /// 2026-10-14: The three projections, `[T, QKV]` each, back to back.
    Parts,
}

/// 2026-10-14: One case's inputs, shared by every arm.
pub struct In {
    pub t: usize,
    pub packed: DevicePtr,
    pub parts: DevicePtr,
    pub weight: DevicePtr,
    pub gate: DevicePtr,
    pub beta: DevicePtr,
    pub conv0: Vec<f32>,
    pub rec0: Vec<f32>,
}

impl In {
    /// 2026-10-14: q, k, v and their row stride in `layout`, from `base` (a copy of that layout's
    /// buffer for the control).
    pub fn xs(&self, layout: Layout, base: DevicePtr) -> [(DevicePtr, usize); 3] {
        match layout {
            Layout::Packed => [0, 1, 2].map(|i| (base.offset(i * QKV * 2), CD)),
            Layout::Parts => [0, 1, 2].map(|i| (base.offset(i * self.t * QKV * 2), QKV)),
        }
    }
    pub fn src(&self, layout: Layout) -> DevicePtr {
        match layout {
            Layout::Packed => self.packed,
            Layout::Parts => self.parts,
        }
    }
}

/// 2026-10-14: One arm's outputs and state.
pub struct Arm {
    pub conv: DevicePtr,
    pub rec: DevicePtr,
    pub core: DevicePtr,
    /// 2026-10-14: The conv output `[T, CD]` (unfused) or the slab stage (FUSED).
    pub cout: DevicePtr,
    /// 2026-10-14: The parts conv's packed copy (unfused, [`Layout::Parts`]).
    pub packed: DevicePtr,
    pub scratch: [DevicePtr; 6],
}

/// 2026-10-14: Bytes of each factor buffer for `n` chunks (`kda_chunk64.cuh`).
fn scratch_bytes(n: usize) -> [usize; 6] {
    let (cb, cd) = (n * H, C * D);
    [
        cb * cd * 4,
        cb * cd * 2,
        cb * cd * 2,
        cb * cd * 2,
        cb * C * C * 2,
        cb * CHUNK64_AD_STRIDE * 4,
    ]
}

impl Arm {
    pub fn new(g: &dyn GpuBackend, x: &In) -> Result<Self> {
        let t = x.t;
        let mut scratch = [DevicePtr::NULL; 6];
        for (p, n) in scratch.iter_mut().zip(scratch_bytes(SLAB)) {
            *p = g.alloc(n)?;
        }
        let a = Arm {
            conv: g.alloc(CD * KC * 4)?,
            rec: g.alloc(H * D * D * 4)?,
            core: g.alloc(t * QKV * 4)?,
            cout: g.alloc(t * CD * 2)?,
            packed: g.alloc(t * CD * 2)?,
            scratch,
        };
        a.reset(g, x)?;
        Ok(a)
    }
    pub fn reset(&self, g: &dyn GpuBackend, x: &In) -> Result<()> {
        let b = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
        g.copy_h2d(&b(&x.conv0), self.conv)?;
        g.copy_h2d(&b(&x.rec0), self.rec)?;
        g.memset(self.core, 0, x.t * QKV * 4)?;
        g.memset(self.cout, 0, x.t * CD * 2)?;
        g.memset(self.packed, 0, x.t * CD * 2)?;
        for (p, n) in self.scratch.iter().zip(scratch_bytes(SLAB)) {
            g.memset(*p, 0, n)?;
        }
        Ok(())
    }
    pub fn free(self, g: &dyn GpuBackend) -> Result<()> {
        for p in [self.conv, self.rec, self.core, self.cout, self.packed]
            .into_iter()
            .chain(self.scratch)
        {
            g.free(p)?;
        }
        Ok(())
    }
}

fn read(g: &dyn GpuBackend, s: u64, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    g.synchronize(s)?;
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-10-14: What a run leaves: core, recurrent state, conv state, and (when recorded) each
/// slab's factors.
pub struct Run {
    pub core: Vec<u8>,
    pub rec: Vec<u8>,
    pub conv: Vec<u8>,
    pub slabs: Vec<[Vec<u8>; 6]>,
}

fn snap(g: &dyn GpuBackend, s: u64, a: &Arm, n: usize) -> Result<[Vec<u8>; 6]> {
    let sz = scratch_bytes(n);
    Ok([
        read(g, s, a.scratch[0], sz[0])?,
        read(g, s, a.scratch[1], sz[1])?,
        read(g, s, a.scratch[2], sz[2])?,
        read(g, s, a.scratch[3], sz[3])?,
        read(g, s, a.scratch[4], sz[4])?,
        read(g, s, a.scratch[5], sz[5])?,
    ])
}

pub fn finish(
    g: &dyn GpuBackend,
    s: u64,
    x: &In,
    a: &Arm,
    slabs: Vec<[Vec<u8>; 6]>,
) -> Result<Run> {
    Ok(Run {
        core: read(g, s, a.core, x.t * QKV * 4)?,
        rec: read(g, s, a.rec, H * D * D * 4)?,
        conv: read(g, s, a.conv, CD * KC * 4)?,
        slabs,
    })
}

fn slabs(t: usize) -> impl Iterator<Item = (usize, usize)> {
    let nc = t.div_ceil(C);
    (0..nc).step_by(SLAB).map(move |c0| (c0, SLAB.min(nc - c0)))
}

fn scan(
    g: &dyn GpuBackend,
    k: &Kernels,
    x: &In,
    a: &Arm,
    c0: usize,
    n: usize,
    s: u64,
) -> Result<()> {
    let mut l = KernelLaunch::new(g, k.norm.map_or(k.scan, |n| n.scan))
        .grid([H as u32, (D / CHUNK64_VB) as u32, 1])
        .block([CHUNK64_SCAN_THREADS, 1, 1])
        .shared_mem(CHUNK64_SCAN_SMEM);
    for p in a.scratch {
        l = l.arg_ptr(p);
    }
    let l = l
        .arg_ptr(a.rec)
        .arg_ptr(a.core)
        .arg_u32(H as u32)
        .arg_u32(x.t as u32)
        .arg_u32(c0 as u32)
        .arg_u32(n as u32)
        .arg_u32(QKV as u32);
    match k.norm {
        None => l,
        Some(o) => l
            .arg_ptr(o.gate)
            .arg_ptr(o.weight)
            .arg_ptr(o.out)
            .arg_ptr(o.part)
            .arg_ptr(o.count)
            .arg_f32(o.eps),
    }
    .launch(s)
}

/// 2026-10-14: The prepare's trailing arguments (gate .. scale), shared by every entry.
fn prep_tail<'a>(l: KernelLaunch<'a>, x: &In, a: &Arm, c0: usize, gate: Gate) -> KernelLaunch<'a> {
    let l = match gate {
        Gate::Rows => l.arg_ptr(x.gate),
        Gate::Raw(r) => l
            .arg_ptr(r.g_raw)
            .arg_ptr(r.dt_bias)
            .arg_ptr(r.a_log)
            .arg_f32(r.lower_bound),
    };
    let mut l = l.arg_ptr(x.beta);
    for p in a.scratch {
        l = l.arg_ptr(p);
    }
    l.arg_u32(H as u32)
        .arg_u32(x.t as u32)
        .arg_u32(c0 as u32)
        .arg_u32(CD as u32)
        .arg_u32(QKV as u32)
        .arg_f32(1.0 / (D as f32).sqrt())
}

fn launch(g: &dyn GpuBackend, k: KernelHandle, grid: [u32; 3]) -> KernelLaunch<'_> {
    KernelLaunch::new(g, k).grid(grid).block([256, 1, 1])
}

/// 2026-10-14: The unfused path: the conv of every row, the window advance, then a prepare
/// (`prep`, reading `gate`) and a scan per slab.
#[allow(clippy::too_many_arguments)]
pub fn unfused(
    g: &dyn GpuBackend,
    k: &Kernels,
    prep: KernelHandle,
    gate: Gate,
    x: &In,
    a: &Arm,
    layout: Layout,
    s: u64,
    mut rec: Option<&mut Vec<[Vec<u8>; 6]>>,
) -> Result<()> {
    let (t, gx) = (x.t, CD.div_ceil(256) as u32);
    let packed = match layout {
        Layout::Packed => {
            launch(g, k.conv, [gx, t as u32, 1])
                .arg_ptr(a.conv)
                .arg_ptr(x.packed)
                .arg_ptr(x.weight)
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(a.cout)
                .arg_u32(CD as u32)
                .arg_u32(KC as u32)
                .arg_u32((2 * QKV) as u32)
                .arg_u32(D as u32)
                .arg_f32(EPS)
                .launch(s)?;
            x.packed
        }
        Layout::Parts => {
            let [(q, _), (kk, _), (v, _)] = x.xs(layout, x.parts);
            launch(g, k.parts, [gx, SPANS, 1])
                .arg_ptr(a.conv)
                .arg_ptr(q)
                .arg_ptr(kk)
                .arg_ptr(v)
                .arg_ptr(x.weight)
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(a.packed)
                .arg_ptr(a.cout)
                .arg_u32(CD as u32)
                .arg_u32(QKV as u32)
                .arg_u32(t as u32)
                .arg_u32(KC as u32)
                .arg_u32((2 * QKV) as u32)
                .arg_u32(D as u32)
                .arg_f32(EPS)
                .launch(s)?;
            a.packed
        }
    };
    launch(g, k.window, [gx, 1, 1])
        .arg_ptr(a.conv)
        .arg_ptr(packed)
        .arg_u32(CD as u32)
        .arg_u32(KC as u32)
        .arg_u32(t as u32)
        .launch(s)?;
    for (c0, n) in slabs(t) {
        let l = KernelLaunch::new(g, prep)
            .grid([n as u32, H as u32, 1])
            .block([CHUNK64_PREP_THREADS, 1, 1])
            .shared_mem(CHUNK64_PREP_SMEM)
            .arg_ptr(a.cout)
            .arg_ptr(a.cout.offset(QKV * 2))
            .arg_ptr(a.cout.offset(2 * QKV * 2));
        prep_tail(l, x, a, c0, gate).launch(s)?;
        if let Some(r) = rec.as_deref_mut() {
            r.push(snap(g, s, a, n)?);
        }
        scan(g, k, x, a, c0, n, s)?;
    }
    Ok(())
}

/// 2026-10-14: The fused path (`prep`, a conv prepare reading `gate`) on inputs `src` (the
/// layout's buffer or the control's copy).
#[allow(clippy::too_many_arguments)]
pub fn fused(
    g: &dyn GpuBackend,
    k: &Kernels,
    prep: KernelHandle,
    gate: Gate,
    x: &In,
    a: &Arm,
    layout: Layout,
    src: DevicePtr,
    s: u64,
    mut rec: Option<&mut Vec<[Vec<u8>; 6]>>,
) -> Result<()> {
    let [(q, stride), (kk, _), (v, _)] = x.xs(layout, src);
    for (c0, n) in slabs(x.t) {
        let l = KernelLaunch::new(g, prep)
            .grid([n as u32, H as u32, 1])
            .block([CHUNK64_PREP_THREADS, 1, 1])
            .shared_mem(CHUNK64_PREP_SMEM)
            .arg_ptr(q)
            .arg_ptr(kk)
            .arg_ptr(v)
            .arg_u32(stride as u32)
            .arg_ptr(a.conv)
            .arg_ptr(x.weight)
            .arg_ptr(DevicePtr::NULL)
            .arg_u32(KC as u32)
            .arg_f32(EPS)
            .arg_ptr(a.cout);
        prep_tail(l, x, a, c0, gate).launch(s)?;
        if let Some(r) = rec.as_deref_mut() {
            r.push(snap(g, s, a, n)?);
        }
        scan(g, k, x, a, c0, n, s)?;
    }
    launch(g, k.window_parts, [CD.div_ceil(256) as u32, 1, 1])
        .arg_ptr(a.conv)
        .arg_ptr(q)
        .arg_ptr(kk)
        .arg_ptr(v)
        .arg_u32(CD as u32)
        .arg_u32(QKV as u32)
        .arg_u32(stride as u32)
        .arg_u32(KC as u32)
        .arg_u32(x.t as u32)
        .launch(s)
}
