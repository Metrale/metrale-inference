// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Byte gate and microbench of GLM-5.3's routed experts at narrow decode widths on
//! the declared W4A4 path (one rank of the TP=3 / EP=3 serve: hidden 4096, expert width 2048,
//! 288 experts of which this rank holds 0..96, top 8): one row (C1 decode) and 2..=8 rows of
//! one sequence (a speculative verify).
//!
//! 1. Byte gate at one row: every output byte of the own-slots sweep
//!    (`w4a4_gemv_mx8_moe_slots_sweep`, gate and up fused, then down) against the slot GEMV
//!    `w4a4_gemv_mx8_moe_slots`, on seeded rows of 0..=8 local experts, a repeated id and
//!    out-of-range ids, and on every row of the routing fixture when one is given; plus a
//!    negative control (a changed expert must change the bytes).
//! 2. Timing (CUDA-graph replay, `timing.rs`): one row per local-expert count and on the
//!    fixture's rows, slot GEMV vs union sweep vs own-slots sweep; 2, 4 and 8 rows of one
//!    sequence at a local union of 3, 5 and 8 experts and on fixture windows.
//! 1b. The same at a 704-wide expert slice (the 64-unit `tp` layout, `k64.rs`): gate and up on
//!    the plain own-slots sweep, down on its `_k64` twin against the `_k64` slot GEMV.
//! 3. With a fixture: `forward_moe` at one row under the fixture's router on its router inputs,
//!    the own-slots sweep against the slot GEMV (the kernel table with the sweep unresolved),
//!    byte for byte on every token, and both timed.
//!
//! The routing fixture (optional) is `GLM_ROUTING_FIXTURE=<prefix>`: `<prefix>.ids` (i32
//! `[layers][tokens][8]`), `.router` (f32 `[layers][288][4096]`), `.bias` (f32
//! `[layers][288]`) and `.hidden` (f32 `[layers][tokens][4096]`, the router input); the ids are
//! the router's top 8 of those inputs.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: exits with an error on the first byte mismatch.
//!
//! ```text
//! METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//! GLM_BENCH_GPU_ORDINAL=0 cargo run -p metrale-model-arch --release \
//!     --example glm5next_moe_narrow_bench --features cuda,gpu-examples
//! ```

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use metrale_model_arch::glm5next_mlp::W4A4_SWEEP_CTAS_PER_SM;

#[allow(dead_code)]
#[path = "../glm5next_moe_wide_bench/device.rs"]
mod device;
use device::*;

mod forward;
mod k64;
mod timing;

pub struct Setup {
    pub kn: Kern,
    pub own: KernelHandle,
    pub gate: Table,
    pub upt: Table,
    pub down: Table,
    pub x_act: Act,
    pub d_act: Act,
    pub out: [DevicePtr; 3],
}

/// 2026-10-09: The routing fixture: `layers` x `tokens` rows of top-8 ids and what made them.
pub struct Fixture {
    pub layers: usize,
    pub tokens: usize,
    pub ids: Vec<i32>,
    pub router: Vec<f32>,
    pub bias: Vec<f32>,
    pub hidden: Vec<f32>,
}

fn read_le<T: Copy>(path: &str, from: fn([u8; 4]) -> T) -> Result<Vec<T>> {
    let b = std::fs::read(path).with_context(|| format!("reading {path}"))?;
    ensure!(b.len() % 4 == 0, "{path}: {} bytes", b.len());
    Ok(b.chunks_exact(4)
        .map(|c| from(c.try_into().expect("4 bytes")))
        .collect())
}

fn fixture() -> Result<Option<Fixture>> {
    let Ok(p) = std::env::var("GLM_ROUTING_FIXTURE") else {
        return Ok(None);
    };
    let ids = read_le(&format!("{p}.ids"), i32::from_le_bytes)?;
    let router = read_le(&format!("{p}.router"), f32::from_le_bytes)?;
    let bias = read_le(&format!("{p}.bias"), f32::from_le_bytes)?;
    let hidden = read_le(&format!("{p}.hidden"), f32::from_le_bytes)?;
    let layers = bias.len() / EXPERTS;
    ensure!(
        layers >= 1 && bias.len() == layers * EXPERTS && ids.len() % (layers * TOP_K) == 0,
        "fixture {p}: {} bias values, {} ids",
        bias.len(),
        ids.len()
    );
    let tokens = ids.len() / (layers * TOP_K);
    ensure!(
        router.len() == layers * EXPERTS * HIDDEN && hidden.len() == layers * tokens * HIDDEN,
        "fixture {p}: router or hidden size does not match {layers} layers x {tokens} tokens"
    );
    Ok(Some(Fixture {
        layers,
        tokens,
        ids,
        router,
        bias,
        hidden,
    }))
}

/// 2026-10-09: One row of `TOP_K` distinct experts, exactly `local` of them on this rank.
pub fn row_with_local(rng: &mut Rng, local: usize) -> Vec<i32> {
    let mut row: Vec<i32> = Vec::with_capacity(TOP_K);
    while row.len() < TOP_K {
        let e = if row.len() < local {
            (rng.next() % LOCAL as u64) as i32
        } else {
            (LOCAL as u64 + rng.next() % (EXPERTS - LOCAL) as u64) as i32
        };
        if !row.contains(&e) {
            row.push(e);
        }
    }
    row
}

pub fn route_of(gpu: &dyn GpuBackend, host: Vec<i32>) -> Result<Route> {
    let rows = host.len() / TOP_K;
    let b: Vec<u8> = host.iter().flat_map(|x| x.to_le_bytes()).collect();
    Ok(Route {
        rows,
        ids: up(gpu, &b)?,
        u_eid: gpu.alloc(rows * TOP_K * 4)?,
        u_slot: gpu.alloc(rows * TOP_K * rows * 4)?,
        host,
    })
}

/// 2026-10-09: The own-slots sweep `kern` (plain or `_k64`) over one or two projections: the ids row is the entry list
/// (u_slot is passed and not read).
#[allow(clippy::too_many_arguments)]
pub fn own_sweep(
    gpu: &dyn GpuBackend,
    kern: KernelHandle,
    ctas: u32,
    a: &Act,
    r: &Route,
    ts: &[&Table],
    outs: &[DevicePtr],
    (n, k, act_div): (usize, usize, usize),
    st: u64,
) -> Result<()> {
    let mut l = KernelLaunch::new(gpu, kern)
        .grid([ctas, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a.aq)
        .arg_ptr(a.as_)
        .arg_ptr(a.ag)
        .arg_ptr(r.ids)
        .arg_ptr(r.u_slot);
    for i in 0..2 {
        let (t, o) = (ts.get(i).unwrap_or(&ts[0]), outs.get(i).unwrap_or(&outs[0]));
        l = l
            .arg_ptr(t.packed)
            .arg_ptr(t.scale)
            .arg_ptr(t.scale2)
            .arg_ptr(*o);
    }
    l.arg_u32(ts.len() as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(r.rows as u32)
        .arg_u32(TOP_K as u32)
        .arg_u32(act_div as u32)
        .arg_u32(EXPERTS as u32)
        .launch(st)
}

/// 2026-10-09: `device::slots_gemv` with kernel `kern` (plain or `_k64`) on stream `st` (that one
/// launches the plain kernel on stream 0).
#[allow(clippy::too_many_arguments)]
pub fn slots_on(
    gpu: &dyn GpuBackend,
    kern: KernelHandle,
    a: &Act,
    r: &Route,
    t: &Table,
    out: DevicePtr,
    (n, k, act_div): (usize, usize, usize),
    st: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kern)
        .grid([div_ceil(n as u32, 16), (r.rows * TOP_K) as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(a.aq)
        .arg_ptr(a.as_)
        .arg_ptr(a.ag)
        .arg_ptr(r.ids)
        .arg_ptr(t.packed)
        .arg_ptr(t.scale)
        .arg_ptr(t.scale2)
        .arg_ptr(out)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(act_div as u32)
        .arg_u32(EXPERTS as u32)
        .launch(st)
}

pub const GU: (usize, usize, usize) = (MI, HIDDEN, TOP_K);
pub const DN: (usize, usize, usize) = (HIDDEN, MI, 1);

/// 2026-10-09: Every output byte of one row: gate and up of one fused own-slots sweep and down
/// of another, each against the slot GEMV. Returns whether anything was written.
fn gate_row(gpu: &dyn GpuBackend, s: &Setup, r: &Route) -> Result<bool> {
    let n = TOP_K * HIDDEN * 2;
    let [o0, o1, o2] = s.out;
    for o in s.out {
        gpu.memset(o, 0, n)?;
    }
    own_sweep(
        gpu,
        s.own,
        s.kn.ctas,
        &s.x_act,
        r,
        &[&s.gate, &s.upt],
        &[o0, o1],
        GU,
        0,
    )?;
    let ng = TOP_K * MI * 2;
    let (g, u) = (read(gpu, o0, ng)?, read(gpu, o1, ng)?);
    let mut wrote = false;
    for (name, t, got) in [("gate", &s.gate, g), ("up", &s.upt, u)] {
        gpu.memset(o2, 0, n)?;
        slots_gemv(gpu, &s.kn, &s.x_act, r, t, o2, GU)?;
        ensure!(
            read(gpu, o2, ng)? == got,
            "{name} differs from the slot GEMV on ids {:?}",
            r.host
        );
        wrote |= got.iter().any(|&b| b != 0);
    }
    gpu.memset(o0, 0, n)?;
    gpu.memset(o2, 0, n)?;
    own_sweep(gpu, s.own, s.kn.ctas, &s.d_act, r, &[&s.down], &[o0], DN, 0)?;
    slots_gemv(gpu, &s.kn, &s.d_act, r, &s.down, o2, DN)?;
    let d = read(gpu, o0, n)?;
    ensure!(
        d == read(gpu, o2, n)?,
        "down differs from the slot GEMV on ids {:?}",
        r.host
    );
    Ok(wrote || d.iter().any(|&b| b != 0))
}

/// 2026-10-09: Part 1.
/// 2026-10-10: Returns the rows it gated, which the width-704 gate (`k64.rs`) reuses.
fn byte_gate(
    gpu: &dyn GpuBackend,
    s: &Setup,
    rng: &mut Rng,
    fx: Option<&Fixture>,
) -> Result<Vec<Vec<i32>>> {
    let mut rows: Vec<(String, Vec<i32>)> = Vec::new();
    for local in 0..=TOP_K {
        for i in 0..4 {
            rows.push((format!("local {local} #{i}"), row_with_local(rng, local)));
        }
    }
    rows.push(("repeated id".into(), vec![5, 5, 17, 100, 200, 3, 250, 40]));
    rows.push((
        "out of range".into(),
        vec![-1, 7, 288, 9, 1000, 120, -5, 95],
    ));
    let seeded = rows.len();
    if let Some(f) = fx {
        rows.extend(
            f.ids
                .chunks_exact(TOP_K)
                .enumerate()
                .map(|(i, r)| (format!("fixture row {i}"), r.to_vec())),
        );
    }
    for (name, host) in &rows {
        let local = host
            .iter()
            .filter(|&&e| e >= 0 && (e as usize) < LOCAL)
            .count();
        let r = route_of(gpu, host.clone())?;
        let wrote = gate_row(gpu, s, &r).with_context(|| name.clone())?;
        ensure!(
            wrote == (local > 0),
            "{name}: wrote {wrote} with {local} local slots"
        );
    }
    println!(
        "byte gate 1 row: {seeded} seeded rows and {} fixture rows: gate, up (fused) and down \
         identical to the slot GEMV",
        rows.len() - seeded
    );
    // 2026-10-09: Negative control: the sweep on a row with one local expert swapped must not
    // match the slot GEMV of the original row.
    let host = row_with_local(rng, 3);
    let mut moved = host.clone();
    moved[0] = (moved[0] + 1) % LOCAL as i32;
    while host.contains(&moved[0]) {
        moved[0] = (moved[0] + 1) % LOCAL as i32;
    }
    let (r, m) = (route_of(gpu, host)?, route_of(gpu, moved)?);
    let ng = TOP_K * MI * 2;
    for o in s.out {
        gpu.memset(o, 0, ng)?;
    }
    own_sweep(
        gpu,
        s.own,
        s.kn.ctas,
        &s.x_act,
        &m,
        &[&s.gate],
        &[s.out[0]],
        GU,
        0,
    )?;
    slots_gemv(gpu, &s.kn, &s.x_act, &r, &s.gate, s.out[1], GU)?;
    ensure!(
        read(gpu, s.out[0], ng)? != read(gpu, s.out[1], ng)?,
        "negative control: a swapped expert left the bytes equal"
    );
    println!("negative control: a swapped expert changes the bytes");
    Ok(rows.into_iter().map(|(_, r)| r).collect())
}

fn main() -> Result<()> {
    let ordinal = std::env::var("GLM_BENCH_GPU_ORDINAL")
        .context("GLM_BENCH_GPU_ORDINAL names the GPU to run on")?
        .parse()?;
    let target = metrale_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4")
        .context("glm-5.3-flash nvfp4 target")?;
    let gpu = MetraleCudaBackend::new(ordinal, &target.modules)?;
    let stream = gpu.create_stream()?;
    let g: &dyn GpuBackend = &gpu;
    let m = "w4a4_gemv_mx_moe";
    let kn = Kern {
        quant: g.kernel(m, "w4a4_quant_rows_static")?,
        row_union: g.kernel("w4a16_gemv", "glm5next_moe_row_union")?,
        slots: g.kernel(m, "w4a4_gemv_mx8_moe_slots")?,
        union: [
            g.kernel(m, "w4a4_gemv_mx8_moe_union")?,
            g.kernel(m, "w4a4_gemv_mx16_moe_union")?,
        ],
        sweep: [
            g.kernel(m, "w4a4_gemv_mx8_moe_union_sweep")?,
            g.kernel(m, "w4a4_gemv_mx16_moe_union_sweep")?,
        ],
        ctas: W4A4_SWEEP_CTAS_PER_SM * g.sm_count()?,
    };
    let own = g.kernel(m, "w4a4_gemv_mx8_moe_slots_sweep")?;
    let fx = fixture()?;
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let (gate, upt) = (
        table(g, &mut rng, MI, HIDDEN, GS_GATE_UP)?,
        table(g, &mut rng, MI, HIDDEN, GS_GATE_UP)?,
    );
    let down = table(g, &mut rng, HIDDEN, MI, GS_DOWN)?;
    let x: Vec<f32> = (0..MAX_ROWS * HIDDEN).map(|_| rng.unit() * 3.0).collect();
    let act: Vec<f32> = (0..MAX_ROWS * TOP_K * MI)
        .map(|_| rng.unit() * 10.0)
        .collect();
    let (xd, ad) = (up_bf16(g, &x)?, up_bf16(g, &act)?);
    let x_act = quantize(g, &kn, xd, MAX_ROWS, HIDDEN, GS_GATE_UP)?;
    let d_act = quantize(g, &kn, ad, MAX_ROWS * TOP_K, MI, GS_DOWN)?;
    let out_bytes = MAX_ROWS * TOP_K * HIDDEN * 2;
    let s = Setup {
        kn,
        own,
        gate,
        upt,
        down,
        x_act,
        d_act,
        out: [
            g.alloc(out_bytes)?,
            g.alloc(out_bytes)?,
            g.alloc(out_bytes)?,
        ],
    };
    // 2026-10-09: GLM_BENCH_TIMING_ONLY=1 runs part 2 alone (repeat timing runs).
    let timing_only = std::env::var("GLM_BENCH_TIMING_ONLY").as_deref() == Ok("1");
    if !timing_only {
        let rows = byte_gate(g, &s, &mut rng, fx.as_ref())?;
        k64::run(g, &s, &mut rng, &rows)?;
    }
    timing::run(g, &s, &mut rng, fx.as_ref(), stream)?;
    match &fx {
        Some(_) if timing_only => Ok(()),
        Some(f) => forward::run(&gpu, &s, f, stream),
        None => {
            println!("no GLM_ROUTING_FIXTURE: forward_moe part skipped");
            Ok(())
        }
    }
}
