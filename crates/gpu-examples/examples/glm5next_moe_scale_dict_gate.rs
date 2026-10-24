// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Byte gate and microbench of `METRALE_GLM_MOE_SCALE_DICT`'s decode entries on the real
//! checkpoint: every `_cs` entry of `w4a4_gemv_mx_moe.cu` (the union sweeps, mx8 and mx16, plain and
//! serial `_sl`; the one-row own-slots sweep; the slot GEMV; and their `_k64` twins) over experts
//! whose E4M3 block scales are compact (5-bit offsets from a per-tensor base,
//! `glm5next_mlp::scale_dict`) against its plain twin over the plain scales.
//!
//! Shapes: the TP=3 rank-0 slices of GLM-5.3-Flash's experts (gate/up rows 0..704 of 2048 over
//! K 4096; down columns 0..704 of K 2048, the `_k64` sweeps), a 2-row verify (top 8, 13
//! experts in the union, 3 shared), activations quantized with the checkpoint's input scales.
//!
//! Owner: gpu examples (GLM-5.3 MoE).
//! Invariants:
//! - Fails unless gate, up and down outputs (poisoned before each run) are byte-identical, for
//!   every layer of `LAYERS`, with one expert of each layer forced onto its plain scales in the
//!   compact arm (the per-expert fallback path).
//! - Prints how many of the loaded tensors fall back (codes spanning more than 32).
//!
//!   GLM_CKPT=/srv/models/glm-5.3-flash-nvfp4 cargo run -p metrale-gpu-examples --release \
//!       --example glm5next_moe_scale_dict_gate --features cuda,gpu-examples
//!
//! Timing: each arm one graph over the `LAYERS` layers (distinct weights), us per layer for the
//! serving entries (the serial gate/up union sweep, the `_k64` down union sweep), and effective
//! GB/s over the union's plain bytes.

#[path = "common/glm_ckpt.rs"]
mod ckpt;
#[path = "common/glm_moe_ckpt_experts.rs"]
mod experts;
#[path = "common/glm5next_rows_fixture.rs"]
mod fixture;

use anyhow::{Context, Result, ensure};
use experts::{EXPERTS, HID, Layer, Proj, SLICE};
use fixture::{Lcg, read, up, up_bf16, up_i32};
use metrale_gpu_examples::graph_arms::{Plan, time_arms};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const ROWS: usize = 2;
const TOP_K: usize = 8;
const LAYERS: [usize; 6] = [5, 12, 20, 28, 36, 44];
/// 2026-10-21: Row 0's and row 1's experts: 3 shared, 13 in the union.
const ROUTE: [[usize; TOP_K]; ROWS] = [
    [3, 17, 40, 77, 101, 150, 199, 240],
    [17, 77, 199, 8, 55, 120, 166, 251],
];

/// 2026-10-21: How an entry reads its experts: the union sweep over the rows' union, the one-row
/// own-slots sweep over row 0's ids, or the slot GEMV over every (row, slot).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Union,
    Own,
    Slots,
}

/// 2026-10-21: Every `_cs` decode entry, by its plain twin's name: gate/up (K 4096), then down (the
/// `_k64` entries, K 704).
const GATE_UP: [(&str, Kind); 6] = [
    ("w4a4_gemv_mx8_moe_union_sweep", Kind::Union),
    ("w4a4_gemv_mx16_moe_union_sweep", Kind::Union),
    ("w4a4_gemv_mx8_moe_union_sweep_sl", Kind::Union),
    ("w4a4_gemv_mx16_moe_union_sweep_sl", Kind::Union),
    ("w4a4_gemv_mx8_moe_slots_sweep", Kind::Own),
    ("w4a4_gemv_mx8_moe_slots", Kind::Slots),
];
const DOWN: [(&str, Kind); 4] = [
    ("w4a4_gemv_mx8_moe_union_sweep_k64", Kind::Union),
    ("w4a4_gemv_mx16_moe_union_sweep_k64", Kind::Union),
    ("w4a4_gemv_mx8_moe_slots_sweep_k64", Kind::Own),
    ("w4a4_gemv_mx8_moe_slots_k64", Kind::Slots),
];

/// 2026-10-21: One entry and its `_cs` twin.
struct Pair {
    name: &'static str,
    kern: [KernelHandle; 2],
    kind: Kind,
    down: bool,
}

struct Quant {
    gu: KernelHandle,
    down: KernelHandle,
}

/// 2026-10-21: The union tables: u_eid (T entries, -1 past the union) and u_slot (u * ROWS + r).
fn union(g: &dyn GpuBackend) -> Result<(DevicePtr, DevicePtr)> {
    let mut eid: Vec<i32> = Vec::new();
    for r in ROUTE {
        for e in r {
            if !eid.contains(&(e as i32)) {
                eid.push(e as i32);
            }
        }
    }
    let mut slot = vec![-1i32; ROWS * TOP_K * ROWS];
    for (u, &e) in eid.iter().enumerate() {
        for (r, route) in ROUTE.iter().enumerate() {
            if let Some(s) = route.iter().position(|&x| x as i32 == e) {
                slot[u * ROWS + r] = s as i32;
            }
        }
    }
    eid.resize(ROWS * TOP_K, -1);
    let bytes = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
    Ok((up(g, &bytes(&eid))?, up(g, &bytes(&slot))?))
}

/// 2026-10-21: Quantized activations: `rows` rows of width `k` (padded to 128 for `_k64`).
struct Act {
    q: DevicePtr,
    s: DevicePtr,
    gs: DevicePtr,
}

fn act(
    g: &dyn GpuBackend,
    kq: KernelHandle,
    rng: &mut Lcg,
    rows: usize,
    k: usize,
    gs: f32,
) -> Result<Act> {
    let kp = k.next_multiple_of(128);
    let a = up_bf16(g, &rng.vec(rows * k, 2.0))?;
    let r = Act {
        q: g.alloc(rows * kp / 2)?,
        s: g.alloc(rows * kp / 16)?,
        gs: g.alloc(rows * 4)?,
    };
    KernelLaunch::new(g, kq)
        .grid([rows as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(r.q)
        .arg_ptr(r.s)
        .arg_ptr(r.gs)
        .arg_u32(k as u32)
        .arg_f32(gs)
        .launch(0)?;
    Ok(r)
}

/// 2026-10-21: One launch of `pair` (its `_cs` twin when `cs`) over `p` (gate and up, or down
/// twice) into `c`: `nk` = (N, K, act_div).
#[allow(clippy::too_many_arguments)]
fn launch(
    g: &dyn GpuBackend,
    pair: &Pair,
    a: &Act,
    io: &Io,
    p: [&Proj; 2],
    c: [DevicePtr; 2],
    nk: (usize, usize, usize),
    cs: bool,
    s: u64,
) -> Result<()> {
    let sc = |x: &Proj| if cs { x.cs } else { x.plain };
    let kern = pair.kern[usize::from(cs)];
    if pair.kind == Kind::Slots {
        for (x, out) in p.iter().zip(c).take(if pair.down { 1 } else { 2 }) {
            let mut l = KernelLaunch::new(g, kern)
                .grid([nk.0.div_ceil(16) as u32, (ROWS * TOP_K) as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(a.q)
                .arg_ptr(a.s)
                .arg_ptr(a.gs)
                .arg_ptr(io.ids)
                .arg_ptr(x.packed)
                .arg_ptr(sc(x))
                .arg_ptr(x.s2)
                .arg_ptr(out)
                .arg_u32(nk.0 as u32)
                .arg_u32(nk.1 as u32)
                .arg_u32(nk.2 as u32)
                .arg_u32(EXPERTS as u32);
            if cs {
                l = l.arg_ptr(x.base);
            }
            l.launch(s)?;
        }
        return Ok(());
    }
    let (rows, u_eid) = match pair.kind {
        Kind::Own => (1, io.ids),
        _ => (ROWS, io.u.0),
    };
    let mut l = KernelLaunch::new(g, kern)
        .grid([48, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a.q)
        .arg_ptr(a.s)
        .arg_ptr(a.gs)
        .arg_ptr(u_eid)
        .arg_ptr(io.u.1);
    for (x, out) in p.iter().zip(c) {
        l = l
            .arg_ptr(x.packed)
            .arg_ptr(sc(x))
            .arg_ptr(x.s2)
            .arg_ptr(out);
    }
    l = l
        .arg_u32(if pair.down { 1 } else { 2 })
        .arg_u32(nk.0 as u32)
        .arg_u32(nk.1 as u32)
        .arg_u32(rows as u32)
        .arg_u32(TOP_K as u32)
        .arg_u32(nk.2 as u32)
        .arg_u32(EXPERTS as u32);
    if cs {
        l = l.arg_ptr(p[0].base).arg_ptr(p[1].base);
    }
    l.launch(s)
}

struct Io {
    x: Act,
    d: Act,
    ids: DevicePtr,
    u: (DevicePtr, DevicePtr),
    out: [DevicePtr; 3],
}

const OUT_BYTES: [usize; 3] = [
    ROWS * TOP_K * SLICE * 2,
    ROWS * TOP_K * SLICE * 2,
    ROWS * TOP_K * HID * 2,
];

/// 2026-10-21: `pair` on layer `l`'s gate and up, or its down.
fn run(g: &dyn GpuBackend, pair: &Pair, l: &Layer, io: &Io, cs: bool, s: u64) -> Result<()> {
    if pair.down {
        let c = [io.out[2], io.out[2]];
        launch(
            g,
            pair,
            &io.d,
            io,
            [&l.down, &l.down],
            c,
            (HID, SLICE, 1),
            cs,
            s,
        )
    } else {
        let c = [io.out[0], io.out[1]];
        launch(
            g,
            pair,
            &io.x,
            io,
            [&l.gate, &l.up],
            c,
            (SLICE, HID, TOP_K),
            cs,
            s,
        )
    }
}

fn main() -> Result<()> {
    let dir = metrale_gpu_examples::env_var("GLM_CKPT")?
        .unwrap_or_else(|| "/srv/models/glm-5.3-flash-nvfp4".into());
    let timing = !std::env::args().any(|a| a == "--no-timing");
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    let gpu = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &gpu;
    let m = "w4a4_gemv_mx_moe";
    let quant = Quant {
        gu: g.kernel(m, "w4a4_quant_rows_static")?,
        down: g.kernel(m, "w4a4_quant_rows_static_k64")?,
    };
    let pairs: Vec<Pair> = GATE_UP
        .iter()
        .map(|&(n, kind)| (n, kind, false))
        .chain(DOWN.iter().map(|&(n, kind)| (n, kind, true)))
        .map(|(name, kind, down)| -> Result<Pair> {
            let kern = [g.kernel(m, name)?, g.kernel(m, &format!("{name}_cs"))?];
            Ok(Pair {
                name,
                kern,
                kind,
                down,
            })
        })
        .collect::<Result<_>>()?;
    let mut used: Vec<usize> = ROUTE.iter().flatten().copied().collect();
    used.sort_unstable();
    used.dedup();
    let t0 = std::time::Instant::now();
    let layers: Vec<Layer> = LAYERS
        .iter()
        .enumerate()
        .map(|(i, &l)| Layer::load(g, &dir, l, &used, ROUTE[0][i % TOP_K], false))
        .collect::<Result<_>>()?;
    let fb: usize = layers.iter().map(Layer::fallback).sum();
    println!(
        "loaded {} layers x {} experts x 3 tensors in {:.1} s; {fb} of {} tensors keep plain scales (span > 32)",
        LAYERS.len(),
        used.len(),
        t0.elapsed().as_secs_f64(),
        LAYERS.len() * used.len() * 3
    );
    let mut rng = Lcg(0x5ca1_ed1c);
    let u = union(g)?;
    let ids = up_i32(
        g,
        &ROUTE
            .iter()
            .flatten()
            .map(|&e| e as i32)
            .collect::<Vec<_>>(),
    )?;
    let ios: Vec<[Io; 2]> = layers
        .iter()
        .map(|l| -> Result<[Io; 2]> {
            let x_seed = rng.0;
            let mk = |rng: &mut Lcg| -> Result<Io> {
                Ok(Io {
                    x: act(g, quant.gu, rng, ROWS, HID, l.gs_gu)?,
                    d: act(g, quant.down, rng, ROWS * TOP_K, SLICE, l.gs_down)?,
                    ids,
                    u,
                    out: OUT_BYTES.map(|b| g.alloc(b).expect("alloc")),
                })
            };
            let a = mk(&mut Lcg(x_seed))?;
            let b = mk(&mut Lcg(x_seed))?;
            rng.next();
            Ok([a, b])
        })
        .collect::<Result<_>>()?;
    for pair in &pairs {
        let outs: &[usize] = if pair.down { &[2] } else { &[0, 1] };
        for (i, (l, io)) in layers.iter().zip(&ios).enumerate() {
            for (arm, cs) in io.iter().zip([false, true]) {
                for (p, b) in arm.out.iter().zip(OUT_BYTES) {
                    g.memset(*p, 0x5A, b)?;
                }
                run(g, pair, l, arm, cs, 0)?;
            }
            g.synchronize(0)?;
            for &j in outs {
                let what = ["gate", "up", "down"][j];
                let (a, b) = (
                    read(g, ios[i][0].out[j], OUT_BYTES[j])?,
                    read(g, ios[i][1].out[j], OUT_BYTES[j])?,
                );
                ensure!(
                    a.iter().any(|&x| x != 0x5A),
                    "{}: layer {}: {what}: the plain entry wrote nothing",
                    pair.name,
                    LAYERS[i]
                );
                ensure!(
                    a == b,
                    "{}: layer {}: {what}: compact scales differ from plain ({} of {} bytes)",
                    pair.name,
                    LAYERS[i],
                    a.iter().zip(&b).filter(|(x, y)| x != y).count(),
                    a.len()
                );
            }
        }
        println!(
            "  {:40} {} layers BYTE-IDENTICAL (one expert per layer forced plain)",
            format!("{}_cs", pair.name),
            LAYERS.len()
        );
    }

    if timing {
        let s = g.create_stream()?;
        let plan = Plan {
            calls: LAYERS.len(),
            rounds: 7,
            trials: 15,
            warm: std::time::Duration::from_millis(300),
        };
        // 2026-10-21: The serving entries: the serial gate/up union sweep, the `_k64` down sweep.
        let (gu, dn) = (&pairs[2], &pairs[GATE_UP.len()]);
        let one = |i: usize, cs: bool, p: &Pair, st: u64| {
            run(g, p, &layers[i], &ios[i][usize::from(cs)], cs, st)
        };
        let arms: [&dyn Fn(u64, usize) -> Result<()>; 4] = [
            &|st, i| one(i, false, gu, st),
            &|st, i| one(i, true, gu, st),
            &|st, i| one(i, false, dn, st),
            &|st, i| one(i, true, dn, st),
        ];
        let r = time_arms(g, s, plan, &arms)?;
        let gu_bytes = layers
            .iter()
            .map(|l| l.gate.bytes + l.up.bytes)
            .sum::<usize>() as f64
            / LAYERS.len() as f64;
        let d_bytes =
            layers.iter().map(|l| l.down.bytes).sum::<usize>() as f64 / LAYERS.len() as f64;
        println!(
            "\n  TIMING: {} layers (distinct weights) per graph, us per launch median [min, max]; GB/s over the plain bytes",
            LAYERS.len()
        );
        for (name, i, b) in [(gu.name, 0, gu_bytes), (dn.name, 2, d_bytes)] {
            let (p, c) = (r[i], r[i + 1]);
            println!(
                "    {name:36} plain {:7.2} [{:7.2}, {:7.2}] {:6.1} GB/s   compact {:7.2} [{:7.2}, {:7.2}] {:6.1} GB/s   saves {:5.2} us/launch",
                p.0,
                p.1,
                p.2,
                b / p.0 / 1e3,
                c.0,
                c.1,
                c.2,
                b / c.0 / 1e3,
                p.0 - c.0
            );
        }
        println!(
            "    per step (42 MoE layers): {:.3} ms",
            ((r[0].0 - r[1].0) + (r[2].0 - r[3].0)) * 42.0 / 1000.0
        );
    }
    println!("\nMoE scale-dictionary decode gate PASS");
    Ok(())
}
