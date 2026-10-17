// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: `GLM_BENCH_TP=1` mode of `glm5next_moe_narrow_bench`: the routed-expert sweeps at
//! the tp expert layout's per-rank shapes (every one of the 288 experts held as a slice: gate
//! and up N 704 at K 4096; down N 4096 at K 704 on the `_k64` entries, ranks 0 and 1, and at K
//! 640 on the plain entries, rank 2), at 1, 2, 4, 8 and 16 rows.
//!
//! 1. Byte gate: every (row, slot) output of the sweep (own slots at one row, the union above)
//!    against the slot GEMV of the same projection (`w4a4_gemv_mx8_moe_slots[_k64]`, the one-tile
//!    body), on random routings, plus a repeated id at one row.
//! 2. Timing (CUDA-graph replay, a different routing per call): gate+up, down 704 and down 640,
//!    each cold and after a dirty-L2 preamble shaped like a KDA layer's state write-back (a
//!    1.44 MB device copy; the preamble alone is timed too). GB/s is over the union's bytes.
//!
//! The build's sweep variant (tiles per item, prefetch depth, CTAs per SM) comes from the
//! kernel's compile-time macros; `GLM_SWEEP_CTAS` overrides the grid.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: exits with an error on the first byte mismatch.

use std::cell::Cell;

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::device::*;
use crate::{slots_on, sweep_on};

const W: usize = 704;
const WP: usize = 768;
const W2: usize = 640;
const ROWS: [usize; 5] = [1, 2, 4, 8, 16];
/// 2026-10-10: A KDA layer's dirty FP32 recurrent state per rank.
const DIRTY: usize = 1_440_000;

struct K {
    quant: KernelHandle,
    quant_k64: KernelHandle,
    slots: KernelHandle,
    slots_k64: KernelHandle,
    own: KernelHandle,
    own_k64: KernelHandle,
    sweep: [KernelHandle; 2],
    sweep_k64: [KernelHandle; 2],
    kn: Kern,
}

struct Tp {
    k: K,
    gate: Table,
    upt: Table,
    d704: Table,
    d640: Table,
    x: Act,
    a704: Act,
    a640: Act,
    out: [DevicePtr; 3],
}

fn quant_into(
    gpu: &dyn GpuBackend,
    kern: KernelHandle,
    x: DevicePtr,
    rows: usize,
    k: usize,
    kp: usize,
    gs: f32,
) -> Result<Act> {
    let a = Act {
        aq: gpu.alloc(rows * kp / 2)?,
        as_: gpu.alloc(rows * kp / 16)?,
        ag: gpu.alloc(rows * 4)?,
    };
    KernelLaunch::new(gpu, kern)
        .grid([rows as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(a.aq)
        .arg_ptr(a.as_)
        .arg_ptr(a.ag)
        .arg_u32(k as u32)
        .arg_f32(gs)
        .launch(0)?;
    Ok(a)
}

/// 2026-10-10: `rows` rows of `TOP_K` distinct experts of all `EXPERTS`.
fn route_rows(gpu: &dyn GpuBackend, rng: &mut Rng, rows: usize) -> Result<Route> {
    crate::route_of(gpu, routing_n(rng, rows))
}

fn routing_n(rng: &mut Rng, rows: usize) -> Vec<i32> {
    let mut ids = Vec::with_capacity(rows * TOP_K);
    for _ in 0..rows {
        let mut row: Vec<i32> = Vec::with_capacity(TOP_K);
        while row.len() < TOP_K {
            let e = (rng.next() % EXPERTS as u64) as i32;
            if !row.contains(&e) {
                row.push(e);
            }
        }
        ids.extend(row);
    }
    ids
}

/// 2026-10-10: The sweep kernel and entry list of a routing, and the gate+up launch (the union
/// build, when there is one, runs before and is not part of it).
impl Tp {
    fn sweep_kern(&self, rows: usize, k64: bool) -> KernelHandle {
        match (rows, k64) {
            (1, false) => self.k.own,
            (1, true) => self.k.own_k64,
            (2..=8, false) => self.k.sweep[0],
            (2..=8, true) => self.k.sweep_k64[0],
            (_, false) => self.k.sweep[1],
            (_, true) => self.k.sweep_k64[1],
        }
    }
    fn entries(&self, r: &Route) -> DevicePtr {
        if r.rows == 1 { r.ids } else { r.u_eid }
    }
    fn gu(&self, gpu: &dyn GpuBackend, r: &Route, st: u64) -> Result<()> {
        let (o0, o1) = (self.out[0], self.out[1]);
        sweep_on(
            gpu,
            self.sweep_kern(r.rows, false),
            self.k.kn.ctas,
            self.entries(r),
            &self.x,
            r,
            &[&self.gate, &self.upt],
            &[o0, o1],
            (W, HIDDEN, TOP_K),
            st,
        )
    }
}

pub fn run(gpu: &dyn GpuBackend, st: u64) -> Result<()> {
    let m = "w4a4_gemv_mx_moe";
    let ctas = match std::env::var("GLM_SWEEP_CTAS") {
        Ok(v) => v.parse()?,
        Err(_) => metrale_model_arch::glm5next_mlp::W4A4_SWEEP_CTAS_PER_SM * gpu.sm_count()?,
    };
    let k = K {
        quant: gpu.kernel(m, "w4a4_quant_rows_static")?,
        quant_k64: gpu.kernel(m, "w4a4_quant_rows_static_k64")?,
        slots: gpu.kernel(m, "w4a4_gemv_mx8_moe_slots")?,
        slots_k64: gpu.kernel(m, "w4a4_gemv_mx8_moe_slots_k64")?,
        own: gpu.kernel(m, "w4a4_gemv_mx8_moe_slots_sweep")?,
        own_k64: gpu.kernel(m, "w4a4_gemv_mx8_moe_slots_sweep_k64")?,
        sweep: [
            gpu.kernel(m, "w4a4_gemv_mx8_moe_union_sweep")?,
            gpu.kernel(m, "w4a4_gemv_mx16_moe_union_sweep")?,
        ],
        sweep_k64: [
            gpu.kernel(m, "w4a4_gemv_mx8_moe_union_sweep_k64")?,
            gpu.kernel(m, "w4a4_gemv_mx16_moe_union_sweep_k64")?,
        ],
        kn: Kern {
            quant: gpu.kernel(m, "w4a4_quant_rows_static")?,
            row_union: gpu.kernel("w4a16_gemv", "glm5next_moe_row_union")?,
            slots: gpu.kernel(m, "w4a4_gemv_mx8_moe_slots")?,
            union: [
                gpu.kernel(m, "w4a4_gemv_mx8_moe_union")?,
                gpu.kernel(m, "w4a4_gemv_mx16_moe_union")?,
            ],
            sweep: [
                gpu.kernel(m, "w4a4_gemv_mx8_moe_union_sweep")?,
                gpu.kernel(m, "w4a4_gemv_mx16_moe_union_sweep")?,
            ],
            ctas,
        },
    };
    println!("tp shapes: sweep grid {ctas} CTAs");
    let mut rng = Rng(0x3c6e_f372_fe94_f82b);
    let gate = table_n(gpu, &mut rng, W, HIDDEN, GS_GATE_UP, EXPERTS)?;
    let upt = table_n(gpu, &mut rng, W, HIDDEN, GS_GATE_UP, EXPERTS)?;
    let d704 = table_n(gpu, &mut rng, HIDDEN, W, GS_DOWN, EXPERTS)?;
    let d640 = table_n(gpu, &mut rng, HIDDEN, W2, GS_DOWN, EXPERTS)?;
    let slots_max = MAX_ROWS * TOP_K;
    let xv: Vec<f32> = (0..MAX_ROWS * HIDDEN).map(|_| rng.unit() * 3.0).collect();
    let av: Vec<f32> = (0..slots_max * W).map(|_| rng.unit() * 10.0).collect();
    let av2: Vec<f32> = (0..slots_max * W2).map(|_| rng.unit() * 10.0).collect();
    let (xd, ad, ad2) = (up_bf16(gpu, &xv)?, up_bf16(gpu, &av)?, up_bf16(gpu, &av2)?);
    let x = quant_into(gpu, k.quant, xd, MAX_ROWS, HIDDEN, HIDDEN, GS_GATE_UP)?;
    let a704 = quant_into(gpu, k.quant_k64, ad, slots_max, W, WP, GS_DOWN)?;
    let a640 = quant_into(gpu, k.quant, ad2, slots_max, W2, W2, GS_DOWN)?;
    let ob = slots_max * HIDDEN * 2;
    let tp = Tp {
        k,
        gate,
        upt,
        d704,
        d640,
        x,
        a704,
        a640,
        out: [gpu.alloc(ob)?, gpu.alloc(ob)?, gpu.alloc(ob)?],
    };
    if std::env::var("GLM_BENCH_TIMING_ONLY").as_deref() != Ok("1") {
        gate_all(gpu, &tp, &mut rng)?;
    }
    timing(gpu, &tp, &mut rng, st)
}

/// 2026-10-10: Part 1: each sweep against the slot GEMV, every (row, slot) byte.
fn gate_all(gpu: &dyn GpuBackend, tp: &Tp, rng: &mut Rng) -> Result<()> {
    let [o0, o1, o2] = tp.out;
    let mut routes: Vec<Route> = Vec::new();
    for rows in ROWS {
        for _ in 0..6 {
            routes.push(route_rows(gpu, rng, rows)?);
        }
    }
    routes.push(crate::route_of(gpu, vec![7, 7, 100, 250, 3, 287, 40, 41])?);
    // 2026-10-10: Dead slots (invalid ids): the one-row sweep must write +0.0 over their rows,
    // so its outputs start from a 0xA5 sentinel while the slot GEMV's start from zero.
    routes.push(crate::route_of(gpu, vec![-1, 9, 400, 12, -7, 288, 30, 31])?);
    for r in &routes {
        let fill = if r.rows == 1 { 0xA5u8 } else { 0u8 };
        if r.rows > 1 {
            row_union(gpu, &tp.k.kn, r, 0)?;
        }
        let ob = r.rows * TOP_K * HIDDEN * 2;
        let ng = r.rows * TOP_K * W * 2;
        for o in tp.out {
            gpu.memset(o, fill, ob)?;
        }
        tp.gu(gpu, r, 0)?;
        let (g, u) = (read(gpu, o0, ng)?, read(gpu, o1, ng)?);
        for (name, t, got) in [("gate", &tp.gate, g), ("up", &tp.upt, u)] {
            gpu.memset(o2, 0, ob)?;
            slots_on(gpu, tp.k.slots, &tp.x, r, t, o2, (W, HIDDEN, TOP_K), 0)?;
            ensure!(
                read(gpu, o2, ng)? == got,
                "tp rows={} {name}: sweep differs from the slot GEMV ({:?})",
                r.rows,
                r.host
            );
        }
        for (name, k64, t, a, kk) in [
            ("down 704", true, &tp.d704, &tp.a704, W),
            ("down 640", false, &tp.d640, &tp.a640, W2),
        ] {
            gpu.memset(o0, fill, ob)?;
            gpu.memset(o2, 0, ob)?;
            let kern = tp.sweep_kern(r.rows, k64);
            sweep_on(
                gpu,
                kern,
                tp.k.kn.ctas,
                tp.entries(r),
                a,
                r,
                &[t],
                &[o0],
                (HIDDEN, kk, 1),
                0,
            )?;
            let sk = if k64 { tp.k.slots_k64 } else { tp.k.slots };
            slots_on(gpu, sk, a, r, t, o2, (HIDDEN, kk, 1), 0)?;
            let got = read(gpu, o0, ob)?;
            ensure!(
                got.iter().any(|&b| b != 0),
                "tp rows={} {name}: wrote nothing",
                r.rows
            );
            ensure!(
                got == read(gpu, o2, ob)?,
                "tp rows={} {name}: sweep differs from the slot GEMV ({:?})",
                r.rows,
                r.host
            );
        }
    }
    println!(
        "tp byte gate: {} routings at 1/2/4/8/16 rows and a repeated id: gate, up, down 704 \
         (k64) and down 640 identical to the slot GEMVs",
        routes.len()
    );
    Ok(())
}

/// 2026-10-10: Part 2.
fn timing(gpu: &dyn GpuBackend, tp: &Tp, rng: &mut Rng, st: u64) -> Result<()> {
    let n = env_or("GLM_BENCH_ITERS", ITERS);
    let (ds, dd) = (gpu.alloc(DIRTY)?, gpu.alloc(DIRTY)?);
    let dirty = || gpu.copy_d2d_async(ds, dd, DIRTY, st);
    let gu_bytes = 2 * tp.gate.bytes_per_expert;
    for rows in ROWS {
        let rs: Vec<Route> = (0..n)
            .map(|_| route_rows(gpu, rng, rows))
            .collect::<Result<_>>()?;
        for r in &rs {
            if rows > 1 {
                row_union(gpu, &tp.k.kn, r, st)?;
            }
        }
        let mean_u = rs
            .iter()
            .map(|r| {
                let mut v = r.host.clone();
                v.sort_unstable();
                v.dedup();
                v.len()
            })
            .sum::<usize>() as f64
            / n as f64;
        let i = Cell::new(0usize);
        let next = || {
            let r = &rs[i.get() % rs.len()];
            i.set(i.get() + 1);
            r
        };
        let o0 = tp.out[0];
        let down = |k64: bool, r: &Route| -> Result<()> {
            let (t, a, kk) = if k64 {
                (&tp.d704, &tp.a704, W)
            } else {
                (&tp.d640, &tp.a640, W2)
            };
            sweep_on(
                gpu,
                tp.sweep_kern(rows, k64),
                tp.k.kn.ctas,
                tp.entries(r),
                a,
                r,
                &[t],
                &[o0],
                (HIDDEN, kk, 1),
                st,
            )
        };
        type F<'a> = &'a dyn Fn() -> Result<()>;
        let gu = || tp.gu(gpu, next(), st);
        let d7 = || down(true, next());
        let d6 = || down(false, next());
        let gu_d = || -> Result<()> {
            dirty()?;
            tp.gu(gpu, next(), st)
        };
        let d7_d = || -> Result<()> {
            dirty()?;
            down(true, next())
        };
        let fs: [(&str, F, usize); 6] = [
            ("gate+up", &gu, gu_bytes),
            ("down 704 (k64)", &d7, tp.d704.bytes_per_expert),
            ("down 640", &d6, tp.d640.bytes_per_expert),
            ("dirty 1.44 MB + gate+up", &gu_d, gu_bytes),
            ("dirty 1.44 MB + down 704", &d7_d, tp.d704.bytes_per_expert),
            ("dirty 1.44 MB alone", &dirty, 0),
        ];
        let cl: Vec<F> = fs.iter().map(|f| f.1).collect();
        let t = time_set(gpu, st, &cl)?;
        println!("tp {rows:2} row(s), mean union {mean_u:.1} experts");
        for ((name, _, b), t) in fs.iter().zip(&t) {
            let gbs = if *b == 0 {
                String::new()
            } else {
                format!("  {:6.1} GB/s", mean_u * *b as f64 / t[0] / 1e3)
            };
            println!("  {name:26} {}{gbs}", fmt(*t));
        }
    }
    Ok(())
}
