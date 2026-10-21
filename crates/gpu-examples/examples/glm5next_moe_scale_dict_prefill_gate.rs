// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Byte gate and microbench of `METRALE_GLM_MOE_SCALE_DICT`'s grouped prefill entries
//! on the real checkpoint: every `_cs` entry of `w4a4_grouped_mx.cu` (the m32 and m128 tiles,
//! plain and `_k64`, the raster `_r` and column-tile `_n` twins, and the fused gate/up `gu` entries)
//! over experts whose E4M3 block scales are compact (`glm5next_mlp::scale_dict`) against its plain
//! twin over the plain scales.
//!
//! Shapes: the TP=3 rank-0 slices of GLM-5.3-Flash's experts (gate/up N 704 over K 4096, a partial
//! 128- or 64-row column tile; down N 4096 over K 704, the `_k64` entries), `LOADED` experts of each
//! of `LAYERS` loaded and aliased over all 288 ids (`glm_moe_ckpt_experts`), top 8. Routings of
//! 17 rows (the first grouped launch), 512 rows skewed (experts of 1..2 slots, a partial m128 tile,
//! unrouted ids) and 8208 rows uniform (the serving prefill sub-chunk). Gate and up read the
//! token rows through the expert-sorted gather; down reads its own slot rows at Kp 768; the fused
//! entries write the down input (codes, scales, globals) under the down input scale.
//!
//! Owner: gpu examples (GLM-5.3 MoE).
//! Invariants:
//! - Fails unless every output (poisoned before each run) is byte-identical between an entry and
//!   its `_cs` twin, on every routing and layer, with one expert per layer forced onto its plain
//!   scales in the compact tables (the per-expert fallback path), and unless the plain entry wrote.
//! - Prints how many of the loaded tensors fall back (codes spanning more than 32).
//!
//!   GLM_CKPT=/srv/models/glm-5.3-flash-nvfp4 cargo run -p metrale-gpu-examples --release \
//!       --example glm5next_moe_scale_dict_prefill_gate --features cuda,gpu-examples
//!
//! Timing (`--no-timing` skips): at 8208 rows, each arm one graph of one launch per layer, us per
//! launch for the serving entries (fused gate/up m128, separate gate m128, down m128 `_k64` raster
//! 4), plain against compact.

#[path = "common/glm_ckpt.rs"]
mod ckpt;
#[path = "common/glm_moe_ckpt_experts.rs"]
mod experts;
#[path = "common/glm5next_rows_fixture.rs"]
mod fixture;

use anyhow::{Context, Result, ensure};
use experts::{EXPERTS, HID, Layer, Proj, SLICE};
use fixture::{Lcg, read, up_bf16, up_i32};
use metrale_gpu_examples::graph_arms::{Plan, time_arms};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_mlp::forward::w4a4_grouped_smem_bytes;

const TOP_K: usize = 8;
const KP: usize = 768;
const LIMIT: f32 = 10.0;
const LAYERS: [usize; 2] = [5, 44];
const LOADED: usize = 40;
const ROWS: [usize; 3] = [17, 512, 8208];
const MODULE: &str = "w4a4_grouped_mx";

/// 2026-10-21: What an entry computes: one projection (gate over the gathered token rows, or down
/// over the slot rows), or the fused gate/up epilogue.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Op {
    Gate,
    Down,
    Fused,
}

/// 2026-10-21: An entry by its plain twin's name: (MT, NT), op, trailing (raster, nc) arguments.
struct Entry {
    name: &'static str,
    tile: (usize, usize),
    op: Op,
    raster: Option<u32>,
    nc: Option<u32>,
}

const fn e(
    name: &'static str,
    tile: (usize, usize),
    op: Op,
    raster: Option<u32>,
    nc: Option<u32>,
) -> Entry {
    Entry {
        name,
        tile,
        op,
        raster,
        nc,
    }
}

const ENTRIES: [Entry; 13] = [
    e("w4a4_grouped_mx_m32", (32, 128), Op::Gate, None, None),
    e("w4a4_grouped_mx_m128", (128, 64), Op::Gate, None, None),
    e("w4a4_grouped_mx_m128_r", (128, 64), Op::Gate, Some(4), None),
    e(
        "w4a4_grouped_mx_m128_n",
        (128, 64),
        Op::Gate,
        Some(0),
        Some(3),
    ),
    e("w4a4_grouped_mx_m32_k64", (32, 128), Op::Down, None, None),
    e("w4a4_grouped_mx_m128_k64", (128, 64), Op::Down, None, None),
    e(
        "w4a4_grouped_mx_m128_k64_r",
        (128, 64),
        Op::Down,
        Some(4),
        None,
    ),
    e(
        "w4a4_grouped_mx_m128_k64_n",
        (128, 64),
        Op::Down,
        Some(0),
        Some(3),
    ),
    e(
        "w4a4_grouped_mx_m128_k64_n",
        (128, 64),
        Op::Down,
        Some(4),
        Some(2),
    ),
    e("w4a4_grouped_gu_m32", (32, 128), Op::Fused, None, None),
    e("w4a4_grouped_gu_m128", (128, 64), Op::Fused, None, None),
    e(
        "w4a4_grouped_gu_m128_r",
        (128, 64),
        Op::Fused,
        Some(0),
        None,
    ),
    e(
        "w4a4_grouped_gu_m128_r",
        (128, 64),
        Op::Fused,
        Some(16),
        None,
    ),
];

/// 2026-10-21: One routing: the expert-sorted slots (offsets per id, sorted token rows), and the
/// tile offsets of the two tile heights.
struct Route {
    slots: usize,
    offsets: DevicePtr,
    sorted: DevicePtr,
    tile_off: [DevicePtr; 2],
}

/// 2026-10-21: `rows` rows of top 8 distinct ids: skewed (`skew`: a quarter of the ids take most
/// slots, id 7 none) or uniform; sorted by expert on the host (`moe_sort_by_expert`'s order:
/// expert, then slot index).
fn route(g: &dyn GpuBackend, rng: &mut Lcg, rows: usize, skew: bool) -> Result<Route> {
    let mut ids = Vec::with_capacity(rows * TOP_K);
    for _ in 0..rows {
        let mut row: Vec<usize> = Vec::with_capacity(TOP_K);
        while row.len() < TOP_K {
            let u = (rng.next().abs() * 1e6) as usize;
            let id = if skew && !u.is_multiple_of(4) {
                u % (EXPERTS / 4)
            } else {
                u % EXPERTS
            };
            if id != 7 && !row.contains(&id) {
                row.push(id);
            }
        }
        ids.extend(row);
    }
    let mut order: Vec<usize> = (0..ids.len()).collect();
    order.sort_by_key(|&s| (ids[s], s));
    let mut offsets = vec![0i32; EXPERTS + 1];
    for &id in &ids {
        offsets[id + 1] += 1;
    }
    for i in 0..EXPERTS {
        offsets[i + 1] += offsets[i];
    }
    let sorted: Vec<i32> = order.iter().map(|&s| (s / TOP_K) as i32).collect();
    let tiles = |mt: i32| -> Vec<i32> {
        let mut t = vec![0i32; EXPERTS + 1];
        for i in 0..EXPERTS {
            t[i + 1] = t[i] + (offsets[i + 1] - offsets[i] + mt - 1) / mt;
        }
        t
    };
    Ok(Route {
        slots: ids.len(),
        offsets: up_i32(g, &offsets)?,
        sorted: up_i32(g, &sorted)?,
        tile_off: [up_i32(g, &tiles(32))?, up_i32(g, &tiles(128))?],
    })
}

/// 2026-10-21: Quantized rows (codes, scales, globals) of width `kp`.
struct Act {
    q: DevicePtr,
    s: DevicePtr,
    gs: DevicePtr,
}

fn quant(
    g: &dyn GpuBackend,
    k: KernelHandle,
    rng: &mut Lcg,
    rows: usize,
    kk: usize,
    gs: f32,
) -> Result<Act> {
    let kp = kk.next_multiple_of(128);
    let x = up_bf16(g, &rng.vec(rows * kk, 2.0))?;
    let a = Act {
        q: g.alloc(rows * kp / 2)?,
        s: g.alloc(rows * kp / 16)?,
        gs: g.alloc(rows * 4)?,
    };
    KernelLaunch::new(g, k)
        .grid([rows as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(a.q)
        .arg_ptr(a.s)
        .arg_ptr(a.gs)
        .arg_u32(kk as u32)
        .arg_f32(gs)
        .launch(0)?;
    Ok(a)
}

/// 2026-10-21: One routing's inputs and one arm's outputs (gate or down `c`; the fused entries'
/// down input `dq`, `ds`, `dg`).
struct Io {
    x: Act,
    d: Act,
    out: [DevicePtr; 4],
}

fn out_bytes(slots: usize) -> [usize; 4] {
    [
        slots * HID.max(SLICE) * 2,
        slots * KP / 2,
        slots * KP / 16,
        slots * 4,
    ]
}

/// 2026-10-21: `ent` (its `_cs` twin when `cs`) over layer `l` on routing `r`.
#[allow(clippy::too_many_arguments)]
fn launch(
    g: &dyn GpuBackend,
    kern: [KernelHandle; 2],
    ent: &Entry,
    l: &Layer,
    r: &Route,
    io: &Io,
    ag: f32,
    cs: bool,
    s: u64,
) -> Result<()> {
    let (mt, nt) = ent.tile;
    let sc = |p: &Proj| if cs { p.cs } else { p.plain };
    let (a, n, k, p) = match ent.op {
        Op::Gate | Op::Fused => (&io.x, SLICE, HID, &l.gate),
        Op::Down => (&io.d, HID, SLICE, &l.down),
    };
    let cols = if ent.op == Op::Fused { nt / 2 } else { nt };
    let gx = n.div_ceil(cols).div_ceil(ent.nc.map_or(1, |c| c as usize));
    let mut lc = KernelLaunch::new(g, kern[usize::from(cs)])
        .grid([gx as u32, (r.slots.div_ceil(mt) + EXPERTS) as u32, 1])
        .block([256, 1, 1])
        .shared_mem(w4a4_grouped_smem_bytes(mt, nt, 3))
        .arg_ptr(a.q)
        .arg_ptr(a.s)
        .arg_ptr(a.gs)
        .arg_ptr(p.packed)
        .arg_ptr(sc(p))
        .arg_ptr(p.s2);
    lc = match ent.op {
        Op::Fused => lc
            .arg_ptr(l.up.packed)
            .arg_ptr(sc(&l.up))
            .arg_ptr(l.up.s2)
            .arg_ptr(io.out[1])
            .arg_ptr(io.out[2])
            .arg_ptr(io.out[3]),
        _ => lc.arg_ptr(io.out[0]),
    };
    let tile_off = r.tile_off[usize::from(mt == 128)];
    let gather = if ent.op == Op::Down {
        DevicePtr::NULL
    } else {
        r.sorted
    };
    lc = lc
        .arg_ptr(r.offsets)
        .arg_ptr(tile_off)
        .arg_ptr(gather)
        .arg_u32(EXPERTS as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32);
    if ent.op == Op::Fused {
        lc = lc.arg_u32(KP as u32).arg_f32(ag).arg_f32(LIMIT);
    }
    if let Some(rg) = ent.raster {
        lc = lc.arg_u32(rg);
    }
    if let Some(nc) = ent.nc {
        lc = lc.arg_u32(nc);
    }
    if cs {
        lc = lc.arg_ptr(p.base);
        if ent.op == Op::Fused {
            lc = lc.arg_ptr(l.up.base);
        }
    }
    lc.launch(s)
}

fn main() -> Result<()> {
    let dir =
        std::env::var("GLM_CKPT").unwrap_or_else(|_| "/srv/models/glm-5.3-flash-nvfp4".into());
    let timing = !std::env::args().any(|a| a == "--no-timing");
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    let gpu = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &gpu;
    let qm = "w4a4_gemv_mx_moe";
    let (q_gu, q_down) = (
        g.kernel(qm, "w4a4_quant_rows_static")?,
        g.kernel(qm, "w4a4_quant_rows_static_k64")?,
    );
    let kerns: Vec<[KernelHandle; 2]> = ENTRIES
        .iter()
        .map(|en| {
            Ok([
                g.kernel(MODULE, en.name)?,
                g.kernel(MODULE, &format!("{}_cs", en.name))?,
            ])
        })
        .collect::<Result<_>>()?;
    let loaded: Vec<usize> = (0..LOADED).map(|i| i * 7).collect();
    let t0 = std::time::Instant::now();
    let layers: Vec<Layer> = LAYERS
        .iter()
        .enumerate()
        .map(|(i, &l)| Layer::load(g, &dir, l, &loaded, loaded[3 + i], true))
        .collect::<Result<_>>()?;
    let fb: usize = layers.iter().map(Layer::fallback).sum();
    println!(
        "loaded {} layers x {LOADED} experts x 3 tensors in {:.1} s (aliased over {EXPERTS} ids); {fb} of {} tensors keep plain scales (span > 32)",
        LAYERS.len(),
        t0.elapsed().as_secs_f64(),
        LAYERS.len() * LOADED * 3
    );
    // 2026-10-21: The loader's pack cost: one rank's slice of every expert, 3 tensors each.
    let tensors = LAYERS.len() * LOADED * 3;
    let per = layers
        .iter()
        .map(Layer::compact_time)
        .sum::<std::time::Duration>()
        / tensors as u32;
    println!(
        "compact: {:.1} us per tensor slice => {:.1} ms per layer per rank ({EXPERTS} experts x 3)",
        per.as_secs_f64() * 1e6,
        per.as_secs_f64() * 1e3 * (EXPERTS * 3) as f64
    );
    let mut rng = Lcg(0x9e37_79b9);
    let mut timed = None;
    for &rows in &ROWS {
        let r = route(g, &mut rng, rows, rows == 512)?;
        let ob = out_bytes(r.slots);
        for (li, l) in layers.iter().enumerate() {
            let seed = rng.0;
            let mk = || -> Result<Io> {
                let mut rg = Lcg(seed);
                Ok(Io {
                    x: quant(g, q_gu, &mut rg, rows, HID, l.gs_gu)?,
                    d: quant(g, q_down, &mut rg, r.slots, SLICE, l.gs_down)?,
                    out: ob.map(|b| g.alloc(b).expect("alloc")),
                })
            };
            let ios = [mk()?, mk()?];
            rng.next();
            for (en, &kern) in ENTRIES.iter().zip(&kerns) {
                for (io, cs) in ios.iter().zip([false, true]) {
                    for (p, b) in io.out.iter().zip(ob) {
                        g.memset(*p, 0x5A, b)?;
                    }
                    launch(g, kern, en, l, &r, io, l.gs_down, cs, 0)?;
                }
                g.synchronize(0)?;
                let outs: &[usize] = if en.op == Op::Fused { &[1, 2, 3] } else { &[0] };
                for &j in outs {
                    let (a, b) = (
                        read(g, ios[0].out[j], ob[j])?,
                        read(g, ios[1].out[j], ob[j])?,
                    );
                    let what = ["output", "down codes", "down scales", "down globals"][j];
                    ensure!(
                        a.iter().any(|&x| x != 0x5A),
                        "{}: {rows} rows, layer {}: {what}: the plain entry wrote nothing",
                        en.name,
                        LAYERS[li]
                    );
                    ensure!(
                        a == b,
                        "{} (raster {:?}, nc {:?}): {rows} rows, layer {}: {what}: compact scales differ from plain ({} of {} bytes)",
                        en.name,
                        en.raster,
                        en.nc,
                        LAYERS[li],
                        a.iter().zip(&b).filter(|(x, y)| x != y).count(),
                        a.len()
                    );
                }
            }
            if rows == 8208 && li + 1 == layers.len() {
                timed = Some((r, ios));
                break;
            }
        }
        println!(
            "  {rows:5} rows ({} slots): {} entries x {} layers BYTE-IDENTICAL (one expert per layer forced plain)",
            rows * TOP_K,
            ENTRIES.len(),
            LAYERS.len()
        );
    }

    if let (true, Some((r, ios))) = (timing, timed) {
        let s = g.create_stream()?;
        let plan = Plan {
            calls: 1,
            rounds: 7,
            trials: 9,
            warm: std::time::Duration::from_millis(300),
        };
        let l = &layers[layers.len() - 1];
        let pick = |name: &str, raster: Option<u32>| {
            ENTRIES
                .iter()
                .zip(&kerns)
                .find(|(en, _)| en.name == name && en.raster == raster)
                .expect("listed")
        };
        let timed: [(&str, (&Entry, &[KernelHandle; 2])); 3] = [
            ("fused gate/up m128", pick("w4a4_grouped_gu_m128", None)),
            ("gate m128", pick("w4a4_grouped_mx_m128", None)),
            (
                "down m128_k64 raster 4",
                pick("w4a4_grouped_mx_m128_k64_r", Some(4)),
            ),
        ];
        println!(
            "\n  TIMING: 8208 rows (65664 slots over {EXPERTS} ids), us per launch median [min, max]"
        );
        for (what, (en, kern)) in timed {
            let one = |cs: bool, st: u64| {
                launch(
                    g,
                    *kern,
                    en,
                    l,
                    &r,
                    &ios[usize::from(cs)],
                    l.gs_down,
                    cs,
                    st,
                )
            };
            let arms: [&dyn Fn(u64, usize) -> Result<()>; 2] =
                [&|st, _| one(false, st), &|st, _| one(true, st)];
            let t = time_arms(g, s, plan, &arms)?;
            println!(
                "    {what:24} plain {:8.1} [{:8.1}, {:8.1}]   compact {:8.1} [{:8.1}, {:8.1}]   saves {:6.1} us ({:+.2}%)",
                t[0].0,
                t[0].1,
                t[0].2,
                t[1].0,
                t[1].1,
                t[1].2,
                t[0].0 - t[1].0,
                (t[1].0 / t[0].0 - 1.0) * 100.0
            );
        }
    }
    println!("\nMoE scale-dictionary prefill gate PASS");
    Ok(())
}
