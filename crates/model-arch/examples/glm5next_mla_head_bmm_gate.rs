// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: MLA-UNABSORB G2, kernel level: `glm5next_mla_head_bmm_q` / `_o`
//! (`METRALE_GLM_MLA_UNABSORB`) on random BF16 operands at the rank-0 shape (22 heads).
//!
//! Owner: model-arch examples (GLM-5.3 DSA).
//! Invariants:
//! - Against f64 (M 1, 17, 1040): every output element within
//!   `0.5 ulp_bf16(|ref| + e) + e`, where `e = 2 K 2^-24 Σ|a||b|` bounds the FP32 accumulation
//!   and the half ulp is the final BF16 rounding.
//! - Row identity: for M in {1, 16, 17, 1040, 8192, 8208} and row offsets {0, 1, 7, 16, 127},
//!   the M rows written at offset o are byte-identical to the same rows of one 8335-row run at
//!   offset 0, and the rows just before and after them keep their 0xFF fill.
//! - Negative control, which must FIRE: the 1040-row `_q` launch with 21 heads (wrong A and C
//!   strides) must differ from the 22-head bytes and break the f64 bound.
//!
//!   cargo run -p metrale-model-arch --release --example glm5next_mla_head_bmm_gate \
//!       --features cuda,gpu-examples
//!
//! Timing (`--no-timing` skips it): M 8192, 8208 and 1040 per kernel, each captured as a CUDA
//! graph and replayed (`graph_timing`), host clock around replay + synchronize, not cudaEvent.

#[path = "common/graph_timing.rs"]
mod graph_timing;
#[path = "common/mla_unabsorb_gate.rs"]
mod mla;

use anyhow::{Context, Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_dsa::unabsorb::{HeadBmm, HeadBmmKernels, launch_head_bmm};
use mla::{Lcg, err_stats, f64s, from_bf16_bytes, head_bmm_par, read, to_bf16_bytes, ulp_bf16};

const HEADS: usize = 22;
const F64_ROWS: [usize; 3] = [1, 17, 1040];
const GRID_ROWS: [usize; 6] = [1, 16, 17, 1040, 8192, 8208];
const OFFSETS: [usize; 5] = [0, 1, 7, 16, 127];
/// 2026-10-15: The widest grid case at the largest offset, plus the row after it.
const CAP: usize = 8208 + 127 + 1;
const TIMED_ROWS: [usize; 3] = [8192, 8208, 1040];
const COPIES: usize = 10;

struct Bufs {
    a: DevicePtr,
    b: DevicePtr,
    c_ref: DevicePtr,
    c: DevicePtr,
    a_host: Vec<f32>,
    b_host: Vec<f32>,
    row_a: usize,
    row_c: usize,
}

fn alloc(g: &dyn GpuBackend, which: HeadBmm, seed: u64) -> Result<Bufs> {
    let (kk, n) = which.dims();
    let mut rng = Lcg(seed);
    let a_host = rng.bf16_vals(CAP * HEADS * kk, 1.0);
    let b_host = rng.bf16_vals(HEADS * n * kk, 1.0 / 16.0);
    let (row_a, row_c) = (HEADS * kk * 2, HEADS * n * 2);
    Ok(Bufs {
        a: mla::up_bytes(g, &to_bf16_bytes(&a_host))?,
        b: mla::up_bytes(g, &to_bf16_bytes(&b_host))?,
        c_ref: g.alloc(CAP * row_c)?,
        c: g.alloc(CAP * row_c)?,
        a_host,
        b_host,
        row_a,
        row_c,
    })
}

fn free(g: &dyn GpuBackend, d: Bufs) -> Result<()> {
    for p in [d.a, d.b, d.c_ref, d.c] {
        g.free(p)?;
    }
    Ok(())
}

/// 2026-10-15: Elements outside the f64 bound, and the share equal to the correctly rounded
/// BF16 of the reference.
fn bound_check(out: &[f32], a: &[f32], b: &[f32], m: usize, which: HeadBmm) -> (usize, f64) {
    let (kk, n) = which.dims();
    let reference = head_bmm_par(&f64s(a), &f64s(b), m, HEADS, kk, n);
    let abs = |v: &[f32]| v.iter().map(|x| x.abs() as f64).collect::<Vec<_>>();
    let mag = head_bmm_par(&abs(a), &abs(b), m, HEADS, kk, n);
    let (mut bad, mut exact) = (0usize, 0usize);
    for ((o, r), s) in out.iter().zip(&reference).zip(&mag) {
        let e = 2.0 * kk as f64 * 2f64.powi(-24) * s;
        let d = (*o as f64 - r).abs();
        if d > 0.5 * ulp_bf16(r.abs() + e) + e {
            bad += 1;
        }
        if *o == half::bf16::from_f64(*r).to_f32() {
            exact += 1;
        }
    }
    (bad, exact as f64 / out.len() as f64)
}

#[allow(clippy::too_many_arguments)]
fn launch(
    g: &dyn GpuBackend,
    k: &HeadBmmKernels,
    which: HeadBmm,
    d: &Bufs,
    at: usize,
    m: usize,
    heads: usize,
    s: u64,
) -> Result<()> {
    let (a, c) = (d.a.offset(at * d.row_a), d.c.offset(at * d.row_c));
    launch_head_bmm(g, k, which, a, d.b, c, m, heads, CAP, s)
}

fn vs_f64(g: &dyn GpuBackend, k: &HeadBmmKernels, which: HeadBmm, d: &Bufs) -> Result<bool> {
    let (kk, n) = which.dims();
    let mut ok = true;
    for m in F64_ROWS {
        launch(g, k, which, d, 0, m, HEADS, g.default_stream())?;
        let out = from_bf16_bytes(&read(g, d.c, m * d.row_c)?);
        let reference = head_bmm_par(
            &f64s(&d.a_host[..m * HEADS * kk]),
            &f64s(&d.b_host),
            m,
            HEADS,
            kk,
            n,
        );
        let st = err_stats(&out, &reference);
        let (bad, exact) = bound_check(&out, &d.a_host[..m * HEADS * kk], &d.b_host, m, which);
        ok &= bad == 0;
        println!(
            "  {which:?} M={m:5} vs f64: {st}; {bad} outside the bound; {:.4}% equal to bf16(ref) -> {}",
            exact * 100.0,
            if bad == 0 { "PASS" } else { "FAIL" }
        );
    }
    Ok(ok)
}

fn row_identity(g: &dyn GpuBackend, k: &HeadBmmKernels, which: HeadBmm, d: &Bufs) -> Result<bool> {
    let s = g.default_stream();
    launch_head_bmm(g, k, which, d.a, d.b, d.c_ref, CAP, HEADS, CAP, s)?;
    let full = read(g, d.c_ref, CAP * d.row_c)?;
    let (mut ok, mut cases) = (true, 0);
    for m in GRID_ROWS {
        for o in OFFSETS {
            g.memset(d.c, 0xFF, CAP * d.row_c)?;
            launch(g, k, which, d, o, m, HEADS, s)?;
            let got = read(g, d.c, CAP * d.row_c)?;
            let rows = o * d.row_c..(o + m) * d.row_c;
            let same = got[rows.clone()] == full[rows.clone()];
            let before = o == 0
                || got[(o - 1) * d.row_c..rows.start]
                    .iter()
                    .all(|b| *b == 0xFF);
            let after = got[rows.end..rows.end + d.row_c].iter().all(|b| *b == 0xFF);
            cases += 1;
            if !(same && before && after) {
                ok = false;
                println!(
                    "  {which:?} M={m} offset {o}: rows identical {same}, row before untouched {before}, row after untouched {after} -> FAIL"
                );
            }
        }
    }
    println!(
        "  {which:?} row identity: {cases} (M, offset) cases vs one {CAP}-row run -> {}",
        if ok {
            "PASS (all byte-identical, neighbours untouched)"
        } else {
            "FAIL"
        }
    );
    Ok(ok)
}

fn negctl(g: &dyn GpuBackend, k: &HeadBmmKernels, d: &Bufs) -> Result<bool> {
    let (m, s) = (1040, g.default_stream());
    let bytes = m * d.row_c;
    g.memset(d.c, 0xFF, CAP * d.row_c)?;
    launch(g, k, HeadBmm::Q, d, 0, m, HEADS, s)?;
    let right = read(g, d.c, bytes)?;
    g.memset(d.c, 0xFF, CAP * d.row_c)?;
    launch(g, k, HeadBmm::Q, d, 0, m, HEADS - 1, s)?;
    let wrong = read(g, d.c, bytes)?;
    let differ = right
        .chunks(2)
        .zip(wrong.chunks(2))
        .filter(|(a, b)| a != b)
        .count();
    let (bad, _) = bound_check(
        &from_bf16_bytes(&wrong),
        &d.a_host[..m * HEADS * 256],
        &d.b_host,
        m,
        HeadBmm::Q,
    );
    let fired = differ > 0 && bad > 0;
    println!(
        "NEGCTL wrong stride (_q, M=1040, heads 21 for 22): {differ} of {} elements differ from the 22-head bytes, {bad} outside the f64 bound -> {}",
        bytes / 2,
        if fired { "FIRED" } else { "DID NOT FIRE" }
    );
    Ok(fired)
}

fn timing(g: &dyn GpuBackend, k: &HeadBmmKernels, which: HeadBmm, d: &Bufs) -> Result<()> {
    let (kk, n) = which.dims();
    let s = g.create_stream()?;
    for m in TIMED_ROWS {
        let t = graph_timing::time_graph(g, s, COPIES, &mut |st| {
            launch(g, k, which, d, 0, m, HEADS, st)
        })?;
        let flops = 2.0 * (m * HEADS * n * kk) as f64;
        let bytes = (m * (d.row_a + d.row_c) + HEADS * n * kk * 2) as f64;
        println!(
            "  TIMING {which:?} M={m:5}: {:8.1} us [{:.1}, {:.1}]  {:6.1} TF/s  {:6.1} GB/s",
            t.0,
            t.1,
            t.2,
            flops / t.0 / 1e6,
            bytes / t.0 / 1e3
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let timed = !std::env::args().any(|a| a == "--no-timing");
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    let gpu = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &gpu;
    let k = HeadBmmKernels::resolve(g)?;
    let base = mla::used_mb(g)?;
    println!(
        "glm5next_mla_head_bmm gate: {HEADS} heads, f64 reference, device in use {base:.0} MB at start\n"
    );
    let (mut ok, mut peak) = (true, 0f64);
    for (which, seed) in [(HeadBmm::Q, 0x51), (HeadBmm::O, 0x0e)] {
        let d = alloc(g, which, seed)?;
        peak = peak.max(mla::used_mb(g)? - base);
        ok &= vs_f64(g, &k, which, &d)?;
        ok &= row_identity(g, &k, which, &d)?;
        if which == HeadBmm::Q {
            ok &= negctl(g, &k, &d)?;
        }
        if timed {
            timing(g, &k, which, &d)?;
        }
        free(g, d)?;
        println!();
    }
    println!(
        "peak device memory of this gate's buffers: {peak:.0} MB above the {base:.0} MB at start"
    );
    if !ok {
        bail!("glm5next_mla_head_bmm gate FAILED");
    }
    println!("glm5next_mla_head_bmm gate PASSED");
    Ok(())
}
