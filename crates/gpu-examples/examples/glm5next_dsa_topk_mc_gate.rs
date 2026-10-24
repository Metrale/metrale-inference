// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Byte gate and microbench of the multi-CTA radix top-k (`dsa_topk_mc_pass` x 4 +
//! `dsa_topk_mc_collect`, `METRALE_GLM_DSA_TOPK_MC`) against `dsa_topk_pools_radix`, both as the
//! captured decode's ceiling launch: a 1M-token ceiling (262,144 pools as the scores' row
//! stride, select_k 512 as the selected row stride), each row's live pools and select_k in its
//! device geometry.
//!
//! 1. Identity: the selected rows (the whole ceiling stride, from the same sentinel fill) must
//!    be identical for 1, 2 and 3 rows, CTA counts 1, 7, 16, 48 and 64, on random scores and on
//!    adversarial rows: all equal, ties at the cut, signed zeros, non-candidates (-FLT_MAX),
//!    a few distinct values, rows at or below select_k (0, 1, 511, 512, 513 pools), rows whose
//!    live counts differ. Then the same workspace is reused for a second, different row set
//!    without clearing (replay safety).
//! 2. Timing: each selection captured 20 times in a CUDA graph, replayed; us per selection.
//!
//! 2026-10-24: And the persistent segment `dsa_topk_mc_seg` (`--glm-dsa-topk-seg`, R4): at every
//! row set and CTA count above it must select the same bytes on `C x rows` blocks, and its
//! control block must read back zero after every launch (replay safety, here and after each
//! timed graph). `sabotage` as the first argument presets the control block's phase-0 count, so
//! pass 1 runs without waiting for pass 0: the gate must then fail.
//!
//! Owner: model-arch examples.
//! Invariants: returns an error unless every compared byte matches.
//!
//! Run (GB10): cargo run --release -p metrale-gpu-examples --features cuda,gpu-examples \
//!   --example glm5next_dsa_topk_mc_gate [-- sabotage]

use anyhow::{Result, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const MAX_POOLS: usize = 262_144;
const SEL: usize = 512;
const REPS: usize = 20;

/// 2026-10-21: `DSA_MC_WS_WORDS(C)` of dsa_topk_mc.cu, in bytes.
fn ws_bytes(c: usize) -> usize {
    (c * 256 + 768 + 2 + 2 * SEL + c + c * SEL + 16) * 4
}

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

fn up<T: Copy>(gpu: &dyn GpuBackend, v: &[T]) -> Result<DevicePtr> {
    let n = std::mem::size_of_val(v);
    // 2026-10-21: SAFETY: `v` is a live slice of `Copy` plain data; the byte view covers exactly
    // its `n` bytes and is only read.
    let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n) };
    let p = gpu.alloc(n.max(16))?;
    gpu.copy_h2d(b, p)?;
    Ok(p)
}

fn read(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    gpu.synchronize(0)?;
    let mut b = vec![0u8; n];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

struct Kernels {
    radix: KernelHandle,
    pass: KernelHandle,
    collect: KernelHandle,
    seg: KernelHandle,
}

/// 2026-10-24: Bytes of `SegCtlN<5>` (ticket, exited, gated, done[5]).
const SEG_CTL_BYTES: usize = 4 * 8;

/// 2026-10-21: Scores of `rows` rows at the ceiling stride, the device geometry of each.
struct Rows {
    sc: DevicePtr,
    geom: DevicePtr,
    rows: usize,
    live: Vec<usize>,
}

fn rows_of(gpu: &dyn GpuBackend, data: &[Vec<f32>]) -> Result<Rows> {
    let mut all = vec![0.0f32; data.len() * MAX_POOLS];
    let mut geom = Vec::new();
    for (r, d) in data.iter().enumerate() {
        all[r * MAX_POOLS..r * MAX_POOLS + d.len()].copy_from_slice(d);
        let np = d.len();
        geom.extend_from_slice(&[
            (np * 4) as i32,
            np as i32 + 1,
            np as i32,
            np.min(SEL) as i32,
            2048,
        ]);
    }
    Ok(Rows {
        sc: up(gpu, &all)?,
        geom: up(gpu, &geom)?,
        rows: data.len(),
        live: data.iter().map(Vec::len).collect(),
    })
}

fn radix(gpu: &dyn GpuBackend, k: &Kernels, r: &Rows, out: DevicePtr, st: u64) -> Result<()> {
    KernelLaunch::new(gpu, k.radix)
        .grid([r.rows as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(r.sc)
        .arg_ptr(out)
        .arg_u32(r.rows as u32)
        .arg_u32(MAX_POOLS as u32)
        .arg_u32(SEL as u32)
        .arg_ptr(r.geom)
        .launch(st)
}

fn mc(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    r: &Rows,
    ws: DevicePtr,
    c: usize,
    out: DevicePtr,
    st: u64,
) -> Result<()> {
    for pass in 0..4u32 {
        KernelLaunch::new(gpu, k.pass)
            .grid([c as u32, r.rows as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(r.sc)
            .arg_ptr(ws)
            .arg_u32(r.rows as u32)
            .arg_u32(MAX_POOLS as u32)
            .arg_u32(SEL as u32)
            .arg_ptr(r.geom)
            .arg_u32(pass)
            .launch(st)?;
    }
    KernelLaunch::new(gpu, k.collect)
        .grid([c as u32, r.rows as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(r.sc)
        .arg_ptr(ws)
        .arg_ptr(out)
        .arg_u32(r.rows as u32)
        .arg_u32(MAX_POOLS as u32)
        .arg_u32(SEL as u32)
        .arg_ptr(r.geom)
        .launch(st)
}

/// 2026-10-24: The five launches of [`mc`] as one `dsa_topk_mc_seg` launch on `c x rows` blocks
/// (the launch `select::topk::launch_mc_seg` makes). `preset` (the sabotage) first writes the
/// control block's phase-0 count as already complete.
#[allow(clippy::too_many_arguments)]
fn seg(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    r: &Rows,
    ws: DevicePtr,
    c: usize,
    out: DevicePtr,
    ctl: DevicePtr,
    preset: bool,
    st: u64,
) -> Result<()> {
    if preset {
        let mut w = [0u32; 8];
        w[3] = (c * r.rows) as u32;
        let b: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
        gpu.copy_h2d(&b, ctl)?;
    }
    KernelLaunch::new(gpu, k.seg)
        .grid([(c * r.rows) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(r.sc)
        .arg_ptr(ws)
        .arg_ptr(out)
        .arg_u32(r.rows as u32)
        .arg_u32(MAX_POOLS as u32)
        .arg_u32(SEL as u32)
        .arg_ptr(r.geom)
        .arg_u32(c as u32)
        .arg_ptr(ctl)
        .launch(st)
}

fn ctl_zero(gpu: &dyn GpuBackend, ctl: DevicePtr) -> Result<()> {
    let b = read(gpu, ctl, SEG_CTL_BYTES)?;
    ensure!(
        b.iter().all(|&x| x == 0),
        "dsa_topk_mc_seg left its control block at {b:?}"
    );
    Ok(())
}

fn graph_us(gpu: &dyn GpuBackend, f: &dyn Fn(u64) -> Result<()>) -> Result<f64> {
    let s = gpu.create_stream()?;
    f(s)?;
    gpu.synchronize(s)?;
    gpu.begin_capture(s)?;
    for _ in 0..REPS {
        f(s)?;
    }
    let g = gpu.end_capture(s)?;
    let mut v = Vec::new();
    for _ in 0..5 {
        gpu.synchronize(s)?;
        let t0 = std::time::Instant::now();
        gpu.launch_graph(g, s)?;
        gpu.synchronize(s)?;
        v.push(t0.elapsed().as_secs_f64() * 1e6 / REPS as f64);
    }
    gpu.destroy_graph(g)?;
    v.sort_by(f64::total_cmp);
    Ok(v[2])
}

/// 2026-10-21: A named row of `n` pools.
fn row(rng: &mut Rng, kind: usize, n: usize) -> (String, Vec<f32>) {
    let v: Vec<f32> = (0..n)
        .map(|i| match kind {
            0 => rng.unit(),
            1 => 1.25,
            2 => (i % 7) as f32,
            3 => match i % 3 {
                0 => 0.0,
                1 => -0.0,
                _ => rng.unit(),
            },
            4 => {
                if i % 5 == 0 {
                    rng.unit()
                } else {
                    f32::MIN
                }
            }
            5 => [0.5f32, 0.25, 0.75][i % 3] + if i % 1000 == 999 { 1e-7 } else { 0.0 },
            6 => (rng.unit() * 1e6).round() / 1e6 * f32::MAX,
            _ => -(i as f32) * 1e-3,
        })
        .collect();
    (format!("kind {kind} n {n}"), v)
}

fn main() -> Result<()> {
    let sabotage = std::env::args().nth(1).as_deref() == Some("sabotage");
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let k = Kernels {
        radix: gpu.kernel("dsa_indexer_rows", "dsa_topk_pools_radix")?,
        pass: gpu.kernel("dsa_topk_mc", "dsa_topk_mc_pass")?,
        collect: gpu.kernel("dsa_topk_mc", "dsa_topk_mc_collect")?,
        seg: gpu.kernel("dsa_topk_mc_seg", "dsa_topk_mc_seg")?,
    };
    let ctl = gpu.alloc(SEG_CTL_BYTES)?;
    gpu.memset(ctl, 0, SEG_CTL_BYTES)?;
    let mut rng = Rng(0xbb67_ae85_84ca_a73b);
    let max_rows = 3;
    let ws = gpu.alloc(max_rows * ws_bytes(64))?;
    // 2026-10-21: Garbage in the workspace: nothing may depend on its contents.
    gpu.memset(ws, 0xc3, max_rows * ws_bytes(64))?;
    let (oa, ob) = (
        gpu.alloc(max_rows * SEL * 4)?,
        gpu.alloc(max_rows * SEL * 4)?,
    );
    let mut sets: Vec<Vec<(String, Vec<f32>)>> = Vec::new();
    for n in [0usize, 1, 300, 511, 512, 513, 2_000, 20_000, 131_072] {
        for kind in 0..8 {
            sets.push(vec![row(&mut rng, kind, n)]);
        }
    }
    for (a, b) in [
        (32_768usize, 32_767usize),
        (4_096, 513),
        (512, 100_000),
        (65_536, 65_535),
    ] {
        for kind in [0usize, 2, 4, 5] {
            sets.push(vec![
                row(&mut rng, kind, a),
                row(&mut rng, (kind + 1) % 8, b),
            ]);
            sets.push(vec![
                row(&mut rng, kind, a),
                row(&mut rng, kind, b),
                row(&mut rng, 3, a / 2),
            ]);
        }
    }
    let mut checked = 0;
    for set in &sets {
        let data: Vec<Vec<f32>> = set.iter().map(|(_, v)| v.clone()).collect();
        let r = rows_of(&gpu, &data)?;
        gpu.memset(oa, 0x7f, max_rows * SEL * 4)?;
        radix(&gpu, &k, &r, oa, 0)?;
        let want = read(&gpu, oa, r.rows * SEL * 4)?;
        for c in [1usize, 7, 16, 48, 64] {
            gpu.memset(ob, 0x7f, max_rows * SEL * 4)?;
            mc(&gpu, &k, &r, ws, c, ob, 0)?;
            let got = read(&gpu, ob, r.rows * SEL * 4)?;
            if got != want {
                let i = got.iter().zip(&want).position(|(x, y)| x != y).unwrap_or(0) / 4;
                let names: Vec<&str> = set.iter().map(|(n, _)| n.as_str()).collect();
                anyhow::bail!(
                    "{names:?} C={c}: multi-CTA top-k differs from dsa_topk_pools_radix at row {} slot {}",
                    i / SEL,
                    i % SEL
                );
            }
            checked += 1;
            gpu.memset(ob, 0x7f, max_rows * SEL * 4)?;
            seg(&gpu, &k, &r, ws, c, ob, ctl, sabotage, 0)?;
            let got = read(&gpu, ob, r.rows * SEL * 4)?;
            ctl_zero(&gpu, ctl)?;
            if got != want {
                let names: Vec<&str> = set.iter().map(|(n, _)| n.as_str()).collect();
                anyhow::bail!("{names:?} C={c}: dsa_topk_mc_seg differs from dsa_topk_pools_radix");
            }
            checked += 1;
        }
        gpu.free(r.sc)?;
        gpu.free(r.geom)?;
    }
    println!(
        "multi-CTA top-k: {checked} launches over {} row sets identical to dsa_topk_pools_radix",
        sets.len()
    );
    for ctx in [1_024usize, 16_384, 65_536, 131_072, 524_288] {
        for rows in [1usize, 2] {
            let n = ctx / 4;
            let data: Vec<Vec<f32>> = (0..rows)
                .map(|_| (0..n).map(|_| rng.unit().max(0.0) * 0.3).collect())
                .collect();
            let r = rows_of(&gpu, &data)?;
            let told = graph_us(&gpu, &|st| radix(&gpu, &k, &r, oa, st))?;
            let want = read(&gpu, oa, rows * SEL * 4)?;
            let mut line = format!(
                "  TIME ctx {ctx:7} rows {rows} ({} pools): radix {told:7.1} us |",
                r.live[0]
            );
            for c in [16usize, 32, 48, 64] {
                let t = graph_us(&gpu, &|st| mc(&gpu, &k, &r, ws, c, ob, st))?;
                ensure!(
                    read(&gpu, ob, rows * SEL * 4)? == want,
                    "ctx {ctx} rows {rows} C={c}: timed selection differs"
                );
                line += &format!(" mc C={c} {t:6.1} us");
                let t = graph_us(&gpu, &|st| seg(&gpu, &k, &r, ws, c, ob, ctl, false, st))?;
                ensure!(
                    read(&gpu, ob, rows * SEL * 4)? == want,
                    "ctx {ctx} rows {rows} C={c}: timed segment selection differs"
                );
                ctl_zero(&gpu, ctl)?;
                line += &format!(" seg {t:6.1} us");
            }
            println!("{line}");
            gpu.free(r.sc)?;
            gpu.free(r.geom)?;
        }
    }
    println!(
        "PASS: the multi-CTA top-k (five launches and the segment) selects dsa_topk_pools_radix's bytes"
    );
    Ok(())
}
