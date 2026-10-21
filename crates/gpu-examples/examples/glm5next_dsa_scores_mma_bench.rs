// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Numerics microtest and microbench of `dsa_index_scores_mma` (the
//! `METRALE_GLM_DSA_SCORES_MMA` path: BF16 tensor cores, FP32 accumulation) against
//! `dsa_index_scores` (FP32 CUDA cores), on a prefill sub-chunk's selection: GLM-5.3's indexer
//! (32 heads of 128, pools of 4 tokens), Q query rows at the end of a context of S tokens.
//!
//! 1. Candidacy: the bytes must be identical (the same rule).
//! 2. Scores of candidates: max |error| over the row's max |score|, and the relative RMS.
//! 3. Selection: `dsa_topk_pools` on both score arrays; per query row, the fraction of the
//!    reference's selected pools the MMA selection also holds (mean and minimum over rows).
//! 4. A negative control: one changed query value must change the MMA scores.
//! 5. Timing (median of 7 synchronous launches): `dsa_index_scores`, `dsa_index_scores_tiled`
//!    and `dsa_index_scores_mma`.
//!
//! Owner: model-arch examples.
//! Invariants: returns an error unless the candidacy is identical and the negative control
//! changes the scores. The numerics are reported, not gated: the gate is the serve's accuracy
//! microtest.
//!
//! Run (GB10, < 2 GB of device memory):
//!   GLM_BENCH_SHAPES=2048x4096,2048x32768 cargo run --release -p metrale-gpu-examples \
//!     --features cuda,gpu-examples --example glm5next_dsa_scores_mma_bench

use anyhow::{Result, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const H: usize = 32;
const D: usize = 128;
const KP: usize = 4;
const TOPK: usize = 2048;

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
    // 2026-10-12: SAFETY: `v` is a live slice of `Copy` plain data; the byte view covers exactly
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

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn i32s(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

struct Io {
    keys: DevicePtr,
    w: DevicePtr,
    idx: DevicePtr,
    pvalid: DevicePtr,
    qpos: DevicePtr,
    out: DevicePtr,
    cand: DevicePtr,
    q_rows: usize,
    pools: usize,
    s: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Base,
    Tiled,
    Mma,
}

fn launch(gpu: &dyn GpuBackend, k: KernelHandle, kind: Kind, io: &Io, q: DevicePtr) -> Result<()> {
    let (grid, block) = match kind {
        Kind::Base => ([io.pools as u32, io.q_rows as u32, 1], 128),
        Kind::Tiled => (
            [
                io.pools.div_ceil(64) as u32,
                io.q_rows.div_ceil(32) as u32,
                1,
            ],
            256,
        ),
        Kind::Mma => ([io.q_rows.div_ceil(8) as u32, 1, 1], 256),
    };
    let l = KernelLaunch::new(gpu, k)
        .grid(grid)
        .block([block, 1, 1])
        .arg_ptr(q)
        .arg_ptr(io.keys)
        .arg_ptr(io.w)
        .arg_ptr(io.idx)
        .arg_ptr(io.pvalid)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(io.qpos)
        .arg_ptr(io.out)
        .arg_ptr(io.cand)
        .arg_u32(io.q_rows as u32)
        .arg_u32(io.pools as u32)
        .arg_u32(H as u32)
        .arg_u32(D as u32)
        .arg_u32(KP as u32)
        .arg_u32(io.s as u32)
        .arg_f32((D as f32).powf(-0.5));
    match kind {
        Kind::Base => l
            .shared_mem(128.max((H * 4) as u32))
            .arg_ptr(DevicePtr::NULL)
            .launch(0),
        Kind::Tiled => l.shared_mem(128 * 96 * 4).launch(0),
        Kind::Mma => l.launch(0),
    }
}

/// 2026-10-12: `dsa_topk_pools` over `scores` into `selected` (`select::select_tokens`'s launch).
fn topk(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    scores: DevicePtr,
    selected: DevicePtr,
    q_rows: usize,
    pools: usize,
    select_k: usize,
) -> Result<()> {
    let np2 = pools.next_power_of_two().clamp(2, 2048);
    KernelLaunch::new(gpu, k)
        .grid([q_rows as u32, 1, 1])
        .block([256, 1, 1])
        .shared_mem((np2 * 16) as u32)
        .arg_ptr(scores)
        .arg_ptr(selected)
        .arg_u32(q_rows as u32)
        .arg_u32(pools as u32)
        .arg_u32(np2 as u32)
        .arg_u32(select_k as u32)
        .arg_ptr(DevicePtr::NULL)
        .launch(0)
}

/// 2026-10-12: `dsa_topk_pools_radix` over `scores` into `selected`.
fn radix(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    scores: DevicePtr,
    selected: DevicePtr,
    q_rows: usize,
    pools: usize,
    select_k: usize,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([q_rows as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(scores)
        .arg_ptr(selected)
        .arg_u32(q_rows as u32)
        .arg_u32(pools as u32)
        .arg_u32(select_k as u32)
        .arg_ptr(DevicePtr::NULL)
        .launch(0)
}

/// 2026-10-12: Rows that stress the radix top-k's tie and edge handling, each byte-compared with
/// `dsa_topk_pools`: every score equal; many ties at the cut; +0.0 and -0.0 mixed; a short row
/// of non-candidates (-FLT_MAX); fewer pools than the selection; one pool.
fn edge_cases(gpu: &dyn GpuBackend, base: KernelHandle, rx: KernelHandle) -> Result<()> {
    let mut rng = Rng(0x5be0_cd19_137e_2179);
    let mut cases: Vec<(&str, Vec<f32>, usize)> = Vec::new();
    cases.push(("all equal", vec![1.25; 3000], 512));
    cases.push((
        "ties at the cut",
        (0..5000).map(|i| (i % 7) as f32).collect(),
        512,
    ));
    cases.push((
        "signed zeros",
        (0..4000)
            .map(|i| match i % 3 {
                0 => 0.0,
                1 => -0.0,
                _ => rng.unit(),
            })
            .collect(),
        512,
    ));
    cases.push((
        "non-candidates",
        (0..2000)
            .map(|i| if i % 5 == 0 { rng.unit() } else { f32::MIN })
            .collect(),
        512,
    ));
    cases.push(("short row", (0..300).map(|_| rng.unit()).collect(), 300));
    cases.push(("one pool", vec![0.5], 1));
    cases.push((
        "random wide",
        (0..20_000).map(|_| rng.unit()).collect(),
        512,
    ));
    for (name, row, select_k) in cases {
        let pools = row.len();
        let rows = 3;
        let all: Vec<f32> = (0..rows).flat_map(|_| row.iter().copied()).collect();
        let sc = up(gpu, &all)?;
        let (sa, sb) = (
            gpu.alloc(rows * select_k * 4)?,
            gpu.alloc(rows * select_k * 4)?,
        );
        topk(gpu, base, sc, sa, rows, pools, select_k)?;
        radix(gpu, rx, sc, sb, rows, pools, select_k)?;
        ensure!(
            read(gpu, sa, rows * select_k * 4)? == read(gpu, sb, rows * select_k * 4)?,
            "radix top-k edge case `{name}` differs from dsa_topk_pools"
        );
        for p in [sc, sa, sb] {
            gpu.free(p)?;
        }
    }
    println!("radix top-k: edge cases identical to dsa_topk_pools");
    Ok(())
}

/// 2026-10-21: The ceiling (captured decode) launch: scalar arguments at a 1M-token ceiling,
/// the live pools in a device geometry (`dsa_write_geom`'s slots). At each live pool count the
/// radix top-k must write the bitonic walk's bytes, untouched slots included; then both are
/// timed at decode pool counts (one row, as one DSA layer's selection of one decode row).
fn ceiling_cases(gpu: &dyn GpuBackend, base: KernelHandle, rx: KernelHandle) -> Result<()> {
    const MAX_POOLS: usize = 262_144;
    const SEL: usize = TOPK / KP;
    let mut rng = Rng(0x1f83_d9ab_fb41_bd6b);
    let scores: Vec<f32> = (0..MAX_POOLS)
        .map(|i| match i % 11 {
            0 => f32::MIN,
            1 | 2 => (i % 5) as f32 * 0.25,
            _ => rng.unit(),
        })
        .collect();
    let sc = up(gpu, &scores)?;
    let geom = gpu.alloc(5 * 4)?;
    let (sa, sb) = (gpu.alloc(SEL * 4)?, gpu.alloc(SEL * 4)?);
    let run = |radix_select: bool, out: DevicePtr| -> Result<()> {
        if !radix_select {
            KernelLaunch::new(gpu, base)
                .grid([1, 1, 1])
                .block([256, 1, 1])
                .shared_mem((2048 * 16) as u32)
                .arg_ptr(sc)
                .arg_ptr(out)
                .arg_u32(1)
                .arg_u32(MAX_POOLS as u32)
                .arg_u32(2048)
                .arg_u32(SEL as u32)
                .arg_ptr(geom)
                .launch(0)
        } else {
            KernelLaunch::new(gpu, rx)
                .grid([1, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(sc)
                .arg_ptr(out)
                .arg_u32(1)
                .arg_u32(MAX_POOLS as u32)
                .arg_u32(SEL as u32)
                .arg_ptr(geom)
                .launch(0)
        }
    };
    let set_geom = |live: usize| -> Result<()> {
        let np2 = live.next_power_of_two().clamp(2, 2048) as i32;
        let g = [(live * KP) as i32, live as i32 + 1, live as i32, live.min(SEL) as i32, np2];
        let b: Vec<u8> = g.iter().flat_map(|v| v.to_le_bytes()).collect();
        gpu.copy_h2d(&b, geom)?;
        Ok(())
    };
    for live in [0usize, 1, 2, 300, 511, 512, 513, 4_096, 32_768, 131_072, MAX_POOLS] {
        set_geom(live)?;
        gpu.memset(sa, 0x7f, SEL * 4)?;
        gpu.memset(sb, 0x7f, SEL * 4)?;
        run(false, sa)?;
        run(true, sb)?;
        ensure!(
            read(gpu, sa, SEL * 4)? == read(gpu, sb, SEL * 4)?,
            "ceiling launch at {live} live pools: radix top-k differs from dsa_topk_pools"
        );
    }
    println!("radix top-k: ceiling launches identical to dsa_topk_pools at 0..262144 live pools");
    for ctx in [1_024usize, 16_384, 65_536, 131_072, 524_288, 1_048_576] {
        let live = ctx / KP;
        set_geom(live)?;
        let tb = median_us(gpu, || run(false, sa))?;
        let tr = median_us(gpu, || run(true, sb))?;
        println!(
            "  ceiling top-k at {ctx:8} ctx ({live:6} pools): dsa_topk_pools {tb:9.1} us | radix {tr:8.1} us ({:.1}x)",
            tb / tr
        );
    }
    for p in [sc, geom, sa, sb] {
        gpu.free(p)?;
    }
    Ok(())
}

fn median_us(gpu: &dyn GpuBackend, f: impl Fn() -> Result<()>) -> Result<f64> {
    let mut v = Vec::new();
    for _ in 0..7 {
        gpu.synchronize(0)?;
        let t0 = std::time::Instant::now();
        f()?;
        gpu.synchronize(0)?;
        v.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    v.sort_by(f64::total_cmp);
    Ok(v[3])
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let base = gpu.kernel("dsa_indexer", "dsa_index_scores")?;
    let tiled = gpu.kernel("dsa_indexer_rows", "dsa_index_scores_tiled")?;
    let mma = gpu.kernel("dsa_indexer_rows", "dsa_index_scores_mma")?;
    let topk_k = gpu.kernel("dsa_indexer", "dsa_topk_pools")?;
    let topk_rx = gpu.kernel("dsa_indexer_rows", "dsa_topk_pools_radix")?;
    edge_cases(&gpu, topk_k, topk_rx)?;
    ceiling_cases(&gpu, topk_k, topk_rx)?;
    let shapes: Vec<(usize, usize)> = std::env::var("GLM_BENCH_SHAPES")
        .ok()
        .map(|v| {
            v.split(',')
                .filter_map(|x| {
                    let (a, b) = x.trim().split_once('x')?;
                    Some((a.parse().ok()?, b.parse().ok()?))
                })
                .collect()
        })
        .unwrap_or_else(|| vec![(2048, 4096), (2048, 32768)]);
    let mut rng = Rng(0x3c6e_f372_fe94_f82b);
    for (q_rows, s) in shapes {
        let pools = s / KP;
        let select_k = (TOPK / KP).min(pools);
        let q: Vec<f32> = (0..q_rows * H * D).map(|_| rng.unit()).collect();
        let keys: Vec<f32> = (0..pools * D).map(|_| rng.unit()).collect();
        let w: Vec<f32> = (0..q_rows * H).map(|_| rng.unit() * 0.2).collect();
        let idx: Vec<i32> = (0..(pools * KP) as i32).collect();
        let pvalid = vec![1u8; pools];
        let qpos: Vec<i32> = (0..q_rows).map(|r| (s - q_rows + r) as i32).collect();
        let qd = up(&gpu, &q)?;
        let io = Io {
            keys: up(&gpu, &keys)?,
            w: up(&gpu, &w)?,
            idx: up(&gpu, &idx)?,
            pvalid: up(&gpu, &pvalid)?,
            qpos: up(&gpu, &qpos)?,
            out: gpu.alloc(q_rows * pools * 4)?,
            cand: gpu.alloc(q_rows * pools)?,
            q_rows,
            pools,
            s,
        };
        let sel_ref = gpu.alloc(q_rows * select_k * 4)?;
        let sel_mma = gpu.alloc(q_rows * select_k * 4)?;
        launch(&gpu, base, Kind::Base, &io, qd)?;
        topk(&gpu, topk_k, io.out, sel_ref, q_rows, pools, select_k)?;
        let (o_ref, c_ref) = (
            f32s(&read(&gpu, io.out, q_rows * pools * 4)?),
            read(&gpu, io.cand, q_rows * pools)?,
        );
        gpu.memset(io.out, 0, q_rows * pools * 4)?;
        gpu.memset(io.cand, 0x55, q_rows * pools)?;
        launch(&gpu, mma, Kind::Mma, &io, qd)?;
        topk(&gpu, topk_k, io.out, sel_mma, q_rows, pools, select_k)?;
        let (o, c) = (
            f32s(&read(&gpu, io.out, q_rows * pools * 4)?),
            read(&gpu, io.cand, q_rows * pools)?,
        );
        ensure!(c == c_ref, "Q={q_rows} S={s}: candidacy differs");
        let (mut worst, mut se, mut sr) = (0.0f64, 0.0f64, 0.0f64);
        for r in 0..q_rows {
            let row = r * pools..(r + 1) * pools;
            let rmax = o_ref[row.clone()]
                .iter()
                .zip(&c_ref[row.clone()])
                .filter(|(_, c)| **c != 0)
                .fold(0.0f64, |m, (v, _)| m.max(v.abs() as f64));
            for i in row {
                if c_ref[i] == 0 {
                    ensure!(o[i] == f32::MIN, "Q={q_rows} S={s}: a non-candidate scored");
                    continue;
                }
                let e = (o[i] - o_ref[i]) as f64;
                se += e * e;
                sr += (o_ref[i] as f64).powi(2);
                if rmax > 0.0 {
                    worst = worst.max(e.abs() / rmax);
                }
            }
        }
        let (a, b) = (
            i32s(&read(&gpu, sel_ref, q_rows * select_k * 4)?),
            i32s(&read(&gpu, sel_mma, q_rows * select_k * 4)?),
        );
        let (mut mean, mut min) = (0.0f64, 1.0f64);
        for r in 0..q_rows {
            let ra = &a[r * select_k..(r + 1) * select_k];
            let mut rb: Vec<i32> = b[r * select_k..(r + 1) * select_k].to_vec();
            rb.sort_unstable();
            let live = ra.iter().filter(|&&p| p >= 0).count().max(1);
            let hit = ra
                .iter()
                .filter(|&&p| p >= 0 && rb.binary_search(&p).is_ok())
                .count();
            let f = hit as f64 / live as f64;
            mean += f;
            min = min.min(f);
        }
        mean /= q_rows as f64;
        // 2026-10-12: The radix top-k must select what dsa_topk_pools selects, in its order, on
        // both score arrays.
        let sel_rx = gpu.alloc(q_rows * select_k * 4)?;
        for (name, sc, want) in [("reference", &o_ref, &a), ("mma", &o, &b)] {
            let scd = up(&gpu, sc)?;
            radix(&gpu, topk_rx, scd, sel_rx, q_rows, pools, select_k)?;
            ensure!(
                i32s(&read(&gpu, sel_rx, q_rows * select_k * 4)?) == *want,
                "Q={q_rows} S={s}: radix top-k differs from dsa_topk_pools on the {name} scores"
            );
            gpu.free(scd)?;
        }
        gpu.free(sel_rx)?;
        let mut q2 = q.clone();
        q2[(q_rows - 1) * H * D] += 0.5;
        let q2d = up(&gpu, &q2)?;
        launch(&gpu, mma, Kind::Mma, &io, q2d)?;
        ensure!(
            f32s(&read(&gpu, io.out, q_rows * pools * 4)?) != o,
            "Q={q_rows} S={s}: negative control left the scores unchanged"
        );
        let tb = median_us(&gpu, || launch(&gpu, base, Kind::Base, &io, qd))?;
        let tt = median_us(&gpu, || launch(&gpu, tiled, Kind::Tiled, &io, qd))?;
        let tm = median_us(&gpu, || launch(&gpu, mma, Kind::Mma, &io, qd))?;
        let tk = median_us(&gpu, || {
            topk(&gpu, topk_k, io.out, sel_mma, q_rows, pools, select_k)
        })?;
        let tr = median_us(&gpu, || {
            radix(&gpu, topk_rx, io.out, sel_mma, q_rows, pools, select_k)
        })?;
        println!(
            "Q={q_rows:5} S={s:6} pools={pools:5}: candidacy identical; scores max err {worst:.2e} \
             of row max, rel RMS {:.2e}; top-{select_k} overlap mean {mean:.4} min {min:.4}",
            (se / sr.max(1e-30)).sqrt()
        );
        println!(
            "  dsa_index_scores {tb:10.1} us | tiled {tt:10.1} us | mma {tm:10.1} us \
             ({:.1}x base, {:.1}x tiled)",
            tb / tm,
            tt / tm
        );
        println!(
            "  dsa_topk_pools {tk:10.1} us | radix {tr:10.1} us ({:.1}x)",
            tk / tr
        );
        for p in [
            qd, q2d, io.keys, io.w, io.idx, io.pvalid, io.qpos, io.out, io.cand, sel_ref, sel_mma,
        ] {
            gpu.free(p)?;
        }
    }
    Ok(())
}
