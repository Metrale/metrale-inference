// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The vocab-parallel FP8 LM head (`layers::fp8_row_head` with a shard window) at
//! GLM-5.3's shape on three simulated TP ranks, against the whole FP8 head, and timed.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Returns an error unless, on the W8A16 route and (when its kernels resolve) the W8A8
//!   route, at 1, 2, 4 and 16 rows: (a) each rank's window holds the whole head's bytes there
//!   and every column outside it is untouched (sentinel); (b) the production composition, each
//!   rank into a zeroed buffer and the three summed in FP32 and rounded to BF16 (what the
//!   all-reduce computes), is the whole head byte for byte.
//! - Negative control: rank 1's window at 16 rows written with the dense pitch (`ldc = len`,
//!   the pre-window launcher) must fail (a), so the comparison can fail.
//!
//! Shape: hidden 4096, vocab 154,856 (the tokenizer-capped GLM-5.3 head), split as
//! `lmhead_vocab_split(154_856, 3, r)`: (0, 51,648), (51,648, 51,648), (103,296, 51,560). The
//! weights are random E4M3 bytes (NaN codes excluded) with row scales 2^-10 .. 2^-3; the
//! activations span 17 binary orders, so FP32 sums round and a changed K order would show.
//!
//! Times (rank 0's window, the slowest): median of 7 batches of 20 launches, host-timed, with
//! the batch spread. Every launch reads its weights from DRAM (>= 211 MB per launch).
//!
//! Run (GB10, ~1.2 GB of device memory):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example glm53_lmhead_fp8_shard_microtest

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops;
use metrale_model_layers::layers::{
    Fp8RowHeadKernels, HeadWindow, W8a8Ctx, fp8_row_head, try_kernel,
};
use metrale_model_layers::weight_map::{DenseWeight, Fp8DenseWeight};

const K: usize = 4096;
const V: usize = 154_856;
const SHARDS: [(usize, usize); 3] = [(0, 51_648), (51_648, 51_648), (103_296, 51_560)];
const ROWS: [usize; 4] = [1, 2, 4, 16];
const MAX_ROWS: usize = 16;
const SENTINEL: u8 = 0x5a;
const S: u64 = 0;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn read(gpu: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    gpu.synchronize(S)?;
    let mut b = vec![0u8; bytes];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn bf16_at(b: &[u8], i: usize) -> f32 {
    bf16::from_bits(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])).to_f32()
}

/// 2026-10-09: Mismatches of rank `(begin, len)`'s output against the whole head's: window
/// bytes that differ, plus columns outside the window that are not the sentinel.
fn window_mismatches(got: &[u8], whole: &[u8], rows: usize, (begin, len): (usize, usize)) -> usize {
    let mut bad = 0;
    for r in 0..rows {
        for c in 0..V {
            let i = 2 * (r * V + c);
            let g = &got[i..i + 2];
            bad += usize::from(if (begin..begin + len).contains(&c) {
                g != &whole[i..i + 2]
            } else {
                g != [SENTINEL; 2]
            });
        }
    }
    bad
}

/// 2026-10-09: What the all-reduce computes from the three zero-backed rank buffers.
fn summed(parts: &[Vec<u8>], elems: usize) -> Vec<u8> {
    (0..elems)
        .flat_map(|i| {
            let s: f32 = parts.iter().map(|p| bf16_at(p, i)).sum();
            bf16::from_f32(s).to_bits().to_le_bytes()
        })
        .collect()
}

struct Setup {
    kn: Fp8RowHeadKernels,
    w8a8: Option<W8a8Ctx>,
    fp8: Fp8DenseWeight,
    x: DevicePtr,
    whole: DevicePtr,
    rank: [DevicePtr; 3],
}

fn head(
    gpu: &dyn GpuBackend,
    s: &Setup,
    w8a8: bool,
    rows: usize,
    out: DevicePtr,
    win: HeadWindow,
) -> Result<()> {
    let ctx = if w8a8 { s.w8a8.as_ref() } else { None };
    fp8_row_head(gpu, &s.kn, ctx, &s.fp8, s.x, K, rows, out, win, S)
}

/// 2026-10-09: The byte gate for one route at one row count; returns the whole head's bytes.
fn gate(gpu: &dyn GpuBackend, s: &Setup, w8a8: bool, rows: usize) -> Result<Vec<u8>> {
    let bytes = rows * V * 2;
    gpu.memset(s.whole, SENTINEL, bytes)?;
    head(gpu, s, w8a8, rows, s.whole, HeadWindow::whole(V))?;
    let whole = read(gpu, s.whole, bytes)?;
    let route = if w8a8 { "W8A8" } else { "W8A16" };
    let mut zeroed = Vec::new();
    for (r, &(begin, len)) in SHARDS.iter().enumerate() {
        let win = HeadWindow { begin, len, ldc: V };
        gpu.memset(s.rank[r], SENTINEL, bytes)?;
        head(gpu, s, w8a8, rows, s.rank[r], win)?;
        let bad = window_mismatches(&read(gpu, s.rank[r], bytes)?, &whole, rows, (begin, len));
        ensure!(
            bad == 0,
            "{route} rows={rows} rank {r}: {bad} mismatched columns"
        );
        gpu.memset(s.rank[r], 0, bytes)?;
        head(gpu, s, w8a8, rows, s.rank[r], win)?;
        zeroed.push(read(gpu, s.rank[r], bytes)?);
    }
    let sum = summed(&zeroed, rows * V);
    let diff = sum
        .chunks(2)
        .zip(whole.chunks(2))
        .filter(|(a, b)| a != b)
        .count();
    ensure!(
        diff == 0,
        "{route} rows={rows}: rank sum differs from the whole head at {diff} logits"
    );
    println!(
        "  {route:5} rows={rows:2}: 3 windows + rank sum == whole head, {} logits byte-identical",
        rows * V
    );
    Ok(whole)
}

/// 2026-10-09: Rank 1's window at 16 rows through the dense-pitch launcher must be caught.
fn negative_control(gpu: &dyn GpuBackend, s: &Setup, whole16: &[u8]) -> Result<()> {
    let (begin, len) = SHARDS[1];
    let bytes = MAX_ROWS * V * 2;
    gpu.memset(s.rank[1], SENTINEL, bytes)?;
    let shard = Fp8DenseWeight {
        weight: s.fp8.weight.offset(begin * K),
        row_scale: s.fp8.row_scale.offset(begin * 4),
    };
    let out = s.rank[1].offset(begin * 2);
    let (m, n) = (MAX_ROWS as u32, len as u32);
    ops::fp8_gemv_rowscale_batch16_rt2(gpu, s.kn.rt16, s.x, &shard, out, m, n, K as u32, S)?;
    let bad = window_mismatches(
        &read(gpu, s.rank[1], bytes)?,
        whole16,
        MAX_ROWS,
        (begin, len),
    );
    ensure!(
        bad > 0,
        "negative control: a dense-pitch window passed the byte gate"
    );
    println!("  negative control: dense-pitch window caught ({bad} mismatched columns)");
    Ok(())
}

fn median_us(gpu: &dyn GpuBackend, mut f: impl FnMut() -> Result<()>) -> Result<(f64, f64, f64)> {
    let mut t = Vec::new();
    f()?;
    for _ in 0..7 {
        gpu.synchronize(S)?;
        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            f()?;
        }
        gpu.synchronize(S)?;
        t.push(t0.elapsed().as_secs_f64() * 1e6 / 20.0);
    }
    t.sort_by(f64::total_cmp);
    Ok((t[3], t[0], t[6]))
}

fn timing(gpu: &dyn GpuBackend, s: &Setup, rng: &mut Rng) -> Result<()> {
    let (begin, len) = SHARDS[0];
    // 2026-10-09: Rank 0's BF16 rows only (the sharded BF16 head reads no others).
    let w: Vec<u8> = (0..len * K)
        .flat_map(|_| {
            bf16::from_f32((rng.next() as i32 as f32) * 2e-12)
                .to_bits()
                .to_le_bytes()
        })
        .collect();
    let bf16_w = DenseWeight {
        weight: upload(gpu, &w)?,
    };
    drop(w);
    let gemv = gpu.kernel("gemv", "dense_gemv_bf16")?;
    let batchm = try_kernel(gpu, "dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm");
    let out = s.rank[0];
    println!("\ntimes (us): median [min, max] of 7 batches x 20; rank 0 window = {len} rows");
    for rows in ROWS {
        let (n, k, m) = (len as u32, K as u32, rows as u32);
        let o = out.offset(begin * 2);
        let bf = median_us(gpu, || {
            if rows == 1 {
                ops::dense_gemv(gpu, gemv, s.x, &bf16_w, o, n, k, S)
            } else {
                ensure!(batchm.0 != 0, "dense_gemv_bf16_batchm not compiled");
                ops::dense_gemv_batchm(gpu, batchm, s.x, &bf16_w, o, m, n, k, V as u32, S)
            }
        })?;
        let win = HeadWindow { begin, len, ldc: V };
        let f8 = median_us(gpu, || head(gpu, s, false, rows, out, win))?;
        let f8w = median_us(gpu, || {
            head(gpu, s, false, rows, s.whole, HeadWindow::whole(V))
        })?;
        let p = |(a, b, c): (f64, f64, f64)| format!("{a:8.1} [{b:7.1}, {c:7.1}]");
        println!(
            "  rows={rows:2}  BF16 shard {}  FP8 shard {}  FP8 whole {}",
            p(bf),
            p(f8),
            p(f8w)
        );
        if s.w8a8.is_some() {
            println!(
                "           W8A8 shard {}",
                p(median_us(gpu, || head(gpu, s, true, rows, out, win))?)
            );
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &gpu;
    let mut rng = Rng(0x6c6d_6865_6164_6638);
    let mut wb = vec![0u8; V * K];
    for c in wb.chunks_mut(4) {
        c.copy_from_slice(&rng.next().to_le_bytes());
    }
    for b in wb.iter_mut() {
        // 2026-10-09: E4M3 NaN is S.1111.111; clear its low bit.
        if *b & 0x7f == 0x7f {
            *b ^= 1;
        }
    }
    let scales: Vec<u8> = (0..V)
        .flat_map(|_| {
            let e = 117 + rng.next() % 8;
            f32::from_bits((e << 23) | (rng.next() & 0x7f_ffff)).to_le_bytes()
        })
        .collect();
    let fp8 = Fp8DenseWeight {
        weight: upload(gpu, &wb)?,
        row_scale: upload(gpu, &scales)?,
    };
    drop(wb);
    let x: Vec<u8> = (0..MAX_ROWS * K)
        .flat_map(|_| {
            let u = rng.next();
            let bits = ((u & 1) << 15) | ((115 + (u >> 1) % 17) << 7) | ((u >> 8) & 0x7f);
            (bits as u16).to_le_bytes()
        })
        .collect();
    let rt = |f| try_kernel(gpu, "fp8_gemv_rt", f);
    let kn = Fp8RowHeadKernels {
        gemv: gpu.kernel("gemv_fp8w", "dense_gemv_fp8w")?,
        batch2: try_kernel(gpu, "dense_gemv_fp8w_batch2", "dense_gemv_fp8w_batch2"),
        rt8: rt("fp8_gemv_rowscale_batch8_rt2"),
        rt16: rt("fp8_gemv_rowscale_batch16_rt2"),
    };
    ensure!(
        kn.batch2.0 != 0 && kn.rt8.0 != 0 && kn.rt16.0 != 0,
        "FP8 head kernels missing: {kn:?}"
    );
    let ctx = W8a8Ctx::new(gpu, K as u32)?;
    let w8a8 = ops::W8a8Weight::new(&[metrale_model_layers::weight_map::Fp8Weight {
        weight: fp8.weight,
        row_scale: fp8.row_scale,
        n: V as u32,
        k: K as u32,
        scale_format: metrale_model_layers::weight_map::WeightQuantFormat::Fp8PerRow,
    }])?;
    let out_bytes = MAX_ROWS * V * 2;
    let s = Setup {
        kn,
        w8a8: ctx.available(&w8a8, MAX_ROWS).then_some(ctx),
        fp8,
        x: upload(gpu, &x)?,
        whole: gpu.alloc(out_bytes)?,
        rank: [
            gpu.alloc(out_bytes)?,
            gpu.alloc(out_bytes)?,
            gpu.alloc(out_bytes)?,
        ],
    };
    if s.w8a8.is_none() {
        println!("W8A8 kernels not compiled for this target: W8A8 route skipped");
    }
    println!("byte gate: vocab {V}, hidden {K}, 3 ranks {SHARDS:?}");
    let mut whole16 = Vec::new();
    for rows in ROWS {
        whole16 = gate(gpu, &s, false, rows)?;
        if s.w8a8.is_some() {
            gate(gpu, &s, true, rows)?;
        }
    }
    negative_control(gpu, &s, &whole16)?;
    timing(gpu, &s, &mut rng)
}
