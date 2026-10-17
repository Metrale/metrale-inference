// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The two-stage DFlash2 top-16 (`dflash2_topk16_part` + `dflash2_topk16_merge`,
//! `kernels/gb10/common/dflash2.cu`) against the one-block `dflash2_topk16`.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless, on every case, the two-stage kernels write the same `top_vals`,
//!   `top_idx` and logits bytes (the selected entries overwritten) as `dflash2_topk16`, and the
//!   3-shard emulation of the tensor-parallel drafter's top-16 the same `top_vals` and
//!   `top_idx` (2026-10-10).
//! - The negative control (one top logit raised by one BF16 ulp in the two-stage input only)
//!   must change the bytes, so the comparison can fail.
//!
//! Cases, `rows` x the GLM-5.3 vocabulary (154880) unless stated: real-valued logits; logits
//! drawn from 24 BF16 values (hundreds of ties in the top 16); one value everywhere; NaN,
//! +inf, -inf and values at or below -1e30 mixed in; rows with fewer than 16 qualifying
//! entries (the (-1e30, 0) padding); a vocabulary that is not a multiple of the part span.
//! Times: median of 7 batches of 50 launches at 8 rows, host-timed.
//!
//! Run (GB10, < 64 MB of device memory):
//!   cargo run --release -p metrale-gpu-examples --features cuda,gpu-examples \
//!     --example dflash2_topk16_parity

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const VOCAB: u32 = 154_880;
/// 2026-10-10: `TOPK16_PART_SPAN` and `TOPK16_PART_THREADS` of `dflash2.cu`.
const SPAN: u32 = 2048;
const THREADS: u32 = 256;
/// 2026-10-10: Ranks of the emulated vocab split (`launch_sharded`).
const SHARDS: u32 = 3;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn unit(&mut self) -> f32 {
        self.next() as f32 / u32::MAX as f32
    }
}

struct Kernels {
    one: KernelHandle,
    part: KernelHandle,
    merge: KernelHandle,
}

struct Bufs {
    /// 2026-10-10: One shard's rows (`launch_sharded`), and the shards' gathered lists.
    shard: DevicePtr,
    gather: DevicePtr,
    logits: DevicePtr,
    vals: DevicePtr,
    idx: DevicePtr,
    cand_vals: DevicePtr,
    cand_idx: DevicePtr,
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect()
}

fn read(gpu: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    gpu.synchronize(0)?;
    let mut b = vec![0u8; bytes];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn launch_one(gpu: &dyn GpuBackend, k: &Kernels, b: &Bufs, rows: u32, n: u32) -> Result<()> {
    KernelLaunch::new(gpu, k.one)
        .grid([rows, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(b.logits)
        .arg_ptr(b.vals)
        .arg_ptr(b.idx)
        .arg_u32(n)
        .launch(0)
}

#[allow(clippy::too_many_arguments)]
fn part(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    logits: DevicePtr,
    cands: (DevicePtr, DevicePtr),
    rows: u32,
    n: u32,
    base: u32,
) -> Result<()> {
    KernelLaunch::new(gpu, k.part)
        .grid([rows, n.div_ceil(SPAN), 1])
        .block([THREADS, 1, 1])
        .arg_ptr(logits)
        .arg_ptr(cands.0)
        .arg_ptr(cands.1)
        .arg_u32(n)
        .arg_u32(base)
        .launch(0)
}

/// 2026-10-10: `dflash2_topk16_merge` with its layout `(parts, part_stride, row_stride)`.
#[allow(clippy::too_many_arguments)]
fn merge(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    logits: DevicePtr,
    cands: (DevicePtr, DevicePtr),
    out: (DevicePtr, DevicePtr),
    rows: u32,
    n: u32,
    base: u32,
    layout: (u32, u32, u32),
    pad: bool,
) -> Result<()> {
    ensure!(
        layout.0 * 16 <= THREADS * 8,
        "{} candidates exceed the merge",
        layout.0 * 16
    );
    KernelLaunch::new(gpu, k.merge)
        .grid([rows, 1, 1])
        .block([THREADS, 1, 1])
        .arg_ptr(logits)
        .arg_ptr(cands.0)
        .arg_ptr(cands.1)
        .arg_ptr(out.0)
        .arg_ptr(out.1)
        .arg_u32(n)
        .arg_u32(layout.0)
        .arg_u32(base)
        .arg_u32(layout.1)
        .arg_u32(layout.2)
        .arg_u32(u32::from(pad))
        .launch(0)
}

fn launch_split(gpu: &dyn GpuBackend, k: &Kernels, b: &Bufs, rows: u32, n: u32) -> Result<()> {
    let parts = n.div_ceil(SPAN);
    let cands = (b.cand_vals, b.cand_idx);
    part(gpu, k, b.logits, cands, rows, n, 0)?;
    let out = (b.vals, b.idx);
    merge(
        gpu,
        k,
        b.logits,
        cands,
        out,
        rows,
        n,
        0,
        (parts, 16, parts * 16),
        true,
    )
}

/// 2026-10-10: The vocab-sharded top-16 of a tensor-parallel drafter, emulated on one GPU:
/// the rows split into `SHARDS` 64-aligned column ranges, each copied to its own buffer, its
/// part kernel at its base and an unpadded merge into a rank-major gather buffer
/// `[shard][vals rows*16 | idx rows*16]`, then one padded merge of the shards' lists.
fn launch_sharded(gpu: &dyn GpuBackend, k: &Kernels, b: &Bufs, rows: u32, n: u32) -> Result<()> {
    let span = (n.div_ceil(64) * 64).div_ceil(SHARDS).div_ceil(64) * 64;
    let block = rows * 32;
    for s in 0..SHARDS {
        let lo = (s * span).min(n);
        let len = ((s + 1) * span).min(n) - lo;
        gpu.copy_d2d_2d_async(
            b.logits.offset(lo as usize * 2),
            n as usize * 2,
            b.shard,
            len as usize * 2,
            len as usize * 2,
            rows as usize,
            0,
        )?;
        let cands = (b.cand_vals, b.cand_idx);
        part(gpu, k, b.shard, cands, rows, len, lo)?;
        let parts = len.div_ceil(SPAN);
        let g = b.gather.offset((s * block) as usize * 4);
        let out = (g, g.offset(rows as usize * 64));
        let layout = (parts, 16, parts * 16);
        merge(
            gpu,
            k,
            DevicePtr::NULL,
            cands,
            out,
            rows,
            len,
            lo,
            layout,
            false,
        )?;
    }
    let cands = (b.gather, b.gather.offset(rows as usize * 64));
    let out = (b.vals, b.idx);
    merge(
        gpu,
        k,
        DevicePtr::NULL,
        cands,
        out,
        rows,
        n,
        0,
        (SHARDS, block, 16),
        true,
    )
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    One,
    Split,
    Sharded,
}

/// 2026-10-10: Run one kernel pair on `logits` (`rows` x `n` f32 values, BF16 on device) and
/// return (top_vals, top_idx, logits after) as bytes.
fn run(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    b: &Bufs,
    logits: &[f32],
    rows: u32,
    n: u32,
    mode: Mode,
) -> Result<[Vec<u8>; 3]> {
    gpu.copy_h2d(&bf16_bytes(logits), b.logits)?;
    gpu.copy_h2d(&vec![0xa5u8; rows as usize * 64], b.vals)?;
    gpu.copy_h2d(&vec![0xa5u8; rows as usize * 64], b.idx)?;
    match mode {
        Mode::One => launch_one(gpu, k, b, rows, n)?,
        Mode::Split => launch_split(gpu, k, b, rows, n)?,
        Mode::Sharded => launch_sharded(gpu, k, b, rows, n)?,
    }
    Ok([
        read(gpu, b.vals, rows as usize * 64)?,
        read(gpu, b.idx, rows as usize * 64)?,
        read(gpu, b.logits, (rows * n) as usize * 2)?,
    ])
}

fn case(rng: &mut Rng, kind: usize, rows: u32, n: u32) -> Vec<f32> {
    let pool: Vec<f32> = (0..24).map(|i| 12.0 + 0.0625 * i as f32).collect();
    (0..rows * n)
        .map(|i| {
            let col = i % n;
            match kind {
                0 => 30.0 * rng.unit() - 10.0,
                1 => pool[rng.next() as usize % pool.len()],
                2 => 7.5,
                3 => match rng.next() % 64 {
                    0 => f32::NAN,
                    1 => f32::INFINITY,
                    2 => f32::NEG_INFINITY,
                    3 => -1e30,
                    4 => -2e30,
                    _ => pool[rng.next() as usize % pool.len()],
                },
                // 2026-10-10: Five qualifying entries per row; the rest at or below -1e30.
                _ => {
                    if col % (n / 5) == 7 {
                        rng.unit()
                    } else if rng.next().is_multiple_of(2) {
                        -1e30
                    } else {
                        f32::NEG_INFINITY
                    }
                }
            }
        })
        .collect()
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let k = Kernels {
        one: gpu.kernel("dflash2", "dflash2_topk16")?,
        part: gpu.kernel("dflash2", "dflash2_topk16_part")?,
        merge: gpu.kernel("dflash2", "dflash2_topk16_merge")?,
    };
    let max_rows = 16usize;
    let b = Bufs {
        shard: gpu.alloc(max_rows * VOCAB as usize * 2)?,
        gather: gpu.alloc(SHARDS as usize * max_rows * 128)?,
        logits: gpu.alloc(max_rows * VOCAB as usize * 2)?,
        vals: gpu.alloc(max_rows * 64)?,
        idx: gpu.alloc(max_rows * 64)?,
        cand_vals: gpu.alloc(max_rows * VOCAB.div_ceil(SPAN) as usize * 64)?,
        cand_idx: gpu.alloc(max_rows * VOCAB.div_ceil(SPAN) as usize * 64)?,
    };
    let mut rng = Rng(0x746f_706b_3136);
    let mut checked = 0;
    for (rows, n) in [(1u32, VOCAB), (8, VOCAB), (16, VOCAB), (3, 151_937)] {
        for kind in 0..5 {
            let logits = case(&mut rng, kind, rows, n);
            let want = run(&gpu, &k, &b, &logits, rows, n, Mode::One)?;
            // 2026-10-10: The sharded top-16 writes no logits (each rank's are its own), so
            // only its lists are compared.
            for (mode, what_n) in [(Mode::Split, 3usize), (Mode::Sharded, 2)] {
                let got = run(&gpu, &k, &b, &logits, rows, n, mode)?;
                for (what, (w, g)) in ["top_vals", "top_idx", "logits"]
                    .iter()
                    .zip(want.iter().zip(got.iter()))
                    .take(what_n)
                {
                    let first = w.chunks(4).zip(g.chunks(4)).position(|(a, b)| a != b);
                    ensure!(
                        first.is_none(),
                        "rows {rows} vocab {n} case {kind} {mode:?}: {what} differs from \
                         dflash2_topk16 at word {first:?}: {:?} vs {:?}",
                        first.map(|i| &w[i * 4..i * 4 + 4]),
                        first.map(|i| &g[i * 4..i * 4 + 4])
                    );
                }
                checked += 1;
            }
        }
    }
    println!(
        "byte gate: {checked} case runs equal (two-stage: top_vals, top_idx, logits after; \
         {SHARDS}-shard: top_vals, top_idx)"
    );

    // 2026-10-10: Negative control: raise row 0's best entry by one BF16 ulp in the
    // two-stage input only; its top_vals must differ.
    let logits = case(&mut rng, 1, 8, VOCAB);
    let want = run(&gpu, &k, &b, &logits, 8, VOCAB, Mode::One)?;
    let best = (0..VOCAB as usize)
        .max_by(|&a, &c| logits[a].total_cmp(&logits[c]))
        .unwrap_or(0);
    let mut bumped = logits.clone();
    bumped[best] = bf16::from_bits(bf16::from_f32(logits[best]).to_bits() + 1).to_f32();
    for mode in [Mode::Split, Mode::Sharded] {
        let got = run(&gpu, &k, &b, &bumped, 8, VOCAB, mode)?;
        ensure!(
            want[0] != got[0],
            "negative control ({mode:?}): a one-ulp change left top_vals equal"
        );
    }
    println!("negative control: a one-ulp bump changes top_vals (comparison can fail)");

    let logits = case(&mut rng, 0, 8, VOCAB);
    gpu.copy_h2d(&bf16_bytes(&logits), b.logits)?;
    for (name, split) in [("dflash2_topk16", false), ("two-stage", true)] {
        let mut t = Vec::new();
        for _ in 0..7 {
            gpu.synchronize(0)?;
            let t0 = std::time::Instant::now();
            for _ in 0..50 {
                if split {
                    launch_split(&gpu, &k, &b, 8, VOCAB)?;
                } else {
                    launch_one(&gpu, &k, &b, 8, VOCAB)?;
                }
            }
            gpu.synchronize(0)?;
            t.push(t0.elapsed().as_secs_f64() * 1e6 / 50.0);
        }
        t.sort_by(f64::total_cmp);
        println!(
            "{name}: 8 rows x {VOCAB}: median {:.1} us (min {:.1}, max {:.1})",
            t[3], t[0], t[6]
        );
    }
    Ok(())
}
