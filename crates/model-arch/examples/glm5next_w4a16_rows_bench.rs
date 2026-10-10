// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The W4A16 dense tier of GLM-5.3 at the row counts of a C1 speculative verify
//! (2..=8 rows of one sequence), per rank of TP=3: the CUDA-core `w4a16_gemv_batch{M}` tiers
//! against the row-tile kernels the tier runs today (`w4a16_tc_rows`, and the segmented
//! split under `METRALE_GLM_W4A16_SEG=split`), with the one-row `w4a16_gemv` of a decode as
//! the reference cost.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless every row of a `w4a16_gemv_batch{M}` launch (M = 2..=8) equals,
//!   byte for byte, `w4a16_gemv` on that row alone (the decode's bits), for every shape.
//! - Negative control: one weight byte changed must change the output, so the comparison can
//!   fail.
//!
//! Times: median of 7 batches over rotated weight copies (>= 64 MB per batch where 512 copies
//! reach it, so the weights stream from DRAM), host-timed, so the narrowest shapes include
//! launch overhead; the per-verify total weights each shape by its
//! count in one GLM-5.3 forward (34 KDA layers x q, k, v, f_a, g_a, b, o; 42 shared experts
//! x gate, up, down).
//!
//! Run (GB10, < 1 GB of device memory):
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example glm5next_w4a16_rows_bench

use anyhow::{Result, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use metrale_model_layers::layers::ops::{self, W4a16Seg};
use metrale_model_layers::weight_map::QuantizedWeight;

/// 2026-10-10: `(n, k, count per forward)` of the rank-0 W4A16 projections of TP=3.
const SHAPES: [(u32, u32, u32); 7] = [
    (2816, 4096, 34 * 3),
    (128, 4096, 34 * 2),
    (22, 4096, 34),
    (4096, 2816, 34),
    (768, 4096, 42 * 2),
    (4096, 768, 42),
    (512, 4096, 0),
];

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

fn weight(gpu: &dyn GpuBackend, rng: &mut Rng, n: u32, k: u32) -> Result<QuantizedWeight> {
    let packed: Vec<u8> = (0..n * k / 2).map(|_| rng.next() as u8).collect();
    let scale: Vec<u8> = (0..n * k / 16)
        .map(|_| (0x08 + rng.next() % 0x70) as u8)
        .collect();
    Ok(QuantizedWeight {
        weight: upload(gpu, &packed)?,
        weight_scale: upload(gpu, &scale)?,
        weight_scale_2: 0.0123,
        ..QuantizedWeight::null()
    })
}

fn read(gpu: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    gpu.synchronize(0)?;
    let mut b = vec![0u8; bytes];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-10-10: The CUDA-core GEMV of `m` rows: `w4a16_gemv` at 1, `w4a16_gemv_batch2`/`3`
/// (no row argument) at 2 and 3, `w4a16_gemv_batch{m}` with the row argument at 4..=8.
#[allow(clippy::too_many_arguments)]
fn gemv_rows(
    gpu: &dyn GpuBackend,
    kernels: &[KernelHandle; 9],
    a: DevicePtr,
    w: &QuantizedWeight,
    c: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    let l = KernelLaunch::new(gpu, kernels[m as usize])
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w.weight)
        .arg_ptr(w.weight_scale)
        .arg_f32(w.weight_scale_2)
        .arg_ptr(c);
    if m <= 3 {
        l.arg_u32(n).arg_u32(k).launch(0)
    } else {
        l.arg_u32(m).arg_u32(n).arg_u32(k).launch(0)
    }
}

/// 2026-10-10: Median over 7 batches of the per-launch time, each batch `launches` calls of
/// `f(i)` with consecutive `i` (the caller rotates weight copies by `i`).
fn median_us(
    gpu: &dyn GpuBackend,
    launches: usize,
    mut f: impl FnMut(usize) -> Result<()>,
) -> Result<f64> {
    let mut t = Vec::new();
    for rep in 0..7 {
        gpu.synchronize(0)?;
        let t0 = std::time::Instant::now();
        for i in 0..launches {
            f(rep * launches + i)?;
        }
        gpu.synchronize(0)?;
        t.push(t0.elapsed().as_secs_f64() * 1e6 / launches as f64);
    }
    t.sort_by(f64::total_cmp);
    Ok(t[3])
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let sms = gpu.sm_count()?;
    let names = [
        "w4a16_gemv",
        "w4a16_gemv",
        "w4a16_gemv_batch2",
        "w4a16_gemv_batch3",
        "w4a16_gemv_batch4",
        "w4a16_gemv_batch5",
        "w4a16_gemv_batch6",
        "w4a16_gemv_batch7",
        "w4a16_gemv_batch8",
    ];
    let mut kernels = [KernelHandle(0); 9];
    for (i, n) in names.iter().enumerate() {
        kernels[i] = gpu.kernel("w4a16_gemv", n)?;
    }
    let mut rng = Rng(0x7734_6131_3672_6f77);
    let kmax = 4096usize;
    let a: Vec<u8> = (0..8 * kmax)
        .flat_map(|_| {
            let u = rng.next();
            let bits = ((u & 1) << 15) | ((115 + (u >> 1) % 17) << 7) | ((u >> 8) & 0x7f);
            (bits as u16).to_le_bytes()
        })
        .collect();
    let ad = upload(&gpu, &a)?;
    let out = gpu.alloc(8 * 4096 * 2)?;
    let one = gpu.alloc(4096 * 2)?;
    // 2026-10-10: Per-forward totals (us) by M: [gemv tiers, tc_rows, seg split].
    let mut totals = [[0.0f64; 3]; 9];
    for (n, k, count) in SHAPES {
        // 2026-10-10: Enough copies that one batch of launches streams >= 64 MB (GB10's L2
        // is far smaller), at most 512; a batch is at least 20 launches.
        let copies = (64usize << 20).div_ceil((n * k / 2) as usize).clamp(2, 512);
        let launches = copies.max(20);
        let ws: Vec<QuantizedWeight> = (0..copies)
            .map(|_| weight(&gpu, &mut rng, n, k))
            .collect::<Result<_>>()?;
        // 2026-10-10: Byte gate: each row of the M-row GEMV equals the one-row GEMV.
        for m in 2..=8u32 {
            gemv_rows(&gpu, &kernels, ad, &ws[0], out, m, n, k)?;
            let got = read(&gpu, out, (m * n * 2) as usize)?;
            for r in 0..m {
                gemv_rows(
                    &gpu,
                    &kernels,
                    ad.offset((r * k * 2) as usize),
                    &ws[0],
                    one,
                    1,
                    n,
                    k,
                )?;
                let want = read(&gpu, one, (n * 2) as usize)?;
                let row = &got[(r * n * 2) as usize..((r + 1) * n * 2) as usize];
                ensure!(
                    row == want.as_slice(),
                    "[{n} x {k}] M={m} row {r}: w4a16_gemv_batch{m} differs from w4a16_gemv"
                );
            }
        }
        let t1 = median_us(&gpu, launches, |i| {
            gemv_rows(&gpu, &kernels, ad, &ws[i % copies], out, 1, n, k)
        })?;
        print!("[{n:>4} x {k:>4}] x{count:<3} M=1 gemv {t1:6.1}");
        totals[1][0] += t1 * f64::from(count);
        for m in 2..=8u32 {
            let tg = median_us(&gpu, launches, |i| {
                gemv_rows(&gpu, &kernels, ad, &ws[i % copies], out, m, n, k)
            })?;
            let tt = median_us(&gpu, launches, |i| {
                ops::w4a16_tc_rows(&gpu, ad, &ws[i % copies], out, m, n, k, k, n, 0)
            })?;
            let s = ops::w4a16_tc_rows_split_for(ops::w4a16_tc_rows_seg_tiles(&[n]), k, sms);
            let ts = median_us(&gpu, launches, |i| {
                let segs = [W4a16Seg {
                    weight: ws[i % copies],
                    output: out,
                    n,
                }];
                ops::w4a16_tc_rows_seg(&gpu, ad, &segs, m, k, k, s, 0)
            })?;
            print!(" | M={m} gemv {tg:6.1} tc {tt:6.1} split{s} {ts:6.1}");
            totals[m as usize][0] += tg * f64::from(count);
            totals[m as usize][1] += tt * f64::from(count);
            totals[m as usize][2] += ts * f64::from(count);
        }
        println!();
    }
    println!(
        "per forward (W4A16 dense, rank 0): M=1 gemv {:.2} ms",
        totals[1][0] / 1e3
    );
    for (m, t) in totals.iter().enumerate().skip(2) {
        println!(
            "per forward M={m}: gemv tiers {:.2} ms, tc_rows {:.2} ms, seg split {:.2} ms",
            t[0] / 1e3,
            t[1] / 1e3,
            t[2] / 1e3
        );
    }

    // 2026-10-10: Negative control: one changed weight byte changes the 4-row output.
    let (n, k) = (768u32, 4096u32);
    let w = weight(&gpu, &mut rng, n, k)?;
    gemv_rows(&gpu, &kernels, ad, &w, out, 4, n, k)?;
    let before = read(&gpu, out, (4 * n * 2) as usize)?;
    let mut b0 = [0u8; 1];
    gpu.copy_d2h(w.weight, &mut b0)?;
    gpu.copy_h2d(&[b0[0] ^ 0x77], w.weight)?;
    gemv_rows(&gpu, &kernels, ad, &w, out, 4, n, k)?;
    let after = read(&gpu, out, (4 * n * 2) as usize)?;
    ensure!(
        before != after,
        "negative control: a changed weight byte left the output equal"
    );
    println!("byte gate: every M-row GEMV row equals the one-row GEMV; negative control fires");
    Ok(())
}
