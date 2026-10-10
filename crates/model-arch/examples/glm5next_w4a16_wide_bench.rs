// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The W4A16 dense tier of GLM-5.3 at prefill widths, per rank of TP=3: the row
//! tiles the tier runs today (`w4a16_tc_rows`, one 64-row launch per 64 rows) against the
//! dequantize + cuBLASLt path of `METRALE_GLM_W4A16_DEQUANT_GEMM_MIN_ROWS`
//! (`glm5next_w4a16_wide::dequant_gemm`: `dequant_nvfp4_to_bf16_g16`, then one BF16 GEMM with the
//! tensor scale as alpha).
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless, for every shape and width, the new path's outputs are within 1%
//!   relative RMS of the row tiles' (the two compute the same products and differ in FP32
//!   summation order only: ~2^-9, one BF16 rounding), and unless the negative control (the
//!   tensor scale off by 5%) breaks that bound.
//!
//! Times: median of 7 host-timed batches, each over rotated weight copies (>= 64 MB of weights
//! per batch, so they stream from DRAM); per 1k tokens: each shape weighted by its count in one
//! GLM-5.3 forward (34 KDA layers x q, k, v, f_a, g_a, b, o; 42 shared experts x gate, up, down).
//!
//! Run (GB10, < 3 GB of device memory):
//!   GLM_BENCH_ROWS=512,2048,4096 cargo run --release -p metrale-model-arch \
//!     --features cuda,gpu-examples --example glm5next_w4a16_wide_bench

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_w4a16_wide::dequant_gemm;
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::QuantizedWeight;

/// 2026-10-10: `(n, k, count per forward)` of the rank-0 W4A16 projections of TP=3.
const SHAPES: [(usize, usize, usize); 6] = [
    (2816, 4096, 34 * 3),
    (128, 4096, 34 * 2),
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

fn weight(gpu: &dyn GpuBackend, rng: &mut Rng, n: usize, k: usize) -> Result<QuantizedWeight> {
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

fn read_bf16(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    gpu.synchronize(0)?;
    let mut b = vec![0u8; n * 2];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect())
}

/// 2026-10-10: `(relative RMS, max abs / RMS)` of `a - b` against `b`: a summation-order change
/// moves a BF16 output by about one rounding (relative RMS ~2^-9), a scale error moves all.
fn rel_diff(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut num, mut den, mut mx) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let d = f64::from(x - y);
        num += d * d;
        den += f64::from(y) * f64::from(y);
        mx = mx.max(d.abs());
    }
    let rms = (den / b.len().max(1) as f64).sqrt().max(1e-30);
    ((num / den.max(1e-30)).sqrt(), mx / rms)
}

/// 2026-10-10: The tier today: `w4a16_tc_rows` in launches of 64 rows.
#[allow(clippy::too_many_arguments)]
fn tiles(
    gpu: &dyn GpuBackend,
    a: DevicePtr,
    w: &QuantizedWeight,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let mut done = 0;
    while done < m {
        let rows = (m - done).min(64);
        ops::w4a16_tc_rows(
            gpu,
            a.offset(done * k * 2),
            w,
            c.offset(done * n * 2),
            rows as u32,
            n as u32,
            k as u32,
            k as u32,
            n as u32,
            0,
        )?;
        done += rows;
    }
    Ok(())
}

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
    let rows_list: Vec<usize> = std::env::var("GLM_BENCH_ROWS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![512, 2048]);
    let mmax = rows_list.iter().copied().max().unwrap_or(2048);
    let mut rng = Rng(0x005e_ed0f_d3a1_1234);
    let kmax = 4096usize;
    let a: Vec<u8> = (0..mmax * kmax)
        .flat_map(|_| {
            let u = rng.next();
            let bits = ((u & 1) << 15) | ((115 + (u >> 1) % 17) << 7) | ((u >> 8) & 0x7f);
            (bits as u16).to_le_bytes()
        })
        .collect();
    let ad = upload(&gpu, &a)?;
    let (out_old, out_new) = (gpu.alloc(mmax * 4096 * 2)?, gpu.alloc(mmax * 4096 * 2)?);
    let scratch_elems = 4096 * 4096;
    let scratch = gpu.alloc(scratch_elems * 2)?;
    for &m in &rows_list {
        let (mut t_old, mut t_new) = (0.0f64, 0.0f64);
        for (n, k, count) in SHAPES {
            let copies = (64usize << 20).div_ceil(n * k / 2).clamp(2, 64);
            let ws: Vec<QuantizedWeight> = (0..copies)
                .map(|_| weight(&gpu, &mut rng, n, k))
                .collect::<Result<_>>()?;
            tiles(&gpu, ad, &ws[0], out_old, m, n, k)?;
            dequant_gemm(
                &gpu,
                &ws[0],
                scratch,
                scratch_elems,
                ad,
                out_new,
                m,
                n,
                k,
                0,
            )?;
            let (o, nw) = (
                read_bf16(&gpu, out_old, m * n)?,
                read_bf16(&gpu, out_new, m * n)?,
            );
            let (rel, mx) = rel_diff(&nw, &o);
            ensure!(
                rel <= 0.01,
                "[{n} x {k}] M={m}: the dequant GEMM is {rel:.2e} (relative RMS) from the row tiles"
            );
            ensure!(
                o.iter().any(|&x| x != 0.0),
                "[{n} x {k}] M={m}: zero output"
            );
            let bad = QuantizedWeight {
                weight_scale_2: ws[0].weight_scale_2 * 1.05,
                ..ws[0]
            };
            dequant_gemm(&gpu, &bad, scratch, scratch_elems, ad, out_new, m, n, k, 0)?;
            let bad = rel_diff(&read_bf16(&gpu, out_new, m * n)?, &o).0;
            ensure!(
                bad > 0.01,
                "[{n} x {k}] M={m}: negative control (scale x1.05) stayed within the bound"
            );
            let launches = copies.max(8);
            let to = median_us(&gpu, launches, |i| {
                tiles(&gpu, ad, &ws[i % copies], out_old, m, n, k)
            })?;
            let tn = median_us(&gpu, launches, |i| {
                dequant_gemm(
                    &gpu,
                    &ws[i % copies],
                    scratch,
                    scratch_elems,
                    ad,
                    out_new,
                    m,
                    n,
                    k,
                    0,
                )
            })?;
            let tf = 2.0 * (m * n * k) as f64;
            println!(
                "M={m:5} [{n:>4} x {k:>4}] x{count:<3} tiles {to:8.1} us ({:5.1} TFLOPS)  dequant+cuBLASLt {tn:8.1} us ({:5.1} TFLOPS)  diff rel RMS {rel:.1e} max/RMS {mx:.1e} (control {bad:.1e})",
                tf / to / 1e6,
                tf / tn / 1e6
            );
            t_old += to * count as f64;
            t_new += tn * count as f64;
            for w in ws {
                gpu.free(w.weight)?;
                gpu.free(w.weight_scale)?;
            }
        }
        println!(
            "M={m:5}: W4A16 dense per forward {:.2} ms -> {:.2} ms; per 1k tokens {:.1} ms -> {:.1} ms",
            t_old / 1e3,
            t_new / 1e3,
            t_old / 1e3 * 1000.0 / m as f64,
            t_new / 1e3 * 1000.0 / m as f64
        );
    }
    Ok(())
}
