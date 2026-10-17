// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: The FP8 dense registry of GLM-5.3 at prefill widths, per rank of TP=3: the W8A8
//! family the registry runs today (`w8a8_proj` in 256-row chunks: one activation quantize, then
//! `w8a8_gemv_rowscale_mb16` per 128 rows, each re-reading the whole weight) against the
//! dequantize + cuBLASLt path of `METRALE_GLM_FP8_DEQUANT_GEMM_MIN_ROWS`
//! (`glm5next_fp8_wide::dequant_gemm`: `dequant_fp8_rowscale_to_bf16`, then one BF16 GEMM).
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless, for every shape and width, the new path's first rows are within
//!   1% relative RMS of a host reference of the same W8A16 products (BF16 activation times the
//!   BF16-rounded E4M3 x row scale, exact sums; the GEMM differs by FP32 summation order and one
//!   BF16 output rounding: ~2^-9), and unless the negative control (row scales x1.05) breaks that
//!   bound. The W8A8 family's distance from that reference (its per-token E4M3 activations) is
//!   reported, not gated.
//!
//! Times: median of 7 host-timed batches, each over rotated weight copies (>= 64 MB of weights
//! per batch, so they stream from DRAM); per forward: each shape weighted by its count in one
//! GLM-5.3 forward on rank 0 (11 DSA layers, 34 KDA layers) under `--dense-quantization w4a16
//! --dsa-quantization w8a8`.
//!
//! Run (GB10, < 3 GB of device memory):
//!   GLM_BENCH_ROWS=2048,4096,8208 cargo run --release -p metrale-model-arch \
//!     --features cuda,gpu-examples --example glm5next_fp8_wide_bench

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_fp8_wide::dequant_gemm;
use metrale_model_layers::layers::ops::{
    W8A8_MAX_ROWS, W8a8Kernels, W8a8Scale, W8a8Scratch, W8a8Weight, w8a8_proj,
};
use metrale_model_layers::weight_map::{
    DenseWeight, Fp8Weight, WeightQuantFormat, quantize_to_fp8,
};

/// 2026-10-12: `(n, k, count per forward, name)` of the rank-0 FP8-registry projections of TP=3.
const SHAPES: [(usize, usize, usize, &str); 7] = [
    (4096, 11264, 11, "dsa.o_absorb"),
    (11264, 1536, 11, "dsa.q_absorb"),
    (1536, 4096, 11, "dsa.q_a"),
    (512, 4096, 11, "dsa.kv_a"),
    (128, 4096, 11 * 2, "dsa.indexer wk|gate"),
    (2816, 128, 34 * 2, "kda.f_b|g_b"),
    (4096, 768, 0, "(shared down, fp8 tier)"),
];

/// 2026-10-12: Rows of the host reference per shape (the exact W8A16 sums are O(R n k)).
const REF_ROWS: usize = 16;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    /// A BF16 in about [-2, 2) with a spread of exponents.
    fn bf16_bits(&mut self) -> u16 {
        let u = self.next();
        (((u & 1) << 15) | ((115 + (u >> 1) % 17) << 7) | ((u >> 8) & 0x7f)) as u16
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn download(gpu: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    gpu.synchronize(0)?;
    let mut b = vec![0u8; bytes];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn read_bf16(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    Ok(download(gpu, p, n * 2)?
        .chunks_exact(2)
        .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect())
}

/// 2026-10-12: E4M3 byte -> f32 as the kernels decode it (bias 7, subnormals, NaN code as 0).
fn e4m3(b: u8) -> f32 {
    let (sign, exp, man) = ((b >> 7) & 1, (b >> 3) & 0xF, b & 7);
    let v = if exp == 0 {
        f32::from(man) * 0.001953125
    } else if exp == 15 && man == 7 {
        0.0
    } else {
        (1.0 + f32::from(man) * 0.125) * 2f32.powi(i32::from(exp) - 7)
    };
    if sign == 1 { -v } else { v }
}

/// 2026-10-12: A random BF16 `[n, k]` weight quantized to the registry's per-row FP8 on the
/// device (the load-time quantizer), as one `W8a8Weight` and its segment.
fn weight(
    gpu: &dyn GpuBackend,
    rng: &mut Rng,
    quant: metrale_gpu_runtime::gpu::KernelHandle,
    n: usize,
    k: usize,
) -> Result<(W8a8Weight, Fp8Weight)> {
    let bf: Vec<u8> = (0..n * k)
        .flat_map(|_| rng.bf16_bits().to_le_bytes())
        .collect();
    let bf16_dev = upload(gpu, &bf)?;
    let q = quantize_to_fp8(&DenseWeight { weight: bf16_dev }, n, k, gpu, quant, 0)?;
    gpu.free(bf16_dev)?;
    let seg = Fp8Weight {
        weight: q.weight,
        row_scale: q.row_scale,
        n: n as u32,
        k: k as u32,
        scale_format: WeightQuantFormat::Fp8PerRow,
    };
    Ok((W8a8Weight::new(&[seg])?, seg))
}

/// 2026-10-12: `(relative RMS, max abs / RMS)` of `a - b` against `b`.
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

/// 2026-10-12: The exact W8A16 products of the first [`REF_ROWS`] rows: `a` BF16 times
/// `bf16(e4m3 * scale)`, summed in f64.
fn host_ref(a: &[f32], w: &[u8], scales: &[f32], rows: usize, n: usize, k: usize) -> Vec<f32> {
    let wd: Vec<f32> = w
        .iter()
        .enumerate()
        .map(|(i, &b)| bf16::from_f32(e4m3(b) * scales[i / k]).to_f32())
        .collect();
    let mut out = vec![0f32; rows * n];
    for r in 0..rows {
        let ar = &a[r * k..(r + 1) * k];
        for j in 0..n {
            let wr = &wd[j * k..(j + 1) * k];
            let s: f64 = ar
                .iter()
                .zip(wr)
                .map(|(&x, &y)| f64::from(x) * f64::from(y))
                .sum();
            out[r * n + j] = s as f32;
        }
    }
    out
}

/// 2026-10-12: The registry today: `w8a8_proj` in chunks of `W8A8_MAX_ROWS` rows.
#[allow(clippy::too_many_arguments)]
fn w8a8(
    gpu: &dyn GpuBackend,
    kernels: &W8a8Kernels,
    scratch: &W8a8Scratch,
    a: DevicePtr,
    w: &W8a8Weight,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let mut done = 0;
    while done < m {
        let rows = (m - done).min(W8A8_MAX_ROWS);
        w8a8_proj(
            gpu,
            kernels,
            w,
            a.offset(done * k * 2),
            k as u32,
            rows,
            c.offset(done * n * 2),
            n as u32,
            scratch,
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
        .unwrap_or_else(|| vec![256, 2048]);
    let mmax = rows_list.iter().copied().max().unwrap_or(2048);
    let kmax = SHAPES.iter().map(|s| s.1).max().unwrap_or(4096);
    let nmax = SHAPES.iter().map(|s| s.0).max().unwrap_or(4096);
    let kernels = W8a8Kernels::load(&gpu);
    ensure!(
        kernels.resolved(W8a8Scale::PerRow),
        "the per-row W8A8 kernels are not in this target"
    );
    let quant = gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?;
    let w8a8_scratch = W8a8Scratch::alloc(&gpu, kmax as u32)?;
    let mut rng = Rng(0x00f8_ed0f_d3a1_5678);
    let a_bits: Vec<u16> = (0..mmax * kmax).map(|_| rng.bf16_bits()).collect();
    let a_host: Vec<f32> = a_bits
        .iter()
        .map(|&b| bf16::from_bits(b).to_f32())
        .collect();
    let a_bytes: Vec<u8> = a_bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let ad = upload(&gpu, &a_bytes)?;
    let (out_old, out_new) = (gpu.alloc(mmax * nmax * 2)?, gpu.alloc(mmax * nmax * 2)?);
    let scratch_elems = SHAPES.iter().map(|s| s.0 * s.1).max().unwrap_or(0);
    let scratch = gpu.alloc(scratch_elems * 2)?;
    for &m in &rows_list {
        let (mut t_old, mut t_new) = (0.0f64, 0.0f64);
        let r = REF_ROWS.min(m);
        for (n, k, count, name) in SHAPES {
            let copies = (64usize << 20).div_ceil(n * k).clamp(2, 64);
            let ws: Vec<(W8a8Weight, Fp8Weight)> = (0..copies)
                .map(|_| weight(&gpu, &mut rng, quant, n, k))
                .collect::<Result<_>>()?;
            // 2026-10-12: The reference rows read A with stride kmax; the device A is [mmax, kmax]
            // too, so both paths take `lda = k` only when k == kmax. Pack A for this k instead.
            let a_k: Vec<u8> = (0..m)
                .flat_map(|row| a_bits[row * kmax..row * kmax + k].iter().copied())
                .flat_map(|b| b.to_le_bytes())
                .collect();
            gpu.copy_h2d(&a_k, ad)?;
            let a_ref: Vec<f32> = (0..r)
                .flat_map(|row| a_host[row * kmax..row * kmax + k].iter().copied())
                .collect();
            w8a8(
                &gpu,
                &kernels,
                &w8a8_scratch,
                ad,
                &ws[0].0,
                out_old,
                m,
                n,
                k,
            )?;
            dequant_gemm(
                &gpu,
                &ws[0].1,
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
                read_bf16(&gpu, out_old, r * n)?,
                read_bf16(&gpu, out_new, r * n)?,
            );
            let wq = download(&gpu, ws[0].1.weight, n * k)?;
            let scales: Vec<f32> = download(&gpu, ws[0].1.row_scale, n * 4)?
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let reference = host_ref(&a_ref, &wq, &scales, r, n, k);
            let (rel_new, mx_new) = rel_diff(&nw, &reference);
            let (rel_old, _) = rel_diff(&o, &reference);
            ensure!(
                rel_new <= 0.01,
                "[{n} x {k}] M={m}: the dequant GEMM is {rel_new:.2e} (relative RMS) from the \
                 W8A16 reference"
            );
            ensure!(
                reference.iter().any(|&x| x != 0.0) && o.iter().any(|&x| x != 0.0),
                "[{n} x {k}] M={m}: zero output"
            );
            let bumped: Vec<u8> = scales
                .iter()
                .flat_map(|s| (s * 1.05).to_le_bytes())
                .collect();
            let bumped_dev = upload(&gpu, &bumped)?;
            let bad = Fp8Weight {
                row_scale: bumped_dev,
                ..ws[0].1
            };
            dequant_gemm(&gpu, &bad, scratch, scratch_elems, ad, out_new, m, n, k, 0)?;
            let control = rel_diff(&read_bf16(&gpu, out_new, r * n)?, &reference).0;
            gpu.free(bumped_dev)?;
            ensure!(
                control > 0.01,
                "[{n} x {k}] M={m}: negative control (scales x1.05) stayed within the bound"
            );
            let launches = copies.max(8);
            let to = median_us(&gpu, launches, |i| {
                w8a8(
                    &gpu,
                    &kernels,
                    &w8a8_scratch,
                    ad,
                    &ws[i % copies].0,
                    out_old,
                    m,
                    n,
                    k,
                )
            })?;
            let tn = median_us(&gpu, launches, |i| {
                dequant_gemm(
                    &gpu,
                    &ws[i % copies].1,
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
            let w8a8_weight_gbs = (m.div_ceil(128) * n * k) as f64 / to / 1e3;
            println!(
                "M={m:5} [{n:>5} x {k:>5}] x{count:<3} {name:<24} W8A8 {to:9.1} us ({:5.1} TFLOPS, weight stream {w8a8_weight_gbs:5.0} GB/s)  dequant+cuBLASLt {tn:9.1} us ({:5.1} TFLOPS)  vs W8A16 ref: new {rel_new:.1e} (max/RMS {mx_new:.1e}), W8A8 {rel_old:.1e}, control {control:.1e}",
                tf / to / 1e6,
                tf / tn / 1e6
            );
            t_old += to * count as f64;
            t_new += tn * count as f64;
            for (_, seg) in ws {
                gpu.free(seg.weight)?;
                gpu.free(seg.row_scale)?;
            }
        }
        println!(
            "M={m:5}: FP8-registry dense per forward {:.2} ms -> {:.2} ms; per 1k tokens {:.1} ms -> {:.1} ms",
            t_old / 1e3,
            t_new / 1e3,
            t_old / 1e3 * 1000.0 / m as f64,
            t_new / 1e3 * 1000.0 / m as f64
        );
    }
    Ok(())
}
