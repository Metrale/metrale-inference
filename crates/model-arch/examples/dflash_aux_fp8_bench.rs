// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The drafter GEMMs `METRALE_DFLASH_AUX_FP8=1` moves to FP8, at the GLM-5.3
//! DFlash2 drafter's shapes: the BF16 prefill GEMM they ran on (`dense_gemm_bf16_pipelined`)
//! against the FP8 row GEMV they run on (`fp8_gemv_rowscale_batch8_rt2`, 1..=8 rows).
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless every FP8 output is within `MAX_REL_L2` of the BF16 output (the
//!   weight quantization error only), and a negative control (the FP8 GEMV on a different
//!   weight) is far outside it.
//!
//! Times: median of 7 batches over rotated weight copies (>= 64 MB per batch), host-timed;
//! the per-propose total weights the conv projection by its ten launches per propose (five
//! layers x two sites) and the rest by one.
//!
//! Run (GB10, < 2 GB of device memory):
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example dflash_aux_fp8_bench

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::{DenseWeight, Fp8DenseWeight};

/// 2026-10-10: FP8 E4M3 with one scale per row moves a K-4096 dot product by ~3% relative L2
/// on uniform weights (3 mantissa bits); 0.08 leaves room and stays far below a wrong weight.
const MAX_REL_L2: f64 = 0.08;

/// 2026-10-10: `(name, n, k, launches per propose)`.
const SHAPES: [(&str, u32, u32, u32); 4] = [
    ("conv kernel projection", 1024, 4096, 10),
    ("selector projection", 256, 4096, 1),
    ("ctx fc", 4096, 20480, 1),
    ("ctx fused K/V", 10240, 4096, 1),
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
    fn bf16_bytes(&mut self, n: usize, amp: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|_| {
                let x = (self.next() as f32 / u32::MAX as f32 - 0.5) * 2.0 * amp;
                bf16::from_f32(x).to_bits().to_le_bytes()
            })
            .collect()
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

fn read_f64(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f64>> {
    gpu.synchronize(0)?;
    let mut b = vec![0u8; n * 2];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b.chunks(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f64())
        .collect())
}

fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum();
    (num / den.max(f64::MIN_POSITIVE)).sqrt()
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
    let gemm = gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?;
    let rt2 = gpu.kernel("fp8_gemv_rt", "fp8_gemv_rowscale_batch8_rt2")?;
    let quant = gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?;
    let mut rng = Rng(0x6175_7866_7038);
    let a = upload(&gpu, &rng.bf16_bytes(8 * 20480, 1.0))?;
    let (out_b, out_f) = (gpu.alloc(8 * 10240 * 2)?, gpu.alloc(8 * 10240 * 2)?);
    let mut totals = [[0.0f64; 2]; 9];
    for (name, n, k, per) in SHAPES {
        let bytes = (n * k) as usize * 2;
        let copies = (64usize << 20).div_ceil(bytes).clamp(2, 64);
        let launches = copies.max(20);
        let bf: Vec<DenseWeight> = (0..copies)
            .map(|_| {
                Ok(DenseWeight {
                    weight: upload(&gpu, &rng.bf16_bytes((n * k) as usize, 0.05))?,
                })
            })
            .collect::<Result<_>>()?;
        let f8: Vec<Fp8DenseWeight> = bf
            .iter()
            .map(|w| w.quantize_to_fp8(&gpu, quant, n as usize, k as usize, 0))
            .collect::<Result<_>>()?;
        // 2026-10-10: Numerics gate at 8 rows, and the negative control on another weight.
        ops::dense_gemm_bf16_pipelined(&gpu, gemm, a, &bf[0], out_b, 8, n, k, 0)?;
        ops::fp8_gemv_rowscale_batch8_rt2(&gpu, rt2, a, &f8[0], out_f, 8, n, k, 0)?;
        let want = read_f64(&gpu, out_b, (8 * n) as usize)?;
        let err = rel_l2(&read_f64(&gpu, out_f, (8 * n) as usize)?, &want);
        ensure!(
            err < MAX_REL_L2,
            "{name}: FP8 rel L2 {err:.4} >= {MAX_REL_L2}"
        );
        ops::fp8_gemv_rowscale_batch8_rt2(&gpu, rt2, a, &f8[1], out_f, 8, n, k, 0)?;
        let wrong = rel_l2(&read_f64(&gpu, out_f, (8 * n) as usize)?, &want);
        ensure!(
            wrong > 0.5,
            "{name}: the negative control stayed within {wrong:.4}"
        );
        print!("{name:<24} [{n:>5} x {k:>5}] x{per:<2} rel L2 {err:.4} (control {wrong:.2})");
        for m in [1u32, 2, 3, 4, 8] {
            let tb = median_us(&gpu, launches, |i| {
                ops::dense_gemm_bf16_pipelined(&gpu, gemm, a, &bf[i % copies], out_b, m, n, k, 0)
            })?;
            let tf = median_us(&gpu, launches, |i| {
                ops::fp8_gemv_rowscale_batch8_rt2(&gpu, rt2, a, &f8[i % copies], out_f, m, n, k, 0)
            })?;
            print!(" | M={m} bf16 {tb:7.1} fp8 {tf:6.1}");
            totals[m as usize][0] += tb * f64::from(per);
            totals[m as usize][1] += tf * f64::from(per);
        }
        println!();
    }
    for m in [1usize, 2, 3, 4, 8] {
        println!(
            "per propose at M={m} (all four sites): bf16 {:.2} ms, fp8 {:.2} ms",
            totals[m][0] / 1e3,
            totals[m][1] / 1e3
        );
    }
    Ok(())
}
