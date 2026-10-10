// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The NVFP4 tier of `--draft-weight-quantization` at the GEMM shapes of rank 0 of
//! the GLM-5.3 DFlash2 drafter over 3 ranks: the drafter's own NVFP4 copy (`DraftQuantizer`)
//! and row GEMM (`draft_weight::nvfp4_rows`) against the BF16 GEMM, timed against the FP8 copy
//! on the FP8 row GEMV the `fp8` tiers run.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless, for every shape, the NVFP4 output at 8 rows is within
//!   `MAX_REL_L2` of the BF16 output (the 4-bit weight error only); a negative control (the
//!   NVFP4 GEMM on another weight) is far outside it; and 70 rows in one `nvfp4_rows` call
//!   (two launches) equal, byte for byte, the same rows one at a time.
//!
//! Times: median of 7 batches over rotated weight copies (>= 64 MB of BF16 per batch, so the
//! copies stream from DRAM), host-timed. The per-propose total weights each shape by its
//! launches per propose on rank 0 (five layers; the conv projection at two sites per layer).
//!
//! Run (GB10, < 3 GB of device memory):
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example dflash_nvfp4_rows_check

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::dflash_head::DraftWeight;
use metrale_model_arch::dflash_head::draft_weight::{DraftQuantizer, nvfp4_rows};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::DenseWeight;

/// 2026-10-10: E2M1 codes under a 16-wide E4M3 scale leave ~10% relative RMS error per uniform
/// weight, and a K >= 1536 dot product averages it to about that; 0.2 leaves room and stays far
/// below a wrong weight (~1.4).
const MAX_REL_L2: f64 = 0.2;

/// 2026-10-10: `(name, n, k, launches per propose)` on rank 0 of 3.
const SHAPES: [(&str, u32, u32, u32); 10] = [
    ("q_proj", 1536, 4096, 5),
    ("k_proj / v_proj", 384, 4096, 10),
    ("o_proj", 4096, 1536, 5),
    ("gate_proj / up_proj", 4096, 4096, 10),
    ("down_proj", 4096, 4096, 5),
    ("lm_head", 51648, 4096, 1),
    ("ctx fc", 4096, 20480, 1),
    ("ctx fused K/V", 3840, 4096, 1),
    ("conv kernel projection", 1024, 4096, 10),
    ("selector projection", 256, 4096, 1),
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

fn read(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    gpu.synchronize(0)?;
    let mut b = vec![0u8; n * 2];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn as_f64(b: &[u8]) -> Vec<f64> {
    b.chunks(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f64())
        .collect()
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
    let qz = |nvfp4| -> Result<DraftQuantizer<'_>> {
        Ok(DraftQuantizer {
            gpu: &gpu,
            nvfp4,
            fp8_k: gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?,
            absmax_k: gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?,
            nvfp4_k: gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?,
            stream: 0,
        })
    };
    let (q4, q8) = (qz(true)?, qz(false)?);
    let mut rng = Rng(0x6e76_6670_3472_6f77);
    const ROWS: usize = 70;
    let a = upload(&gpu, &rng.bf16_bytes(ROWS * 20480, 1.0))?;
    let out = [gpu.alloc(ROWS * 51648 * 2)?, gpu.alloc(ROWS * 51648 * 2)?];
    let mut totals = [[0.0f64; 2]; 9];
    for (name, n, k, per) in SHAPES {
        let (nu, ku) = (n as usize, k as usize);
        let amp = (3.0 / k as f32).sqrt();
        let copies = (64usize << 20).div_ceil(nu * ku * 2).clamp(2, 32);
        let bf: Vec<DenseWeight> = (0..copies)
            .map(|_| {
                Ok(DenseWeight {
                    weight: upload(&gpu, &rng.bf16_bytes(nu * ku, amp))?,
                })
            })
            .collect::<Result<_>>()?;
        let w4: Vec<DraftWeight> = bf
            .iter()
            .map(|w| q4.copy(w, nu, ku, &[], None))
            .collect::<Result<_>>()?;
        let w8: Vec<DraftWeight> = bf
            .iter()
            .map(|w| q8.copy(w, nu, ku, &[], None))
            .collect::<Result<_>>()?;
        let (DraftWeight::Nvfp4(q0), DraftWeight::Nvfp4(q1)) = (&w4[0], &w4[1]) else {
            anyhow::bail!("{name}: the NVFP4 quantizer built another format");
        };
        // 2026-10-10: Numerics at 8 rows, and the negative control on another weight.
        ops::dense_gemm_bf16_pipelined(&gpu, gemm, a, &bf[0], out[0], 8, n, k, 0)?;
        let want = as_f64(&read(&gpu, out[0], 8 * nu)?);
        nvfp4_rows(&gpu, a, q0, out[1], 8, n, k, 0)?;
        let err = rel_l2(&as_f64(&read(&gpu, out[1], 8 * nu)?), &want);
        ensure!(
            err < MAX_REL_L2,
            "{name}: NVFP4 rel L2 {err:.4} >= {MAX_REL_L2}"
        );
        nvfp4_rows(&gpu, a, q1, out[1], 8, n, k, 0)?;
        let wrong = rel_l2(&as_f64(&read(&gpu, out[1], 8 * nu)?), &want);
        ensure!(
            wrong > 0.5,
            "{name}: the negative control stayed within {wrong:.4}"
        );
        // 2026-10-10: The row chunking: 70 rows (64 + 6) in one call against one row at a time.
        nvfp4_rows(&gpu, a, q0, out[0], ROWS as u32, n, k, 0)?;
        for r in 0..ROWS {
            nvfp4_rows(
                &gpu,
                a.offset(r * ku * 2),
                q0,
                out[1].offset(r * nu * 2),
                1,
                n,
                k,
                0,
            )?;
        }
        ensure!(
            read(&gpu, out[0], ROWS * nu)? == read(&gpu, out[1], ROWS * nu)?,
            "{name}: {ROWS} rows in one call differ from the rows one at a time"
        );
        let DraftWeight::Fp8(_) = &w8[0] else {
            anyhow::bail!("{name}: the FP8 quantizer built another format");
        };
        print!("{name:<24} [{n:>5} x {k:>5}] x{per:<2} rel L2 {err:.4} (control {wrong:.2})");
        let launches = copies.max(20);
        for m in [1u32, 4, 8] {
            let t8 = median_us(&gpu, launches, |i| match &w8[i % copies] {
                DraftWeight::Fp8(f) => {
                    ops::fp8_gemv_rowscale_batch8_rt2(&gpu, rt2, a, f, out[0], m, n, k, 0)
                }
                DraftWeight::Nvfp4(_) => unreachable!("checked above"),
            })?;
            let t4 = median_us(&gpu, launches, |i| match &w4[i % copies] {
                DraftWeight::Nvfp4(q) => nvfp4_rows(&gpu, a, q, out[1], m, n, k, 0),
                DraftWeight::Fp8(_) => unreachable!("checked above"),
            })?;
            print!(" | M={m} fp8 {t8:6.1} nvfp4 {t4:6.1}");
            totals[m as usize][0] += t8 * f64::from(per);
            totals[m as usize][1] += t4 * f64::from(per);
        }
        println!();
        for w in bf {
            gpu.free(w.weight)?;
        }
        for w in w4.iter().chain(&w8) {
            let (a, b) = match w {
                DraftWeight::Fp8(f) => (f.weight, f.row_scale),
                DraftWeight::Nvfp4(q) => (q.weight, q.weight_scale),
            };
            gpu.free(a)?;
            gpu.free(b)?;
        }
    }
    for m in [1usize, 4, 8] {
        println!(
            "rank 0 per propose at M={m} (every drafter GEMM): fp8 {:.2} ms, nvfp4 {:.2} ms",
            totals[m][0] / 1e3,
            totals[m][1] / 1e3
        );
    }
    Ok(())
}
