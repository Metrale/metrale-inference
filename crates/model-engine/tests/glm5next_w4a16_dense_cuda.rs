// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: GLM-5.3's `--dense-quantization w4a16` on the GPU, at GLM's TP=3 shapes under the
//! tier (a KDA q projection at 22 heads, a KDA o projection at 20 heads, the shared expert's gate
//! and its down at the 768-wide split):
//! - the W4A16 output matches the host product of the same activations with the dequantized
//!   NVFP4 weight (read back from the device), which pins the quantizer's and the kernel's layout
//!   and leaves only FP32 summation order and the BF16 output rounding;
//! - it tracks the BF16 GEMV of the unquantized weight within the NVFP4 tolerance;
//! - a row's bits are the same alone and inside a 150-row launch (64 + 64 + 22).
//!
//! Owner: model-engine tests.
//! Invariants: none beyond the types.
//!
//! Run on a GB10 whose GPU is free, with an external timeout:
//! ```text
//! METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//! cargo test -p metrale-model-engine --test glm5next_w4a16_dense_cuda --no-run
//! GLM_W4A4_GPU_ORDINAL=0 timeout 300s cargo test -p metrale-model-engine \
//! --test glm5next_w4a16_dense_cuda -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use anyhow::{Context, Result, ensure};
use half::bf16;
use metrale_cache::kv_dequant::{NVFP4_E2M1_LUT, e4m3_lut};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_w4a16_dense::{self as w4a16, Nvfp4QuantKernels};
use metrale_model_layers::layers::ops::{DenseMmKernels, dense_mm_bf16};
use metrale_model_layers::layers::{DenseQuantization, set_dense_quantization_from_cli};
use metrale_model_layers::weight_map::{DenseWeight, quantize_to_nvfp4};

/// 2026-10-09: Against the BF16 weight: NVFP4 weights carry the format's rounding error. A host
/// simulation of this recipe on these distributions gives cosine 0.994-0.996 and relative L2
/// 0.09-0.11 (uniform and Gaussian weights, K 768 and 4096); the bounds leave about 1.5x.
const MIN_COSINE_BF16: f64 = 0.985;
const MAX_REL_L2_BF16: f64 = 0.17;
/// 2026-10-09: Against the host product with the dequantized weight: FP32 summation order and the
/// BF16 output rounding (2^-9 relative) only.
const MAX_REL_L2_DEQUANT: f64 = 0.01;

fn upload(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = v
        .iter()
        .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
        .collect();
    let p = gpu.alloc(b.len())?;
    gpu.copy_h2d(&b, p)?;
    Ok(p)
}

fn read(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut raw = vec![0u8; n * 2];
    gpu.copy_d2h(p, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect())
}

fn rand(seed: &mut u64, n: usize, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            ((*seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * amp
        })
        .collect()
}

fn agreement(got: &[f32], want: &[f32]) -> (f64, f64) {
    let (mut dot, mut gg, mut ww, mut dd) = (0f64, 0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let (g, w) = (g as f64, w as f64);
        dot += g * w;
        gg += g * g;
        ww += w * w;
        dd += (g - w) * (g - w);
    }
    (
        dot / (gg.sqrt() * ww.sqrt()).max(1e-30),
        (dd / ww.max(1e-30)).sqrt(),
    )
}

/// 2026-10-09: The `[n, k]` weight `quantize_to_nvfp4` wrote, dequantized on the host: packed
/// E2M1 (even element in the low nibble) times its E4M3 group scale times the tensor scale.
fn dequant(
    gpu: &dyn GpuBackend,
    packed: DevicePtr,
    scale: DevicePtr,
    s2: f32,
    n: usize,
    k: usize,
) -> Result<Vec<f32>> {
    let (mut p, mut s) = (vec![0u8; n * k / 2], vec![0u8; n * k / 16]);
    gpu.copy_d2h(packed, &mut p)?;
    gpu.copy_d2h(scale, &mut s)?;
    Ok((0..n * k)
        .map(|i| {
            let code = (p[i / 2] >> (4 * (i % 2))) & 0xF;
            NVFP4_E2M1_LUT[code as usize] * e4m3_lut()[s[i / 16] as usize] * s2
        })
        .collect())
}

#[test]
#[ignore = "requires an explicitly selected idle CUDA device and the glm-5.3-flash nvfp4 kernels"]
fn w4a16_dense_projections_match_their_nvfp4_weights_and_are_row_invariant() -> Result<()> {
    set_dense_quantization_from_cli(DenseQuantization::W4a16);
    let ordinal = std::env::var("GLM_W4A4_GPU_ORDINAL")?.parse()?;
    let target = metrale_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4")
        .context("glm-5.3-flash nvfp4 target")?;
    let gpu = MetraleCudaBackend::new(ordinal, &target.modules)?;
    let stream = gpu.create_stream()?;
    let bf = DenseMmKernels {
        gemm: gpu.kernel("gemm", "dense_gemm_bf16")?,
        gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
        batchm: gpu.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")?,
    };
    let kernels = Nvfp4QuantKernels::load(&gpu)?;
    const ROWS: usize = 150;
    let mut seed = 0x51ed_270b_27d1_9a3fu64;
    for (n, k, what) in [
        (2816usize, 4096usize, "kda.q_proj 22 heads"),
        (4096, 2560, "kda.o_proj 20 heads"),
        (768, 4096, "shared gate rank 0"),
        (4096, 768, "shared down rank 0"),
    ] {
        let w = upload(&gpu, &rand(&mut seed, n * k, 0.02))?;
        let key = w4a16::register(&gpu, &kernels, w, n, k, what, stream)?;
        // 2026-10-09: The same kernels on the same bytes write the same NVFP4 weight again.
        let again = quantize_to_nvfp4(
            &DenseWeight { weight: w },
            n,
            k,
            &gpu,
            kernels.absmax,
            kernels.quantize,
            stream,
        )?;
        let wq = dequant(
            &gpu,
            again.weight,
            again.weight_scale,
            again.weight_scale_2,
            n,
            k,
        )?;
        let xs = rand(&mut seed, ROWS * k, 2.0);
        let x = upload(&gpu, &xs)?;
        let (w4_out, bf_out) = (gpu.alloc(ROWS * n * 2)?, gpu.alloc(ROWS * n * 2)?);
        for rows in [1usize, 16, 64] {
            ensure!(w4a16::proj(&gpu, key, x, w4_out, rows, n, k, stream)?);
            dense_mm_bf16(&gpu, &bf, x, w, bf_out, rows, n, k, stream)?;
            gpu.synchronize(stream)?;
            let got = read(&gpu, w4_out, rows * n)?;
            // 2026-10-09: The host product covers the first 16 rows; it is the slow part.
            let checked = rows.min(16) * n;
            let host: Vec<f32> = (0..checked)
                .map(|i| {
                    let (r, c) = (i / n, i % n);
                    (0..k)
                        .map(|j| {
                            bf16::from_f32(xs[r * k + j]).to_f32() as f64 * wq[c * k + j] as f64
                        })
                        .sum::<f64>() as f32
                })
                .collect();
            let (_, rel_dq) = agreement(&got[..checked], &host);
            let (cos, rel) = agreement(&got, &read(&gpu, bf_out, rows * n)?);
            println!(
                "{what} rows={rows}: vs dequant rel L2 {rel_dq:.5}; vs BF16 cosine {cos:.5}, \
                 rel L2 {rel:.4}"
            );
            ensure!(
                rel_dq <= MAX_REL_L2_DEQUANT,
                "{what} rows={rows}: {rel_dq} vs the dequantized weight"
            );
            ensure!(
                cos >= MIN_COSINE_BF16 && rel <= MAX_REL_L2_BF16,
                "{what} rows={rows}: {cos} / {rel} vs BF16"
            );
        }
        ensure!(w4a16::proj(&gpu, key, x, w4_out, ROWS, n, k, stream)?);
        gpu.synchronize(stream)?;
        let wide = read(&gpu, w4_out, ROWS * n)?;
        for r in [0usize, 63, 64, 127, 128, 149] {
            ensure!(w4a16::proj(
                &gpu,
                key,
                x.offset(r * k * 2),
                w4_out,
                1,
                n,
                k,
                stream
            )?);
            gpu.synchronize(stream)?;
            let one = read(&gpu, w4_out, n)?;
            ensure!(
                one.iter()
                    .map(|v| v.to_bits())
                    .eq(wide[r * n..(r + 1) * n].iter().map(|v| v.to_bits())),
                "{what} row {r}: bits differ between 1 and {ROWS} rows"
            );
        }
    }
    Ok(())
}
