// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: GLM-5.3's `--dense-quantization fp8` on the GPU: a projection registered in the
//! FP8 registry runs W8A8 and tracks the BF16 GEMV of the same weight within the FP8 W8A8
//! tolerance, at GLM's shapes (the KDA q projection at TP=3, the DSA absorbed o projection, a
//! shared-expert down at the 768-wide FP8 split), and a row's output is bit-identical alone and
//! inside a 300-row launch (two 256-row chunks).
//!
//! Owner: model-engine tests.
//! Invariants: none beyond the types.
//!
//! Run on a GB10 whose GPU is free, with an external timeout:
//! ```text
//! METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//! cargo test -p metrale-model-engine --test glm5next_fp8_dense_cuda --no-run
//! GLM_W4A4_GPU_ORDINAL=0 timeout 300s cargo test -p metrale-model-engine \
//! --test glm5next_fp8_dense_cuda -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use anyhow::{Context, Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_fp8_dense;
use metrale_model_layers::layers::ops::{DenseMmKernels, dense_mm_bf16};
use metrale_model_layers::layers::{DenseQuantization, set_dense_quantization_from_cli};

/// 2026-10-09: E4M3 weights and activations carry about 2^-4 relative error per element; the
/// outputs agree in direction, not in bits. Unmeasured bounds; tighten after the first run.
const MIN_COSINE: f64 = 0.995;
const MAX_REL_L2: f64 = 0.08;

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

#[test]
#[ignore = "requires an explicitly selected idle CUDA device and the glm-5.3-flash nvfp4 kernels"]
fn fp8_dense_projections_track_bf16_and_are_row_invariant() -> Result<()> {
    set_dense_quantization_from_cli(DenseQuantization::Fp8);
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
    let quantize = gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?;
    glm5next_fp8_dense::prepare(&gpu, 11264)?;
    let mut seed = 0x51ed_270b_27d1_9a3fu64;
    for (n, k, what) in [
        (2816usize, 4096usize, "kda.q_proj tp3"),
        (4096, 11264, "dsa.o_absorb tp3"),
        (4096, 768, "shared down rank 0"),
    ] {
        let w = upload(&gpu, &rand(&mut seed, n * k, 0.02))?;
        glm5next_fp8_dense::register(&gpu, quantize, w, n, k, what, stream)?;
        let x = upload(&gpu, &rand(&mut seed, 300 * k, 2.0))?;
        let (fp8_out, bf_out) = (gpu.alloc(300 * n * 2)?, gpu.alloc(300 * n * 2)?);
        for rows in [1usize, 16] {
            ensure!(glm5next_fp8_dense::proj(
                &gpu, w, x, fp8_out, rows, n, k, stream
            )?);
            dense_mm_bf16(&gpu, &bf, x, w, bf_out, rows, n, k, stream)?;
            gpu.synchronize(stream)?;
            let (cos, rel) = agreement(
                &read(&gpu, fp8_out, rows * n)?,
                &read(&gpu, bf_out, rows * n)?,
            );
            println!("{what} rows={rows}: cosine {cos:.5}, rel L2 {rel:.4}");
            ensure!(
                cos >= MIN_COSINE && rel <= MAX_REL_L2,
                "{what} rows={rows}: {cos} / {rel}"
            );
        }
        ensure!(glm5next_fp8_dense::proj(
            &gpu, w, x, fp8_out, 300, n, k, stream
        )?);
        gpu.synchronize(stream)?;
        let wide = read(&gpu, fp8_out, 300 * n)?;
        for r in [0usize, 255, 256, 299] {
            ensure!(glm5next_fp8_dense::proj(
                &gpu,
                w,
                x.offset(r * k * 2),
                fp8_out,
                1,
                n,
                k,
                stream
            )?);
            gpu.synchronize(stream)?;
            let one = read(&gpu, fp8_out, n)?;
            ensure!(
                one.iter()
                    .map(|v| v.to_bits())
                    .eq(wide[r * n..(r + 1) * n].iter().map(|v| v.to_bits())),
                "{what} row {r}: bits differ between 1 and 300 rows"
            );
        }
    }
    Ok(())
}
