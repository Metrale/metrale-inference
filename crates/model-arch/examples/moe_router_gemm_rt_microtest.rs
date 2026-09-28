// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `moe_router_gemm_rt` against `dense_gemm_bf16_router` (the MoE router gate
//! GEMM), both through their `ops` launchers, at the Qwen3.6-35B-A3B router shape (256
//! experts, hidden 2048) and at tail shapes.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless every case's output is byte-identical over all M x N logits
//!   (both outputs are first filled with a sentinel).
//!
//! Inputs are BF16 with a wide dynamic range (router inputs are normed hidden states, the
//! gate weights are small), so an accumulation-order difference would show. Mean times over
//! 10 launches are printed.
//!
//! Run (GB10): cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!   --example moe_router_gemm_rt_microtest

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::DenseWeight;
use std::time::Instant;

const SENTINEL: u8 = 0x5a;
/// 2026-09-28: (M, N, K).
const CASES: [(usize, usize, usize); 5] = [
    (8200, 256, 2048),
    (32776, 256, 2048),
    (1024, 256, 2048),
    (4099, 200, 2048),
    (1500, 384, 1040),
];

fn bf16_bytes(state: &mut u64, n: usize, scale: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|_| {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (*state >> 40) as f32 / (1u64 << 24) as f32;
            let e = ((*state >> 20) & 7) as i32 - 3;
            bf16::from_f32((2.0 * u - 1.0) * scale * 2f32.powi(e))
                .to_bits()
                .to_le_bytes()
        })
        .collect()
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let old = gpu.kernel("gemm", "dense_gemm_bf16_router")?;
    let new = gpu.kernel("moe_router_gemm_prefill", "moe_router_gemm_rt")?;
    let mut state = 0x726f_7574_2026_0928u64;
    for (m, n, k) in CASES {
        let a = upload(&gpu, &bf16_bytes(&mut state, m * k, 4.0))?;
        let w = DenseWeight {
            weight: upload(&gpu, &bf16_bytes(&mut state, n * k, 0.05))?,
        };
        let bytes = m * n * 2;
        let outs = [gpu.alloc(bytes)?, gpu.alloc(bytes)?];
        for o in outs {
            gpu.memset(o, SENTINEL, bytes)?;
        }
        let (m32, n32, k32) = (m as u32, n as u32, k as u32);
        let run_old = || ops::dense_gemm_router(&gpu, old, a, &w, outs[0], m32, n32, k32, 0);
        let run_new = || ops::moe_router_gemm_rt(&gpu, new, a, &w, outs[1], m32, n32, k32, 0);
        run_old()?;
        run_new()?;
        gpu.synchronize(0)?;
        let mut h = [vec![0u8; bytes], vec![0u8; bytes]];
        gpu.copy_d2h(outs[0], &mut h[0])?;
        gpu.copy_d2h(outs[1], &mut h[1])?;
        let diff = h[0]
            .chunks_exact(2)
            .zip(h[1].chunks_exact(2))
            .filter(|(x, y)| x != y)
            .count();
        ensure!(
            diff == 0,
            "M={m} N={n} K={k}: {diff} of {} logits differ",
            m * n
        );
        let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
            gpu.synchronize(0)?;
            let t = Instant::now();
            for _ in 0..10 {
                f()?;
            }
            gpu.synchronize(0)?;
            Ok(t.elapsed().as_secs_f64() * 1e2)
        };
        let (t0, t1) = (time(&run_old)?, time(&run_new)?);
        println!(
            "M={m:5} N={n} K={k}: bit-identical; router {t0:.3} ms, rt {t1:.3} ms ({:.2}x)",
            t0 / t1
        );
        for p in [a, w.weight, outs[0], outs[1]] {
            gpu.free(p)?;
        }
    }
    println!("PASS: moe_router_gemm_rt is bit-identical to dense_gemm_bf16_router");
    Ok(())
}
