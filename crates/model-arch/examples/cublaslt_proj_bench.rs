// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Wall-time bench of the cuBLASLt projection paths at the Qwen3.8-27B shapes, for
//! the wide-row (prefill, wide decode) side of the W8A8 and W4A16 projections: the FP8 GEMM with
//! per-row weight and per-token activation scales (`cublaslt::fp8_gemm_act_weight_t_rowwise`,
//! the declared W8A8 numerics), the BF16 GEMM (`cublaslt::bf16_gemm_act_weight_t`) on a weight
//! already in BF16, and the NVFP4-to-BF16 dequantization (`dequant_nvfp4_to_bf16`) a transient
//! BF16 copy would cost. Weights and activations are fixed bytes (speed only).
//!
//! Each point is the mean of `ITERS` back-to-back calls on stream 0 after `WARMUP`, timed between
//! two syncs, so a cuBLASLt point includes its per-call host work (descriptor setup and the
//! algorithm heuristic), as an eager prefill would pay it.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//!   cargo run -p metrale-model-arch --release --example cublaslt_proj_bench --features cuda,gpu-examples
//!
//! Env: METRALE_PEAK_GBPS, the peak for the floor (required: the bench runs on more than one
//! class).

use anyhow::{Context, Result};
use metrale_gpu_runtime::cublaslt;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use std::time::Instant;

const WARMUP: usize = 10;
const ITERS: usize = 50;

/// 2026-10-05: `(label, N, K)`: the 27B's FP8 projections (stacked as its W8A8 arms read them).
const FP8_SHAPES: &[(&str, u32, u32)] = &[
    ("gdn qkv|z ", 16384, 5120),
    ("gdn out   ", 5120, 6144),
    ("attn q|k|v", 14336, 5120),
    ("ffn56 gate", 17408, 5120),
    ("ffn56 down", 5120, 17408),
    ("lm_head   ", 248320, 5120),
];

/// 2026-10-05: The NVFP4 FFN projections (layers 0-55).
const NVFP4_SHAPES: &[(&str, u32, u32)] =
    &[("ffn gate/up", 17408, 5120), ("ffn down   ", 5120, 17408)];

/// 2026-10-05: Multiples of 16 (the FP8 path pads M to 16 rows).
const ROWS: &[u32] = &[16, 32, 64, 128, 256, 512, 1024, 2048];

fn time(g: &dyn GpuBackend, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..WARMUP {
        f()?;
    }
    g.synchronize(0)?;
    let t0 = Instant::now();
    for _ in 0..ITERS {
        f()?;
    }
    g.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64)
}

fn filled(g: &dyn GpuBackend, bytes: usize, v: u8) -> Result<DevicePtr> {
    let p = g.alloc(bytes)?;
    g.memset(p, v, bytes)?;
    Ok(p)
}

fn main() -> Result<()> {
    let peak: f64 = std::env::var("METRALE_PEAK_GBPS")
        .context("set METRALE_PEAK_GBPS to the device's DRAM peak in GB/s")?
        .parse()?;
    let g0 = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;
    let m_max = *ROWS.iter().max().unwrap() as usize;

    for &(label, n, k) in FP8_SHAPES {
        let (nu, ku) = (n as usize, k as usize);
        // 2026-10-05: 0x30 is E4M3 0.0625; 0x3C800000 is the FP32 scale 0.015625.
        let w = filled(g, nu * ku, 0x30)?;
        let ws = g.alloc(nu * 4)?;
        g.copy_h2d(&vec![0x3C80_0000u32.to_le_bytes(); nu].concat(), ws)?;
        let a = filled(g, m_max * ku, 0x30)?;
        let s = g.alloc(m_max * 4)?;
        g.copy_h2d(&vec![0x3C80_0000u32.to_le_bytes(); m_max].concat(), s)?;
        let out = g.alloc(m_max * nu * 2)?;
        let bytes = (nu * ku) as f64;
        eprintln!(
            "── FP8 rowwise cuBLASLt {label} N={n} K={k}, floor {:.1} us ──",
            bytes / (peak * 1e9) * 1e6
        );
        for &m in ROWS {
            let us = time(g, || {
                cublaslt::fp8_gemm_act_weight_t_rowwise(a.0, s.0, w.0, ws.0, out.0, m, n, k, 0)
            })?;
            let tflops = 2.0 * m as f64 * n as f64 * k as f64 / (us * 1e-6) / 1e12;
            eprintln!("  M={m:>4}  {us:>8.1} us  {tflops:>6.1} TFLOP/s");
        }
        for p in [w, ws, a, s, out] {
            let _ = g.free(p);
        }
    }

    let dq = g.kernel("dequant_nvfp4_bf16", "dequant_nvfp4_to_bf16").ok();
    // 2026-10-09: The per-group exact form the dense FFN's cuBLASLt arm launches.
    let dq16 = g
        .kernel("dequant_nvfp4_bf16", "dequant_nvfp4_to_bf16_g16")
        .ok();
    for &(label, n, k) in NVFP4_SHAPES {
        let (nu, ku) = (n as usize, k as usize);
        let packed = filled(g, nu * ku / 2, 0x23)?;
        let scales = filled(g, nu * ku / 16, 0x38)?;
        let w = g.alloc(nu * ku * 2)?;
        let a = filled(g, m_max * ku * 2, 0x3C)?;
        let out = g.alloc(m_max * nu * 2)?;
        eprintln!("── NVFP4 FFN {label} N={n} K={k} ──");
        match dq {
            Some(h) => {
                let us = time(g, || {
                    KernelLaunch::new(g, h)
                        .grid([n, 1, 1])
                        .block([256, 1, 1])
                        .arg_ptr(packed)
                        .arg_ptr(scales)
                        .arg_ptr(w)
                        .arg_f32(1.0)
                        .arg_u32(n)
                        .arg_u32(k)
                        .launch(0)
                })?;
                let moved = (nu * ku / 2 + nu * ku / 16 + nu * ku * 2) as f64;
                eprintln!(
                    "  dequant_nvfp4_to_bf16  {us:>8.1} us  {:>7.1} GB/s moved",
                    moved / (us * 1e-6) / 1e9
                );
            }
            None => eprintln!("  dequant_nvfp4_to_bf16 not in this target's module set"),
        }
        if let Some(h) = dq16 {
            let groups = (nu * ku / 16) as u64;
            let us = time(g, || {
                KernelLaunch::new(g, h)
                    .grid([groups.div_ceil(256).min(132 * 32) as u32, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(packed)
                    .arg_ptr(scales)
                    .arg_ptr(w)
                    .arg_u64(groups)
                    .launch(0)
            })?;
            let moved = (nu * ku / 2 + nu * ku / 16 + nu * ku * 2) as f64;
            eprintln!(
                "  dequant_nvfp4_to_bf16_g16 {us:>8.1} us  {:>7.1} GB/s moved",
                moved / (us * 1e-6) / 1e9
            );
        }
        for &m in ROWS {
            let us = time(g, || {
                cublaslt::bf16_gemm_act_weight_t(a.0, w.0, out.0, m, n, k, 0)
            })?;
            let tflops = 2.0 * m as f64 * n as f64 * k as f64 / (us * 1e-6) / 1e12;
            eprintln!("  BF16 cuBLASLt M={m:>4}  {us:>8.1} us  {tflops:>6.1} TFLOP/s");
        }
        for p in [packed, scales, w, a, out] {
            let _ = g.free(p);
        }
    }
    Ok(())
}
