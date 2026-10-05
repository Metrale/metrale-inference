// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Wall-time bench of the declared-W8A8 decode projection (`ops::w8a8_proj`: the
//! per-row activation quantizer plus the WxAy skinny-engine GEMV, `w8a8_gemv_rowscale_mb*`) at the
//! Qwen3.8-27B FP8 shapes, rows 1..=256, against the weight-bandwidth floor `N*K bytes / peak`.
//!
//! Each point is the mean of `ITERS` back-to-back launches on stream 0 after `WARMUP`, timed
//! between two syncs. Weights and activations are fixed bytes (speed only).
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//!   cargo run -p metrale-model-arch --release --example w8a8_gemv_bench --features cuda,gpu-examples
//!
//! Env: METRALE_PEAK_GBPS, the peak for the floor and the %-of-peak column (required: the
//! bench runs on more than one class).

use anyhow::{Context, Result};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_layers::layers::ops::{self, W8a8Kernels, W8a8Scratch, W8a8Weight};
use metrale_model_layers::weight_map::{Fp8Weight, WeightQuantFormat};
use std::time::Instant;

const WARMUP: usize = 20;
const ITERS: usize = 200;

/// 2026-10-05: `(label, segments' N, K)`: the stacked weights the 27B's W8A8 arms read.
const SHAPES: &[(&str, &[u32], u32)] = &[
    ("gdn qkv|z      ", &[10240, 6144], 5120),
    ("gdn out        ", &[5120], 6144),
    ("attn q|k|v     ", &[12288, 1024, 1024], 5120),
    ("attn o         ", &[5120], 6144),
    ("ffn56 gate     ", &[17408], 5120),
    ("ffn56 down     ", &[5120], 17408),
    ("lm_head        ", &[248320], 5120),
];

const ROWS: &[usize] = &[1, 4, 8, 16, 32, 64, 128, 256];

fn main() -> Result<()> {
    let peak: f64 = std::env::var("METRALE_PEAK_GBPS")
        .context("set METRALE_PEAK_GBPS to the device's DRAM peak in GB/s")?
        .parse()?;
    let g0 = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;
    let kernels = W8a8Kernels::load(g);
    let scratch = W8a8Scratch::alloc(g, 17408)?;
    let max_rows = *ROWS.iter().max().unwrap();
    for &(label, segs, k) in SHAPES {
        let mut parts = Vec::new();
        for &n in segs {
            let w = g.alloc(n as usize * k as usize)?;
            let s = g.alloc(n as usize * 4)?;
            // 2026-10-05: 0x30 is E4M3 0.0625; 0x3C800000 is the FP32 scale 0.015625.
            g.memset(w, 0x30, n as usize * k as usize)?;
            let ones: Vec<u8> = (0..n).flat_map(|_| 0x3C80_0000u32.to_le_bytes()).collect();
            g.copy_h2d(&ones, s)?;
            parts.push(Fp8Weight {
                weight: w,
                row_scale: s,
                n,
                k,
                scale_format: WeightQuantFormat::Fp8PerRow,
            });
        }
        let wt = W8a8Weight::new(&parts)?;
        let n = wt.n();
        let x = g.alloc(max_rows * k as usize * 2)?;
        g.memset(x, 0x3C, max_rows * k as usize * 2)?;
        let out = g.alloc(max_rows * n as usize * 2)?;
        let bytes = n as f64 * k as f64;
        let floor_us = bytes / (peak * 1e9) * 1e6;
        eprintln!(
            "── {label} N={n} K={k}  weights {:.1} MB, floor {floor_us:.1} us ──",
            bytes / 1e6
        );
        for &m in ROWS {
            for _ in 0..WARMUP {
                ops::w8a8_proj(g, &kernels, &wt, x, k, m, out, n, &scratch, 0)?;
            }
            g.synchronize(0)?;
            let t0 = Instant::now();
            for _ in 0..ITERS {
                ops::w8a8_proj(g, &kernels, &wt, x, k, m, out, n, &scratch, 0)?;
            }
            g.synchronize(0)?;
            let us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
            let gbps = bytes / (us * 1e-6) / 1e9;
            eprintln!(
                "  M={m:>3}  {us:>8.1} us  {gbps:>7.1} GB/s  {:>5.1}% of peak  {:>5.2}x floor",
                100.0 * gbps / peak,
                us / floor_us
            );
        }
        for p in parts {
            let _ = (g.free(p.weight), g.free(p.row_scale));
        }
        let _ = (g.free(x), g.free(out));
    }
    Ok(())
}
