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
//! bench runs on more than one class). With `METRALE_W8A8_BENCH_ENTRIES` (comma-separated
//! `w8a8_gemv` entry names), each named entry is also launched directly at every row count it
//! serves (8 rows per token tile: an `mbN` entry serves up to 8N rows), on pre-quantized
//! activations, so the time is the GEMV alone: the way to compare schedule points (KU, blocks
//! per SM) of the engine on a class before one is selected.

use anyhow::{Context, Result};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
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
    let entries: Vec<String> = std::env::var("METRALE_W8A8_BENCH_ENTRIES")
        .map(|v| {
            v.split(',')
                .filter(|e| !e.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
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
        for entry in &entries {
            entry_rows(g, entry, &parts, k, n, x, out, peak, floor_us, bytes)?;
        }
        for p in parts {
            let _ = (g.free(p.weight), g.free(p.row_scale));
        }
        let _ = (g.free(x), g.free(out));
    }
    Ok(())
}

/// 2026-10-05: Time one `w8a8_gemv` entry launched directly (the engine's geometry: grid
/// ceil(N / 16), block 256) at every row count of `ROWS` it serves. The activation bytes in `x`
/// are read as E4M3 with unit per-token scales: speed only.
#[allow(clippy::too_many_arguments)]
fn entry_rows(
    g: &dyn GpuBackend,
    entry: &str,
    parts: &[Fp8Weight],
    k: u32,
    n: u32,
    x: DevicePtr,
    out: DevicePtr,
    peak: f64,
    floor_us: f64,
    bytes: f64,
) -> Result<()> {
    let h = g.kernel("w8a8_gemv", entry)?;
    let mb: usize = entry
        .split('_')
        .find_map(|t| t.strip_prefix("mb").and_then(|v| v.parse().ok()))
        .context("entry name carries no mbN")?;
    let max_rows = *ROWS.iter().max().unwrap();
    let a_scale = g.alloc(max_rows * 4)?;
    let ones: Vec<u8> = (0..max_rows).flat_map(|_| 1.0f32.to_le_bytes()).collect();
    g.copy_h2d(&ones, a_scale)?;
    let seg = |i: usize| parts.get(i).or(parts.last()).copied().unwrap();
    let (n1, n2) = match parts.len() {
        1 => (n, n),
        2 => (parts[0].n, n),
        _ => (parts[0].n, parts[0].n + parts[1].n),
    };
    for &m in ROWS.iter().filter(|&&m| m <= 8 * mb) {
        let launch = || {
            KernelLaunch::new(g, h)
                .grid([n.div_ceil(16), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(x)
                .arg_ptr(a_scale)
                .arg_ptr(seg(0).weight)
                .arg_ptr(seg(0).row_scale)
                .arg_ptr(seg(1).weight)
                .arg_ptr(seg(1).row_scale)
                .arg_ptr(seg(2).weight)
                .arg_ptr(seg(2).row_scale)
                .arg_ptr(out)
                .arg_u32(m as u32)
                .arg_u32(n)
                .arg_u32(k)
                .arg_u32(k)
                .arg_u32(n)
                .arg_u32(n1)
                .arg_u32(n2)
                .launch(0)
        };
        for _ in 0..WARMUP {
            launch()?;
        }
        g.synchronize(0)?;
        let t0 = Instant::now();
        for _ in 0..ITERS {
            launch()?;
        }
        g.synchronize(0)?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
        let gbps = bytes / (us * 1e-6) / 1e9;
        eprintln!(
            "    {entry:<32} M={m:>3}  {us:>8.1} us  {gbps:>7.1} GB/s  {:>5.1}% of peak  {:>5.2}x floor",
            100.0 * gbps / peak,
            us / floor_us
        );
    }
    let _ = g.free(a_scale);
    Ok(())
}
