// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Wall-time bench of the NVFP4 W4A16 tensor-core GEMV row tiers
//! (`w4a16_gemv_tc8`, `w4a16_gemv_tc16` and their schedule points in
//! `kernels/gb10/common/w4a16_gemv_tc.cu`) at the Qwen3.8-27B FFN shapes, against the
//! weight-bandwidth floor `(N*K/2 + N*K/16) bytes / peak`.
//!
//! Each point is the mean of `ITERS` back-to-back launches on stream 0 after `WARMUP`, timed
//! between two syncs. Weights and activations are fixed bytes (speed only); the points of a tier
//! give the same bits by construction (see the kernel file).
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//!   cargo run -p metrale-model-arch --release --example w4a16_gemv_tc_bench --features cuda,gpu-examples
//!
//! Env: METRALE_PEAK_GBPS (required: the bench runs on more than one class). Every point of
//! `metrale_kernels::w4a16_gemv_tc_entries::W4A16_GEMV_TC_POINTS` that resolves is timed at the
//! rows its tier serves.

use anyhow::{Context, Result};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use std::time::Instant;

const WARMUP: usize = 20;
const ITERS: usize = 200;

/// 2026-10-09: `(label, N, K)`: the dense FFN's NVFP4 projections (layers 0-55).
const SHAPES: &[(&str, u32, u32)] = &[
    ("ffn gate    ", 17408, 5120),
    ("ffn gate|up ", 34816, 5120),
    ("ffn down    ", 5120, 17408),
];

/// 2026-10-09: `(rows served, NT)` of a point of `metrale_kernels::w4a16_gemv_tc_entries`.
fn geometry(point: &str) -> (u32, u32) {
    let mt = if point.starts_with("tc16") { 16 } else { 8 };
    (
        mt,
        metrale_kernels::w4a16_gemv_tc_entries::w4a16_gemv_tc_nt(point),
    )
}

fn main() -> Result<()> {
    let peak: f64 = std::env::var("METRALE_PEAK_GBPS")
        .context("set METRALE_PEAK_GBPS to the device's DRAM peak in GB/s")?
        .parse()?;
    let g0 = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;
    let rows_all: &[u32] = &[1, 2, 4, 8, 12, 16];
    for &(label, n, k) in SHAPES {
        let (nu, ku) = (n as usize, k as usize);
        let packed = g.alloc(nu * ku / 2)?;
        g.memset(packed, 0x23, nu * ku / 2)?;
        let scales = g.alloc(nu * ku / 16)?;
        g.memset(scales, 0x38, nu * ku / 16)?;
        let a = g.alloc(16 * ku * 2)?;
        g.memset(a, 0x3C, 16 * ku * 2)?;
        let out = g.alloc(16 * nu * 2)?;
        let bytes = (nu * ku / 2 + nu * ku / 16) as f64;
        let floor_us = bytes / (peak * 1e9) * 1e6;
        eprintln!(
            "── {label} N={n} K={k}  weights {:.1} MB, floor {floor_us:.1} us ──",
            bytes / 1e6
        );
        let points = metrale_kernels::w4a16_gemv_tc_entries::W4A16_GEMV_TC_POINTS;
        for point in points.iter().flat_map(|t| t.iter()) {
            let entry = format!("w4a16_gemv_{point}");
            let Ok(h) = g.kernel("w4a16_gemv_tc", &entry) else {
                eprintln!("    {entry:<30} not in this module set");
                continue;
            };
            let (mt, nt) = geometry(point);
            for &m in rows_all.iter().filter(|&&m| m <= mt && (mt == 8 || m > 8)) {
                let launch = || {
                    KernelLaunch::new(g, h)
                        .grid([n.div_ceil(8 * nt), 1, 1])
                        .block([256, 1, 1])
                        .arg_ptr(a)
                        .arg_ptr(packed)
                        .arg_ptr(scales)
                        .arg_f32(1.0)
                        .arg_ptr(out)
                        .arg_u32(m)
                        .arg_u32(n)
                        .arg_u32(k)
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
                    "    {entry:<30} M={m:>2}  {us:>8.1} us  {gbps:>7.1} GB/s  {:>5.1}% of peak  {:>5.2}x floor",
                    100.0 * gbps / peak,
                    us / floor_us
                );
            }
        }
        for p in [packed, scales, a, out] {
            let _ = g.free(p);
        }
    }
    Ok(())
}
