// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `w8a16_gemm_pipe128` (what `ops::w8a16_gemm_pipelined` launches from 256 rows
//! when the module is present) against `w8a16_gemm_pipelined` launched directly, at the W8A16
//! prefill projection shapes of Qwen3.6-35B-A3B.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless, at every shape, the two outputs are byte-identical over all
//!   M x N values (both are first filled with a sentinel).
//!
//! Weights use all 256 byte codes, NaN codes included (the LUT decodes those to zero on both
//! legs); activations are BF16 with a wide range. Mean times over 10 launches are printed.
//!
//! Run (GB10): cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!   --example w8a16_gemm_pipe128_microtest

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use metrale_model_layers::layers::ops;
use std::time::Instant;

const SENTINEL: u8 = 0x5a;
/// 2026-09-28: (M, N, K): the GDN out-projection and the attention output and gated-query
/// projections at chunk and short-prompt row counts, M not a multiple of 128 included.
const SHAPES: [(usize, usize, usize); 5] = [
    (8200, 2048, 4096),
    (5111, 2048, 4096),
    (8200, 8192, 2048),
    (300, 2048, 4096),
    (1025, 384, 256),
];

fn next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 16
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    ensure!(
        gpu.has_module("w8a16_gemm_pipe128"),
        "this build lacks the twin module"
    );
    ensure!(
        std::env::var_os("METRALE_NO_W8A16_PIPE128").is_none(),
        "unset METRALE_NO_W8A16_PIPE128: ops would launch the original on both legs"
    );
    let original = gpu.kernel("w8a16_gemm_pipelined", "w8a16_gemm_pipelined")?;
    let mut state = 0x7738_6131_2026_0928u64;
    for (m, n, k) in SHAPES {
        let a: Vec<u8> = (0..m * k)
            .flat_map(|_| {
                let v = next(&mut state);
                let u = (v & 0xffff) as f32 / 65536.0 - 0.5;
                bf16::from_f32(u * 2f32.powi(((v >> 16) & 7) as i32 - 2))
                    .to_bits()
                    .to_le_bytes()
            })
            .collect();
        let w: Vec<u8> = (0..n * k).map(|_| next(&mut state) as u8).collect();
        let s: Vec<u8> = (0..(n / 128) * (k / 128))
            .flat_map(|_| (((next(&mut state) % 64 + 1) as f32) / 4096.0).to_le_bytes())
            .collect();
        let (a_d, w_d, s_d) = (upload(&gpu, &a)?, upload(&gpu, &w)?, upload(&gpu, &s)?);
        let bytes = m * n * 2;
        let outs = [gpu.alloc(bytes)?, gpu.alloc(bytes)?];
        for o in outs {
            gpu.memset(o, SENTINEL, bytes)?;
        }
        let (m32, n32, k32) = (m as u32, n as u32, k as u32);
        let run_original = || -> Result<()> {
            KernelLaunch::new(&gpu, original)
                .grid([div_ceil(n32, 32), div_ceil(m32, 128), 1])
                .block([256, 1, 1])
                .arg_ptr(a_d)
                .arg_ptr(w_d)
                .arg_ptr(s_d)
                .arg_ptr(outs[0])
                .arg_u32(m32)
                .arg_u32(n32)
                .arg_u32(k32)
                .launch(0)
        };
        let run_twin =
            || ops::w8a16_gemm_pipelined(&gpu, original, a_d, w_d, s_d, outs[1], m32, n32, k32, 0);
        run_original()?;
        run_twin()?;
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
            "M={m} N={n} K={k}: {diff} of {} values differ",
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
        let (t0, t1) = (time(&run_original)?, time(&run_twin)?);
        println!(
            "M={m:5} N={n:5} K={k:4}: bit-identical; pipelined {t0:.3} ms, pipe128 {t1:.3} ms ({:.2}x)",
            t0 / t1
        );
        for p in [a_d, w_d, s_d, outs[0], outs[1]] {
            gpu.free(p)?;
        }
    }
    println!("PASS: w8a16_gemm_pipe128 is bit-identical to w8a16_gemm_pipelined");
    Ok(())
}
