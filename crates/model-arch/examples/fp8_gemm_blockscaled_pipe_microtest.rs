// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-27: `fp8_gemm_blockscaled_pipe_128x64` (what `ops::fp8_gemm_t_blockscaled`
//! launches when the module is present) against the legacy `fp8_gemm_t_blockscaled`
//! kernel launched directly, at the W8A8 prefill projection shapes of Qwen3.6-35B-A3B.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless, at every shape, the two outputs are byte-identical over all
//!   M x N values and the guard rows past M still hold the sentinel.
//!
//! Operands use every non-NaN E4M3 code. M values include ones that are not multiples of
//! the 128-row tile. Mean times over 10 launches are printed.
//!
//! Run (GB10): cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!   --example fp8_gemm_blockscaled_pipe_microtest

use anyhow::{Result, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use metrale_model_layers::layers::ops;
use std::time::Instant;

const GUARD_ROWS: usize = 2;
const SENTINEL: u8 = 0x5a;
/// 2026-09-27: (M, N, K): the GDN in-projection, the gated q projection, the o/out
/// projection, the shared expert's gate/up and down, at chunk and short-prompt row counts.
const SHAPES: [(usize, usize, usize); 7] = [
    (8200, 12288, 2048),
    (1100, 8192, 2048),
    (8200, 2048, 4096),
    (300, 512, 2048),
    (4097, 2048, 512),
    (37, 12288, 2048),
    (129, 192, 256),
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
    fn e4m3(&mut self) -> u8 {
        loop {
            let c = (self.next() >> 8) as u8;
            if c & 0x7f != 0x7f {
                return c;
            }
        }
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    ensure!(
        gpu.has_module("fp8_gemm_blockscaled_pipe"),
        "this build lacks the pipelined module"
    );
    ensure!(
        std::env::var_os("METRALE_NO_FP8_GEMM_PIPE").is_none(),
        "unset METRALE_NO_FP8_GEMM_PIPE: ops would launch the legacy kernel on both legs"
    );
    let legacy = gpu.kernel("fp8_gemm_t_blockscaled", "fp8_gemm_t_blockscaled")?;
    let mut rng = Rng(0x7069_7065_2026_0927);
    for (m, n, k) in SHAPES {
        let scales = |count: usize, rng: &mut Rng| -> Vec<u8> {
            (0..count)
                .flat_map(|_| (((rng.next() % 64 + 1) as f32) / 4096.0).to_le_bytes())
                .collect()
        };
        let a = upload(&gpu, &(0..m * k).map(|_| rng.e4m3()).collect::<Vec<_>>())?;
        let a_s = upload(&gpu, &scales(m * k / 128, &mut rng))?;
        let b = upload(&gpu, &(0..n * k).map(|_| rng.e4m3()).collect::<Vec<_>>())?;
        let b_s = upload(&gpu, &scales(n.div_ceil(128) * k / 128, &mut rng))?;
        let bytes = (m + GUARD_ROWS) * n * 2;
        let outs = [gpu.alloc(bytes)?, gpu.alloc(bytes)?];
        for o in outs {
            gpu.memset(o, SENTINEL, bytes)?;
        }
        let (m32, n32, k32) = (m as u32, n as u32, k as u32);
        let run_legacy = || -> Result<()> {
            KernelLaunch::new(&gpu, legacy)
                .grid([div_ceil(n32, 128), div_ceil(m32, 64), 1])
                .block([128, 1, 1])
                .arg_ptr(a)
                .arg_ptr(a_s)
                .arg_ptr(b)
                .arg_ptr(b_s)
                .arg_ptr(outs[0])
                .arg_u32(m32)
                .arg_u32(n32)
                .arg_u32(k32)
                .launch(0)
        };
        let run_pipe = || -> Result<()> {
            ops::fp8_gemm_t_blockscaled(&gpu, legacy, a, a_s, b, b_s, outs[1], m32, n32, k32, 0)
        };
        run_legacy()?;
        run_pipe()?;
        gpu.synchronize(0)?;
        let mut host = [vec![0u8; bytes], vec![0u8; bytes]];
        gpu.copy_d2h(outs[0], &mut host[0])?;
        gpu.copy_d2h(outs[1], &mut host[1])?;
        let live = m * n * 2;
        let diff = host[0][..live]
            .chunks_exact(2)
            .zip(host[1][..live].chunks_exact(2))
            .filter(|(x, y)| x != y)
            .count();
        ensure!(
            diff == 0,
            "M={m} N={n} K={k}: {diff} of {} values differ",
            m * n
        );
        ensure!(
            host.iter()
                .all(|h| h[live..].iter().all(|&x| x == SENTINEL)),
            "M={m} N={n} K={k}: write past row M"
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
        let (t_old, t_new) = (time(&run_legacy)?, time(&run_pipe)?);
        println!(
            "M={m:5} N={n:5} K={k:4}: bit-identical; legacy {t_old:.3} ms, pipe {t_new:.3} ms ({:.2}x)",
            t_old / t_new
        );
        for p in [a, a_s, b, b_s, outs[0], outs[1]] {
            gpu.free(p)?;
        }
    }
    println!("PASS: fp8_gemm_blockscaled_pipe_128x64 is bit-identical to fp8_gemm_t_blockscaled");
    Ok(())
}
