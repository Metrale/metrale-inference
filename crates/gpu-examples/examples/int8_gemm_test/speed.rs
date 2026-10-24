// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The speed section for one prefill shape: `int8_gemm_faith2`, printed as one
//! TFLOP/s line.
//!
//! Owner: model-arch examples (int8 GEMM kernels).
//! Invariants: every buffer allocated here is freed before `speed_shape` returns `Ok`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use std::time::Instant;

use crate::{Rng, bytemuck_f32, bytemuck_i8, up};

pub(super) fn speed_shape(
    gpu: &dyn GpuBackend,
    stream: u64,
    hfaith2: KernelHandle,
    label: &str,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let nb = k / 32;
    let mut rng = Rng(7);
    let a_i8: Vec<i8> = (0..m * k).map(|_| rng.i8()).collect();
    let b_i8: Vec<i8> = (0..n * k).map(|_| rng.i8()).collect();
    let a_sc: Vec<f32> = (0..m * nb).map(|_| rng.pos_scale()).collect();
    let b_sc: Vec<f32> = (0..n * nb).map(|_| rng.pos_scale()).collect();
    let a_p = up(gpu, bytemuck_i8(&a_i8))?;
    let b_p = up(gpu, bytemuck_i8(&b_i8))?;
    let as_p = up(gpu, bytemuck_f32(&a_sc))?;
    let bs_p = up(gpu, bytemuck_f32(&b_sc))?;
    let c_p = gpu.alloc(m * n * 2)?;
    let go = || -> Result<()> {
        KernelLaunch::new(gpu, hfaith2)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(a_p)
            .arg_ptr(b_p)
            .arg_ptr(as_p)
            .arg_ptr(bs_p)
            .arg_ptr(c_p)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)
    };
    for _ in 0..3 {
        go()?;
    }
    gpu.synchronize(stream)?;
    let iters = 30;
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let t0 = Instant::now();
    for _ in 0..iters {
        go()?;
    }
    gpu.synchronize(stream)?;
    let tf = flops / (t0.elapsed().as_secs_f64() / iters as f64) / 1e12;
    println!("{label}: FAITH2 {tf:.2}  (bf16=30, llama Q4K=65/Q6K=41)");
    for p in [a_p, b_p, as_p, bs_p, c_p] {
        let _ = gpu.free(p);
    }
    Ok(())
}
