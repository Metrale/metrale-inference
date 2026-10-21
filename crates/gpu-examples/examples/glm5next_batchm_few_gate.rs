// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Bit-identity gate and microbench: `dense_gemv_bf16_batchm_few` (and `_fp32out`)
//! against the runtime-M `dense_gemv_bf16_batchm` (and `_fp32out`), the entry `glm_mm` launches at
//! 2..=8 rows without `METRALE_GLM_BATCHM_FEW`.
//!
//! Owner: gpu examples (GLM-5.3 dense projections).
//! Invariants:
//! - Fails unless every output byte matches at M = 2..=8 for both output types, at the serving
//!   shapes (the MoE router 288 x 4096, the DSA indexer wq_b 4096 x 1536 at TP=3) and at shapes
//!   with a partial last block (N % 4 != 0), a partial slab (K / 8 not a multiple of 64) and more
//!   than one prefetch group (K / 8 > 512), on three input families: drawn values, a wide
//!   exponent range with signed zeros, and rows built to cancel.
//! - A negative control must DIFFER: one weight of the first output row set to 1e20.
//!
//!   cargo build -p metrale-gpu-examples --release --example glm5next_batchm_few_gate \
//!       --features cuda,gpu-examples
//!
//! Timing (FP32 out, M = 2, the verify step's rows): the router at `ROUTER_CALLS` and wq_b at
//! `WQB_CALLS` calls with distinct weights per graph (graph_arms: clock-warmed, interleaved).
//! `--no-timing` skips it.

#[path = "common/glm5next_rows_fixture.rs"]
mod fixture;

use anyhow::{Result, bail};
use fixture::{Lcg, read, same, up_bf16};
use metrale_gpu_examples::graph_arms::{Plan, time_arms};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const MODULE_OLD: &str = "dense_gemv_bf16_batchm";
const MODULE_NEW: &str = "dense_gemv_bf16_batchm_few";
/// 2026-10-21: (name, N, K).
const SHAPES: [(&str, usize, usize); 4] = [
    ("router", 288, 4096),
    ("wq_b", 4096, 1536),
    ("partial", 290, 1000),
    ("long", 6, 8200),
];
const ROUTER_CALLS: usize = 42;
const WQB_CALLS: usize = 11;

fn launch(
    g: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    kk: usize,
    s: u64,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([n.div_ceil(4) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .arg_u32(n as u32)
        .launch(s)
}

/// 2026-10-21: `len` values of family `f`: 0 drawn; 1 wide exponents (2^-40 .. 2^40) with signed
/// zeros; 2 pairs (x, -x * (1 + tiny)) so sums cancel to a few ulps.
fn values(rng: &mut Lcg, len: usize, f: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let r = rng.next();
            match f {
                0 => r,
                1 => match i % 17 {
                    0 => 0.0,
                    1 => -0.0,
                    _ => r * 2f32.powi(((i * 7919) % 81) as i32 - 40),
                },
                _ => {
                    if i % 2 == 0 {
                        1.0 + r
                    } else {
                        -(1.0 + r) * (1.0 + 1e-3 * r)
                    }
                }
            }
        })
        .collect()
}

fn zeros(g: &dyn GpuBackend, n: usize) -> Result<DevicePtr> {
    let p = g.alloc(n.max(1))?;
    g.memset(p, 0, n)?;
    Ok(p)
}

fn main() -> Result<()> {
    let timing = !std::env::args().any(|a| a == "--no-timing");
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let old = [
        g.kernel(MODULE_OLD, "dense_gemv_bf16_batchm")?,
        g.kernel(MODULE_OLD, "dense_gemv_bf16_batchm_fp32out")?,
    ];
    let new = [
        g.kernel(MODULE_NEW, "dense_gemv_bf16_batchm_few")?,
        g.kernel(MODULE_NEW, "dense_gemv_bf16_batchm_few_fp32out")?,
    ];
    let mut rng = Lcg(0x6261_7463);
    println!("batchm few gate: dense_gemv_bf16_batchm is the ORACLE\n");
    for (name, n, kk) in SHAPES {
        for fam in 0..3 {
            let wv = values(&mut rng, n * kk, fam);
            let w = up_bf16(g, &wv)?;
            let av = values(&mut rng, 8 * kk, fam);
            let a = up_bf16(g, &av)?;
            // 2026-10-21: The weight of output 0 against row 0's largest activation set to 1e20,
            // far above every other term of that sum.
            let j = (0..kk).fold(0, |j, i| if av[i].abs() > av[j].abs() { i } else { j });
            let mut w2 = wv.clone();
            w2[j] = if w2[j] == 1e20 { 2e20 } else { 1e20 };
            let w2 = up_bf16(g, &w2)?;
            for m in 2..=8usize {
                for (o, out) in [(0usize, 2usize), (1, 4)] {
                    let bytes = m * n * out;
                    let (c0, c1, c2) = (zeros(g, bytes)?, zeros(g, bytes)?, zeros(g, bytes)?);
                    launch(g, old[o], a, w, c0, m, n, kk, 0)?;
                    launch(g, new[o], a, w, c1, m, n, kk, 0)?;
                    launch(g, new[o], a, w2, c2, m, n, kk, 0)?;
                    g.synchronize(0)?;
                    let (x, y, z) = (
                        read(g, c0, bytes)?,
                        read(g, c1, bytes)?,
                        read(g, c2, bytes)?,
                    );
                    let tag = if o == 0 { "bf16" } else { "f32" };
                    same(
                        &format!("{name} N={n} K={kk} fam={fam} M={m} {tag}"),
                        &x,
                        &y,
                    )?;
                    if z == y {
                        bail!("{name} fam={fam} M={m} {tag}: control (one weight) did not move C");
                    }
                    for p in [c0, c1, c2] {
                        g.free(p)?;
                    }
                }
            }
            for p in [w, a, w2] {
                g.free(p)?;
            }
        }
        println!(
            "  {name:8} N={n:5} K={kk:5}  M=2..8 x 3 families x bf16/f32  BYTE-IDENTICAL; control differs"
        );
    }

    if timing {
        let s = g.create_stream()?;
        let plan = |calls| Plan {
            calls,
            rounds: 5,
            trials: 9,
            warm: std::time::Duration::from_millis(300),
        };
        println!(
            "\n  TIMING (FP32 out, M=2): distinct weights per call, us per call median [min, max]"
        );
        for (name, n, kk, calls) in [
            ("router", 288, 4096, ROUTER_CALLS),
            ("wq_b", 4096, 1536, WQB_CALLS),
        ] {
            let ws: Vec<DevicePtr> = (0..calls)
                .map(|_| up_bf16(g, &values(&mut rng, n * kk, 0)))
                .collect::<Result<_>>()?;
            let a = up_bf16(g, &values(&mut rng, 2 * kk, 0))?;
            let c = zeros(g, 2 * n * 4)?;
            let arm_old = |st: u64, i: usize| launch(g, old[1], a, ws[i], c, 2, n, kk, st);
            let arm_new = |st: u64, i: usize| launch(g, new[1], a, ws[i], c, 2, n, kk, st);
            let r = time_arms(g, s, plan(calls), &[&arm_old, &arm_new])?;
            let gbs = |us: f64| (n * kk * 2) as f64 / us / 1000.0;
            println!(
                "    {name:6} {n}x{kk}  old {:6.2} [{:6.2}, {:6.2}] {:5.0} GB/s  few {:6.2} [{:6.2}, {:6.2}] {:5.0} GB/s  saves {:.3} ms/step at {calls} calls",
                r[0].0,
                r[0].1,
                r[0].2,
                gbs(r[0].0),
                r[1].0,
                r[1].1,
                r[1].2,
                gbs(r[1].0),
                (r[0].0 - r[1].0) * calls as f64 / 1000.0
            );
        }
    }
    println!("\nbatchm few gate PASS");
    Ok(())
}
