// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Ordering gate for the early graph launch (`metrale_gpu_runtime::early_launch`,
//! `METRALE_GLM_GRAPH_EARLY_LAUNCH`) on a real device: a graph captured behind the gate copies
//! buffer A into B, launched early on its side stream between two fills of A on the compute
//! stream, behind a one-block GEMV over 256 MiB that keeps the compute stream busy for
//! milliseconds while the copy engine stays idle (so an ungated copy would run at once; fills
//! alone do not show it: they share the copy engine's queue with the copy). B must hold exactly the first fill: a graph that ran
//! before the first fill finished, or a second fill that ran before the copy finished, leaves
//! other bytes in B.
//!
//! Owner: gpu-runtime examples.
//! Invariants:
//! - The run fails (panics) unless every byte of B equals the first fill in every iteration, for
//!   the early launch and for the plain launch on the compute stream (the control).
//!
//!   cargo run -p metrale-gpu-examples --release --example early_launch_gate \
//!       --features cuda,gpu-examples

use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::early_launch::EarlyLaunch;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-10-21: 256 MiB: a fill takes ~1 ms on GB10, a wide window for a misordered copy.
const N: usize = 256 << 20;
/// 2026-10-21: The delay GEMV's reduction length: 4 rows of it read all of the 256 MiB buffer.
const DELAY_K: u32 = (N / 8) as u32;

fn main() -> anyhow::Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let compute = g.default_stream();
    let (a, b, d, c) = (g.alloc(N)?, g.alloc(N)?, g.alloc(N)?, g.alloc(4096)?);
    let gemv = g.kernel("gemv", "dense_gemv_bf16")?;
    g.memset(d, 0, N)?;
    let early = EarlyLaunch::new(g, compute)?;
    EarlyLaunch::begin_capture_with(Some(&early), g, compute)?;
    g.copy_d2d_async(a, b, N, compute)?;
    let gated = g.end_capture(compute)?;
    g.begin_capture(compute)?;
    g.copy_d2d_async(a, b, N, compute)?;
    let plain = g.end_capture(compute)?;
    anyhow::ensure!(gated.0 != 0 && plain.0 != 0, "capture returned no graph");
    let mut host = vec![0u8; N];
    let mut failures = 0usize;
    for (name, early_launch) in [("early", true), ("plain", false)] {
        for it in 0..24u8 {
            let (first, second) = (0x10 + it, 0xA0 + it);
            g.memset(b, 0, N)?;
            KernelLaunch::new(g, gemv)
                .grid([1, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(d)
                .arg_ptr(d)
                .arg_ptr(c)
                .arg_u32(4)
                .arg_u32(DELAY_K)
                .launch(compute)?;
            g.memset_async(a, first, N, compute)?;
            let (with, graph) = if early_launch {
                (Some(&early), gated)
            } else {
                (None, plain)
            };
            EarlyLaunch::launch_with(with, g, graph, compute)?;
            g.memset_async(a, second, N, compute)?;
            g.synchronize(compute)?;
            g.copy_d2h(b, &mut host)?;
            let bad = host.iter().filter(|&&x| x != first).count();
            if bad != 0 {
                failures += 1;
                println!(
                    "{name} iteration {it}: {bad} bytes of B are not the first fill {first:#x}"
                );
            }
        }
        println!("{name}: 24 iterations checked");
    }
    g.destroy_graph(gated)?;
    g.destroy_graph(plain)?;
    assert_eq!(
        failures, 0,
        "early launch ordering gate: {failures} failing iterations"
    );
    println!("PASS: every byte of B is the first fill (early and plain launch)");
    Ok(())
}
