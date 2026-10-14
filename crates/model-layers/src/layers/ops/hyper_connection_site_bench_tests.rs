// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Timing of the one-launch hyper-connection site (`hc_lowrank_site`) against the
//! three-launch route it replaces (`hc_pre_stage_bf16` + `hc_lowrank_down_inject` +
//! `hc_lowrank_up_mix`), on the model's site (H 2560, hc 4, rank 320, with the injection) at
//! T in {1, 2, 4, 8, 16}. `hc_post` is the same on both routes and is not timed.
//!
//! Each route is captured as a CUDA graph of 32 sites cycling over 8 weight sets (105 MB, twice
//! the H100's L2, so the weights stream from HBM as they do in a forward) and replayed 200
//! times; us/site is the replay wall time over the sites, which in a graph is device time.
//!
//! GPU test: run with
//! ```text
//! METRALE_HC_SITE_FUSED=1 cargo test -p metrale-model-layers --release \
//!     hc_site_fused_timing -- --ignored --nocapture --test-threads=1
//! ```

use std::time::Instant;

use metrale_gpu_runtime::gpu::GpuBackend;

use super::hyper_connection_lowrank::gemv::HcGemv;
use super::hyper_connection_site_tests::{
    Bufs, EPS, launch_current, resolved_site, site_args, streams, weights,
};

const SHAPE: (u32, u32, u32) = (2560, 4, 320);
const SETS: u32 = 8;
const SITES_PER_GRAPH: u32 = 32;
const REPLAYS: u32 = 200;
const WARMUP: u32 = 10;

/// 2026-10-14: Replay `graph` and return us per site.
fn time_graph(
    g: &dyn GpuBackend,
    graph: metrale_gpu_runtime::gpu::GraphHandle,
    stream: u64,
) -> f64 {
    for _ in 0..WARMUP {
        g.launch_graph(graph, stream).unwrap();
    }
    g.synchronize(stream).unwrap();
    let t0 = Instant::now();
    for _ in 0..REPLAYS {
        g.launch_graph(graph, stream).unwrap();
    }
    g.synchronize(stream).unwrap();
    t0.elapsed().as_secs_f64() * 1e6 / f64::from(REPLAYS * SITES_PER_GRAPH)
}

#[test]
#[ignore]
fn hc_site_fused_timing() {
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 not in this build");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let site = resolved_site(g);
    let gemv = HcGemv::resolve(g);
    let sets: Vec<_> = (0..SETS).map(|i| weights(g, SHAPE, 31 * i)).collect();
    let (h, hc, rank) = SHAPE;
    let k = u64::from(hc * h);
    let weight_bytes = (2 * u64::from(rank) * k + u64::from(hc) * k + k) * 2;
    println!(
        "hc site, H {h} hc {hc} rank {rank}, {:.2} MB of weights per site",
        weight_bytes as f64 / 1e6
    );
    println!("   T   current us/site   fused us/site   speedup   fused GB/s");
    let stream = g.create_stream().unwrap();
    for t in [1u32, 2, 4, 8, 16] {
        let inputs: Vec<_> = sets
            .iter()
            .enumerate()
            .map(|(i, w)| (streams(g, w, t, 500 + i as u32), Bufs::new(g, w, t)))
            .collect();

        g.begin_capture(stream).unwrap();
        for i in 0..SITES_PER_GRAPH as usize {
            let (x, b) = &inputs[i % inputs.len()];
            launch_current(g, &gemv, &sets[i % sets.len()], *x, b, true, stream);
        }
        let current = g.end_capture(stream).unwrap();

        g.begin_capture(stream).unwrap();
        for i in 0..SITES_PER_GRAPH as usize {
            let (x, b) = &inputs[i % inputs.len()];
            let w = &sets[i % sets.len()];
            let a = site_args(w, *x, b, true);
            assert!(site.admits(&a, t, h, hc, rank, 2 * (k as usize) * t as usize));
            site.launch(g, &a, t, h, hc, rank, EPS, stream).unwrap();
        }
        let fused = g.end_capture(stream).unwrap();

        let us_current = time_graph(g, current, stream);
        let us_fused = time_graph(g, fused, stream);
        g.destroy_graph(current).unwrap();
        g.destroy_graph(fused).unwrap();
        println!(
            "{t:4} {us_current:17.2} {us_fused:15.2} {:9.2}x {:12.0}",
            us_current / us_fused,
            weight_bytes as f64 / (us_fused * 1e3)
        );
    }
}
