// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Microbench for `METRALE_GLM_AR_L2_PREFETCH`: how much faster a one-row
//! `w4a16_gemv` runs when its weights were read through L2 (`l2_warm_ldcg`) just before, and
//! how long the warm itself takes (in the serve it hides under the all-reduce wait). Also proves
//! the `L2Warm` fork/join captures into a CUDA graph and replays.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Every site has its own weight and the sites together exceed L2, so the cold arm reads DRAM.
//! - Arms, each captured as a graph over `SITES` sites and replayed `TRIALS` times, us/site:
//!   `cold` gemv; `warm` the warm alone; `warm+gemv` in order on one stream (gemv after warm =
//!   `warm+gemv` - `warm`); `fork` the layer's path (`L2Warm::fork` on the side stream, the next
//!   fork or the final join on the main one, then the gemv).
//! - The output is not checked: the warm only reads (`l2_warm.cu`), the GEMV is unchanged.
//!
//!   cargo run -p metrale-model-arch --release --example glm5next_l2_warm_bench \
//!       --features cuda,gpu-examples

use anyhow::Result;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_layer::l2_warm::{L2Warm, Range};
use metrale_model_layers::layers::ops::w4a16_gemv;
use metrale_model_layers::weight_map::QuantizedWeight;

/// 2026-10-10: `(n, k)`: the KDA q/k/v projection at TP 3 (2816 x 4096, ~6.5 MB) and a narrower
/// one.
const SHAPES: [(usize, usize); 2] = [(2816, 4096), (1408, 4096)];
const SITES: usize = 12;
const TRIALS: usize = 9;

/// 2026-10-10: One site: an NVFP4 weight with random codes and unit-ish scales.
struct Site {
    w: QuantizedWeight,
    ranges: Vec<Range>,
}

fn site(g: &dyn GpuBackend, n: usize, k: usize, seed: u64) -> Result<Site> {
    let mut s = seed | 1;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s as u8
    };
    let codes: Vec<u8> = (0..n * k / 2).map(|_| rnd()).collect();
    // 2026-10-10: E4M3 0x38 = 1.0 for every group scale.
    let scales = vec![0x38u8; n * k / 16];
    let (wp, sp) = (g.alloc(codes.len())?, g.alloc(scales.len())?);
    g.copy_h2d(&codes, wp)?;
    g.copy_h2d(&scales, sp)?;
    let mut w = QuantizedWeight::null();
    w.weight = wp;
    w.weight_scale = sp;
    w.weight_scale_2 = 1.0;
    Ok(Site {
        w,
        ranges: vec![(wp, codes.len()), (sp, scales.len())],
    })
}

fn graph_us(
    g: &dyn GpuBackend,
    stream: u64,
    mut f: impl FnMut(u64, usize) -> Result<()>,
) -> Result<(f64, f64, f64)> {
    g.begin_capture(stream)?;
    for i in 0..SITES {
        if let Err(e) = f(stream, i) {
            g.abort_capture_if_active(stream);
            return Err(e);
        }
    }
    let graph = g.end_capture(stream)?;
    g.launch_graph(graph, stream)?;
    g.synchronize(stream)?;
    let mut v = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        let t0 = std::time::Instant::now();
        g.launch_graph(graph, stream)?;
        g.synchronize(stream)?;
        v.push(t0.elapsed().as_secs_f64() * 1e6 / SITES as f64);
    }
    g.destroy_graph(graph)?;
    v.sort_by(|a, b| a.total_cmp(b));
    Ok((v[TRIALS / 2], v[0], v[TRIALS - 1]))
}

fn warm(g: &dyn GpuBackend, k: KernelHandle, sink: DevicePtr, r: &[Range], st: u64) -> Result<()> {
    let blocks = g.sm_count()? * 2;
    for &(p, bytes) in r {
        KernelLaunch::new(g, k)
            .grid([blocks, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(p)
            .arg_u64((bytes / 16) as u64)
            .arg_ptr(sink)
            .launch(st)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let gemv = g.kernel("w4a16_gemv", "w4a16_gemv")?;
    let warm_k = g.kernel("l2_warm", "l2_warm_ldcg")?;
    let sink = g.alloc(4)?;
    let stream = g.create_stream()?;
    println!("l2 warm bench: graph of {SITES} sites x {TRIALS} replays, us/site median [min, max]");
    for (n, k) in SHAPES {
        let sites: Vec<Site> = (0..SITES)
            .map(|i| site(g, n, k, 0x9e37_79b9 ^ (i as u64) << 20 ^ n as u64))
            .collect::<Result<_>>()?;
        let a = g.alloc(k * 2)?;
        g.copy_h2d(&vec![0x3fu8; k * 2], a)?;
        let c = g.alloc(n * 2)?;
        let gemv_at =
            |st: u64, i: usize| w4a16_gemv(g, gemv, a, &sites[i].w, c, n as u32, k as u32, st);
        let cold = graph_us(g, stream, gemv_at)?;
        let w = graph_us(g, stream, |st, i| {
            warm(g, warm_k, sink, &sites[i].ranges, st)
        })?;
        let both = graph_us(g, stream, |st, i| {
            warm(g, warm_k, sink, &sites[i].ranges, st)?;
            gemv_at(st, i)
        })?;
        let l2 = L2Warm::new(g, usize::MAX)?;
        l2.set_plan(sites.iter().map(|s| s.ranges.clone()).collect())?;
        let fork = graph_us(g, stream, |st, i| {
            l2.fork(g, i, 1, st)?;
            l2.join(g, st)?;
            gemv_at(st, i)
        })?;
        let mb = (n * k / 2 + n * k / 16) as f64 / 1e6;
        println!(
            "  [{n} x {k}] {mb:.2} MB: cold gemv {:6.2} [{:6.2},{:6.2}] | warm {:6.2} [{:6.2},{:6.2}] | \
             warm+gemv {:6.2} [{:6.2},{:6.2}] -> gemv after warm ~{:6.2} | fork path {:6.2} [{:6.2},{:6.2}]",
            cold.0,
            cold.1,
            cold.2,
            w.0,
            w.1,
            w.2,
            both.0,
            both.1,
            both.2,
            both.0 - w.0,
            fork.0,
            fork.1,
            fork.2
        );
        for s in &sites {
            g.free(s.w.weight)?;
            g.free(s.w.weight_scale)?;
        }
        g.free(a)?;
        g.free(c)?;
    }
    Ok(())
}
