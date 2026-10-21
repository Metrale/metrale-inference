// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-21: Microbench of programmatic dependent launch (`METRALE_GLM_PDL`) on the GLM-5.3
//! decode chain (`common/pdl_chain.rs`): `LAYERS` KDA layers, `LAYERS` MoE layers, and `LAYERS`
//! KDA + MoE pairs, each with its own weights (so the W4A16 weights stream from DRAM as in the
//! serve), captured into one graph per arm on a backend with the mode and on one without, then
//! timed interleaved after a clock warm-up (GB10's SM clock ramps after an idle spell). Reports
//! us per layer for each arm, the saving, and the projection to the serve's step at the 1k trace's
//! layer counts (34 KDA, 42 MoE layers per step, `KDA_PER_STEP` / `MOE_PER_STEP`).
//!
//! Owner: gpu-examples.
//! Invariants: every arm is timed in every round, in the same order, after the same warm-up.
//!
//!   cargo run -p metrale-gpu-examples --release --example pdl_chain_bench --features cuda,gpu-examples

#[path = "common/pdl_chain_bufs.rs"]
mod bufs;
#[path = "common/pdl_chain.rs"]
mod chain;

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use bufs::*;
use chain::{Kern, kda_layer, moe_layer};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{GpuBackend, GraphHandle};

const LAYERS: usize = 8;
const ROUNDS: usize = 9;
const TRIALS: usize = 15;
const WARM: Duration = Duration::from_millis(300);
const KDA_PER_STEP: f64 = 34.0;
const MOE_PER_STEP: f64 = 42.0;

#[derive(Clone, Copy)]
enum Part {
    Kda,
    Moe,
    Pair,
}

fn capture(g: &dyn GpuBackend, k: &Kern, b: &Bufs, part: Part, s: u64) -> Result<GraphHandle> {
    g.begin_capture(s)?;
    let r = (|| -> Result<()> {
        for l in &b.layers {
            if matches!(part, Part::Kda | Part::Pair) {
                kda_layer(g, k, b, l, s)?;
            }
            if matches!(part, Part::Moe | Part::Pair) {
                moe_layer(g, k, b, l, s)?;
            }
        }
        Ok(())
    })();
    if let Err(e) = r {
        g.abort_capture_if_active(s);
        return Err(e);
    }
    g.end_capture(s)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> Result<()> {
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    let off = MetraleCudaBackend::new(0, &glm.modules)?;
    let on = MetraleCudaBackend::new(0, &glm.modules)?;
    let (ko, kn) = (Kern::resolve(&off)?, Kern::resolve(&on)?);
    anyhow::ensure!(
        on.enable_programmatic_launches()?,
        "programmatic launches unavailable"
    );
    let s = off.default_stream();
    let b = Bufs::new(&off, 0x5eed_0002, LAYERS)?;
    let parts = [
        ("KDA layer", Part::Kda),
        ("MoE layer", Part::Moe),
        ("KDA+MoE pair", Part::Pair),
    ];
    // 2026-10-21: (off, on) graphs per part.
    let mut arms: Vec<(GraphHandle, GraphHandle)> = Vec::new();
    for &(_, p) in &parts {
        arms.push((capture(&off, &ko, &b, p, s)?, capture(&on, &kn, &b, p, s)?));
    }
    let mut per: Vec<[Vec<f64>; 2]> = (0..parts.len()).map(|_| [Vec::new(), Vec::new()]).collect();
    for _ in 0..ROUNDS {
        let t0 = Instant::now();
        while t0.elapsed() < WARM {
            for &(a, c) in &arms {
                off.launch_graph(a, s)?;
                on.launch_graph(c, s)?;
            }
            off.synchronize(s)?;
        }
        for (i, &(a, c)) in arms.iter().enumerate() {
            for (j, gr) in [a, c].into_iter().enumerate() {
                let mut v = Vec::with_capacity(TRIALS);
                for _ in 0..TRIALS {
                    let t = Instant::now();
                    off.launch_graph(gr, s)?;
                    off.synchronize(s)?;
                    v.push(t.elapsed().as_secs_f64() * 1e6 / LAYERS as f64);
                }
                per[i][j].push(median(&mut v));
            }
        }
    }
    println!(
        "PDL microbench: {LAYERS} distinct layers per graph, M = {T}, {ROUNDS} rounds x {TRIALS} replays"
    );
    println!(
        "{:<14} {:>22} {:>22} {:>9}",
        "us per layer", "PDL off [min..max]", "PDL on [min..max]", "saved"
    );
    let mut saved = [0.0f64; 3];
    for (i, (name, _)) in parts.iter().enumerate() {
        let st = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            (v[v.len() / 2], v[0], v[v.len() - 1])
        };
        let (a, c) = (st(&mut per[i][0]), st(&mut per[i][1]));
        saved[i] = a.0 - c.0;
        println!(
            "{name:<14} {:>8.1} [{:>5.1}..{:>5.1}] {:>8.1} [{:>5.1}..{:>5.1}] {:>6.2} us ({:+.1}%)",
            a.0,
            a.1,
            a.2,
            c.0,
            c.1,
            c.2,
            saved[i],
            -100.0 * saved[i] / a.0
        );
    }
    let proj = (saved[0] * KDA_PER_STEP + saved[1] * MOE_PER_STEP) / 1000.0;
    println!(
        "projected step saving: {:.3} ms ({KDA_PER_STEP} KDA x {:.2} us + {MOE_PER_STEP} MoE x {:.2} us); \
         pair arm alone x {KDA_PER_STEP}: {:.3} ms",
        proj,
        saved[0],
        saved[1],
        saved[2] * KDA_PER_STEP / 1000.0
    );
    for (a, c) in arms {
        off.destroy_graph(a)?;
        on.destroy_graph(c)?;
    }
    Ok(())
}
