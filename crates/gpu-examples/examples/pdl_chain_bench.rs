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
//! `PDL_BENCH_BASE_PTX=<dir>` adds an arm on the kernels before the change (see `with_base`).
//! 2026-10-21: The "PDL on new" arm runs the new side of the bit-identical launch levers
//! (`Variant`; `PDL_CHAIN_OLD=serial_sweep` keeps the named ones old).
//!
//!   cargo run -p metrale-gpu-examples --release --example pdl_chain_bench --features cuda,gpu-examples

#[path = "common/pdl_chain_bufs.rs"]
mod bufs;
#[path = "common/pdl_chain.rs"]
mod chain;

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use bufs::*;
use chain::{Kern, Variant, kda_layer, moe_layer};
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

/// 2026-10-21: The GLM module set with each module whose `<name>.ptx` is in `dir` replaced by
/// that file (leaked: a backend's modules are `'static`), for the base arm.
fn with_base(
    modules: &[(&'static str, &'static [u8])],
    dir: &str,
) -> Result<Vec<(&'static str, &'static [u8])>> {
    let mut n = 0;
    let out = modules
        .iter()
        .map(
            |&(name, blob)| match std::fs::read(format!("{dir}/{name}.ptx")) {
                Ok(b) => {
                    n += 1;
                    (name, &*Box::leak(b.into_boxed_slice()))
                }
                Err(_) => (name, blob),
            },
        )
        .collect();
    anyhow::ensure!(n > 0, "{dir}: no module PTX found");
    Ok(out)
}

fn main() -> Result<()> {
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    let off = MetraleCudaBackend::new(0, &glm.modules)?;
    let on = MetraleCudaBackend::new(0, &glm.modules)?;
    anyhow::ensure!(
        on.enable_programmatic_launches()?,
        "programmatic launches unavailable"
    );
    // 2026-10-21: `PDL_BENCH_BASE_PTX=<dir>`: a third arm on the kernels before this change
    // (their PTX, one `<module>.ptx` per module), launched serialised, in the same process.
    let base = match std::env::var("PDL_BENCH_BASE_PTX") {
        Ok(dir) => Some(MetraleCudaBackend::new(0, &with_base(&glm.modules, &dir)?)?),
        Err(_) => None,
    };
    // 2026-10-21: A third arm: PDL on with the new side of the levers `Variant` names.
    let v = Variant::from_env();
    let mut backs: Vec<(&str, &MetraleCudaBackend, Variant)> = vec![
        ("PDL off", &off, Variant::OLD),
        ("PDL on", &on, Variant::OLD),
        ("PDL on new", &on, v),
    ];
    if let Some(b) = &base {
        backs.push(("base (old)", b, Variant::OLD));
    }
    println!("new arm: {v:?}");
    let s = off.default_stream();
    let b = Bufs::new(&off, 0x5eed_0002, LAYERS, 0.0)?;
    let parts = [
        ("KDA layer", Part::Kda),
        ("MoE layer", Part::Moe),
        ("KDA+MoE pair", Part::Pair),
    ];
    let mut graphs: Vec<Vec<GraphHandle>> = Vec::new();
    for &(_, p) in &parts {
        let mut row = Vec::new();
        for &(_, g, v) in &backs {
            row.push(capture(g, &Kern::resolve(g, v)?, &b, p, s)?);
        }
        graphs.push(row);
    }
    let mut per: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); backs.len()]; parts.len()];
    for _ in 0..ROUNDS {
        let t0 = Instant::now();
        while t0.elapsed() < WARM {
            for row in &graphs {
                for &gr in row {
                    off.launch_graph(gr, s)?;
                }
            }
            off.synchronize(s)?;
        }
        for (i, row) in graphs.iter().enumerate() {
            for (j, &gr) in row.iter().enumerate() {
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
        "PDL microbench: {LAYERS} distinct layers per graph, M = {T}, {ROUNDS} rounds x {TRIALS} replays; us per layer, median [min..max]"
    );
    let header: Vec<String> = backs.iter().map(|(n, ..)| format!("{n:>22}")).collect();
    println!(
        "{:<14}{} {:>12} {:>12}{}",
        "",
        header.join(""),
        "on vs off",
        "new vs on",
        if base.is_some() { "  on vs base" } else { "" }
    );
    let mut saved = vec![[0.0f64; 3]; parts.len()];
    for (i, (name, _)) in parts.iter().enumerate() {
        let st: Vec<(f64, f64, f64)> = per[i]
            .iter_mut()
            .map(|v| {
                v.sort_by(f64::total_cmp);
                (v[v.len() / 2], v[0], v[v.len() - 1])
            })
            .collect();
        let cells: Vec<String> = st
            .iter()
            .map(|a| format!("{:>8.1} [{:>5.1}..{:>5.1}]", a.0, a.1, a.2))
            .collect();
        saved[i][0] = st[0].0 - st[1].0;
        saved[i][1] = st.get(3).map_or(f64::NAN, |b| b.0 - st[1].0);
        saved[i][2] = st[1].0 - st[2].0;
        print!(
            "{name:<14}{} {:>9.2} us {:>9.2} us",
            cells.join(""),
            saved[i][0],
            saved[i][2]
        );
        if base.is_some() {
            print!(" {:>9.2} us", saved[i][1]);
        }
        println!();
    }
    for (c, what) in [(0usize, "on vs off"), (2, "new vs on"), (1, "on vs base")] {
        if c == 1 && base.is_none() {
            continue;
        }
        println!(
            "projected step saving ({what}): {:.3} ms = {KDA_PER_STEP} KDA x {:.2} us + {MOE_PER_STEP} MoE x {:.2} us",
            (saved[0][c] * KDA_PER_STEP + saved[1][c] * MOE_PER_STEP) / 1000.0,
            saved[0][c],
            saved[1][c]
        );
    }
    for row in graphs {
        for gr in row {
            off.destroy_graph(gr)?;
        }
    }
    Ok(())
}
