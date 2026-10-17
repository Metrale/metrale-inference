// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: Part 4 of `glm5next_moe_narrow_bench`, no fixture needed: `forward_moe` with the
//! shared expert forked onto a side stream (`METRALE_GLM_SHARED_FORK`'s workspace) against the
//! same site without the fork, on a seeded router and shared expert, at 1..=4 rows: every output
//! byte, eager and replayed from a CUDA graph, then both timed as graphs.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Exits with an error on the first byte mismatch.
//! - Negative control: the same site with other shared-expert weights must change the bytes, so
//!   the comparison covers the shared expert's contribution (a missing or late join would read a
//!   previous token's `shared_out`).

use std::cell::Cell;

use anyhow::{Result, ensure};
use metrale_config::{ActivationQuantization, Nvfp4Act, ProjFamily};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_arch::glm5next_mlp::build::build_moe;
use metrale_model_arch::glm5next_mlp::forward::{Glm5NextMlpWorkspace, forward_moe};
use metrale_model_arch::glm5next_mlp::precision::{GroupPrecision, MlpGroup};
use metrale_model_arch::glm5next_mlp::weights::{Glm5NextExpertWeights, Glm5NextMoeWeights};
use metrale_model_arch::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};

use crate::Setup;
use crate::device::*;
use crate::forward::config;

const TOKENS: usize = 32;
const FORK_ROWS: usize = 4;

/// 2026-10-11: The site with a router drawn from `router_seed` and a shared expert from
/// `shared_seed` (the routed experts are the setup's tables).
fn moe(
    g: &dyn GpuBackend,
    s: &Setup,
    c: &Glm5NextMlpConfig,
    k: &Glm5NextMlpKernels,
    router_seed: u64,
    shared_seed: u64,
) -> Result<Glm5NextMoeWeights> {
    let mut rr = Rng(router_seed);
    let router: Vec<f32> = (0..EXPERTS * HIDDEN).map(|_| rr.unit() * 0.05).collect();
    let bias = vec![0.0f32; EXPERTS];
    let mut rs = Rng(shared_seed);
    let shared: Vec<f32> = (0..HIDDEN * 688 * 3).map(|_| rs.unit() * 0.02).collect();
    let load = |name: &str| -> Result<Vec<f32>> {
        Ok(match name {
            "mlp.gate.weight" => router.clone(),
            "mlp.gate.e_score_correction_bias" => bias.clone(),
            _ => shared.clone(),
        })
    };
    let expert = |id: usize| -> Result<Glm5NextExpertWeights> {
        Ok(Glm5NextExpertWeights {
            gate_proj: s.gate.projs[id],
            up_proj: s.upt.projs[id],
            down_proj: s.down.projs[id],
        })
    };
    build_moe(
        g,
        c,
        688 * 3,
        &load,
        &expert,
        &|has| {
            GroupPrecision::resolve(
                MlpGroup::RoutedExperts,
                ActivationQuantization::default()
                    .ladder(ProjFamily::Moe)
                    .clone(),
                Nvfp4Act::A4,
                k.w4a4_expert_rows(),
                has,
            )
        },
        MAX_ROWS,
    )
}

pub fn run(gpu: &MetraleCudaBackend, s: &Setup, st: u64) -> Result<()> {
    let g: &dyn GpuBackend = gpu;
    let c = config();
    let k = Glm5NextMlpKernels::resolve(g, metrale_gpu_examples::glm_levers()?)?;
    let w = moe(g, s, &c, &k, 0x510e_527f_ade6_82d1, 0x1f83_d9ab_fb41_bd6b)?;
    let mut rng = Rng(0x5be0_cd19_137e_2179);
    let ws = Glm5NextMlpWorkspace::new_forked(g, &c, MAX_ROWS, false, 0)?;
    let wf = Glm5NextMlpWorkspace::new_forked(g, &c, MAX_ROWS, false, FORK_ROWS)?;
    let x: Vec<f32> = (0..TOKENS * HIDDEN).map(|_| rng.unit() * 3.0).collect();
    let xd = up_bf16(g, &x)?;
    let (o_ref, o_fork) = (s.out[0], s.out[1]);
    let row_bytes = HIDDEN * 2;

    // 2026-10-11: Eager, then one captured forward replayed, per row count and start token.
    for rows in 1..=FORK_ROWS {
        for t in (0..=TOKENS - rows).step_by(3) {
            let xt = xd.offset(t * row_bytes);
            forward_moe(g, &k, &c, &w, xt, o_ref, rows, &ws, false, st)?;
            g.synchronize(st)?;
            let want = read(g, o_ref, rows * row_bytes)?;
            forward_moe(g, &k, &c, &w, xt, o_fork, rows, &wf, false, st)?;
            g.synchronize(st)?;
            ensure!(
                read(g, o_fork, rows * row_bytes)? == want,
                "rows {rows} token {t}: forked forward_moe differs (eager)"
            );
            g.memset(o_fork, 0xA5, rows * row_bytes)?;
            g.begin_capture(st)?;
            if let Err(e) = forward_moe(g, &k, &c, &w, xt, o_fork, rows, &wf, true, st) {
                g.abort_capture_if_active(st);
                return Err(e);
            }
            let graph = g.end_capture(st)?;
            g.launch_graph(graph, st)?;
            g.synchronize(st)?;
            g.destroy_graph(graph)?;
            ensure!(
                read(g, o_fork, rows * row_bytes)? == want,
                "rows {rows} token {t}: forked forward_moe differs (graph replay)"
            );
        }
    }
    // 2026-10-11: Negative control: the same router with other shared-expert weights must
    // change the output.
    let w2 = moe(g, s, &c, &k, 0x510e_527f_ade6_82d1, 0x9b05_688c_2b3e_6c1f)?;
    forward_moe(g, &k, &c, &w, xd, o_ref, 1, &ws, false, st)?;
    forward_moe(g, &k, &c, &w2, xd, o_fork, 1, &wf, false, st)?;
    g.synchronize(st)?;
    ensure!(
        read(g, o_ref, row_bytes)? != read(g, o_fork, row_bytes)?,
        "negative control: other weights left the forked output equal"
    );
    println!(
        "shared fork: forward_moe at 1..={FORK_ROWS} rows byte-identical with and without the \
         fork (eager and graph replay); negative control fires"
    );

    // 2026-10-11: Timing, one row, graphs of `ITERS` sites (one per token in turn).
    let i = Cell::new(0usize);
    let step = |wsx: &Glm5NextMlpWorkspace, o| -> Result<()> {
        let t = (i.get() * 7) % TOKENS;
        i.set(i.get() + 1);
        forward_moe(g, &k, &c, &w, xd.offset(t * row_bytes), o, 1, wsx, true, st)
    };
    let t = time_set(g, st, &[&|| step(&ws, o_ref), &|| step(&wf, o_fork)])?;
    println!(
        "  forward_moe 1 row: shared on the main stream {}  shared forked {}",
        fmt(t[0]),
        fmt(t[1])
    );
    Ok(())
}
