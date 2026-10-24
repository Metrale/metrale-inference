// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The CPU equivalence gate of the plan-executed routed MoE site: on the recording
//! device port, the hand-written `forward_moe` under the served recipe's levers and the site run
//! from each rank's served plan put the same operations on the stream (kernel by name, grid,
//! block, shared memory, every argument, every memset), on every MoE layer of every rank at
//! decode n=1 and verify n=2, the shared expert forked as served. A one-argument sabotage of the plan turns it red at that launch.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_circuit::{KernelId, Mode};
use metrale_gpu_runtime::gpu::recording::Op;

use super::*;
use crate::glm5next_plan::compile_moe;
use crate::glm5next_plan::fixture::{RUNS, moe_layers, plan, ranks, ranks_at, worlds};
use crate::glm5next_plan::fixture_site::{OUT, STREAM, Site, X, gate, site, site_at, weights};

fn legacy(s: &Site, w: &Glm5NextMoeWeights, rows: usize) -> Vec<Op> {
    super::super::forward_moe(&s.gpu, &s.k, &s.cfg, w, X, OUT, rows, &s.ws, true, STREAM).unwrap();
    s.gpu.take()
}

fn planned(s: &Site, w: &Glm5NextMoeWeights, steps: MoeSteps<KernelId>) -> Vec<Op> {
    let rows = steps.rows;
    let p = bind(&s.gpu, steps, &s.cfg, w, &s.ws).unwrap();
    run_with_legacy_shared(&p, &s.gpu, &s.k, &s.cfg, w, &s.ws, X, OUT, rows, STREAM).unwrap();
    s.gpu.take()
}

/// 2026-10-23: Every routed MoE site of every rank, at decode n=1 and verify n=2, launches from
/// its plan exactly what `forward_moe` launches under the served levers. The logs are not
/// trivially equal: each holds the site's whole launch sequence, which the counts pin.
#[test]
fn every_plan_executed_moe_site_launches_what_the_legacy_site_launches() {
    let (mut sites, mut want_sites) = (0, 0);
    for (world, rank) in worlds()
        .into_iter()
        .flat_map(|w| (0..w).map(move |r| (w, r)))
    {
        let (c, rules, policy) = ranks_at(world).unwrap();
        let s = site_at(rank, world);
        want_sites += 2 * 42;
        for (mode, rows) in RUNS {
            let p = plan(&c, &rules, &policy, rank, mode, rows);
            for l in moe_layers(&c[rank]) {
                let w = weights(l, false);
                let steps = compile_moe(&c[rank], &p, l).unwrap().steps;
                let (a, b) = (legacy(&s, &w, rows as usize), planned(&s, &w, steps));
                if let Some(d) = gate(&a, &b) {
                    panic!("rank {rank} {mode:?} n={rows} layer {l}: {d}");
                }
                // 2026-10-23: the fork (record, side waits), the shared expert's 4 launches and
                // its join record on the side stream (the served METRALE_GLM_SHARED_FORK=2 takes
                // both row counts), router (1 per row, or 1), top-k, quant, [union], gate/up
                // sweep, SwiGLU, quant, down sweep, the main stream's wait, combine.
                let want = if rows == 1 { 16 } else { 17 };
                assert_eq!(a.len(), want, "rank {rank} {mode:?} layer {l}: {a:#?}");
                let side = a
                    .iter()
                    .filter(|o| matches!(o, Op::Launch { stream, .. } if *stream != STREAM))
                    .count();
                assert_eq!(
                    side, 4,
                    "rank {rank} layer {l}: the shared expert runs forked"
                );
                sites += 1;
            }
        }
    }
    assert_eq!(sites, want_sites);
    assert!(sites > 3 * 2 * 42, "more than the served world's sites");
}

fn one_site(rank: usize, mode: Mode, rows: u64) -> (Site, Glm5NextMoeWeights, MoeSteps<KernelId>) {
    let (c, rules, policy) = ranks();
    let p = plan(&c, &rules, &policy, rank, mode, rows);
    let steps = compile_moe(&c[rank], &p, 4).unwrap().steps;
    (site(rank), weights(4, false), steps)
}

/// 2026-10-23: Changing one field of the plan turns the gate red at the launch that carries it:
/// the combine's grid (blocks per row), the top-k block width, and the down quantizer (the plain
/// entry where the 704-column slice needs the `_k64` twin). The unchanged plan is green on the
/// same site, so each red is the sabotage's.
#[test]
fn a_one_field_sabotage_turns_the_gate_red_at_its_launch() {
    let (s, w, steps) = one_site(0, Mode::Verify, 2);
    let a = legacy(&s, &w, 2);
    assert_eq!(gate(&a, &planned(&s, &w, steps.clone())), None);
    let plan_side = |d: &str| d.lines().nth(2).unwrap().to_string();
    let mut blend = steps.clone();
    blend.blend.blocks_per_row = 1;
    let d = gate(&a, &planned(&s, &w, blend)).expect("a changed combine grid is red");
    assert!(plan_side(&d).contains("glm5next_moe_combine"), "{d}");
    let mut topk = steps.clone();
    topk.topk.block = 256;
    let d = gate(&a, &planned(&s, &w, topk)).expect("a changed top-k block is red");
    assert!(plan_side(&d).contains("glm5next_router_topk_rank"), "{d}");
    let mut down = steps;
    down.down.quant.func = "w4a4_quant_rows_static".into();
    let d = gate(&a, &planned(&s, &w, down)).expect("the plain down quantizer is red");
    assert!(
        plan_side(&d).contains("w4a4_gemv_mx_moe::w4a4_quant_rows_static\""),
        "{d}"
    );
}

/// 2026-10-23: `bind` refuses a site the plan was not compiled for: rank 2's plan on rank 0's
/// site (another expert slice), and compact-scale entries over plain tables.
#[test]
fn bind_refuses_another_ranks_site_and_another_scale_form() {
    let (_, w, steps2) = one_site(2, Mode::Decode, 1);
    let s0 = site(0);
    let e = bind(&s0.gpu, steps2, &s0.cfg, &w, &s0.ws).err().unwrap();
    assert!(format!("{e:#}").contains("expert slice"), "{e:#}");
    let (s, _, steps) = one_site(0, Mode::Decode, 1);
    let e = bind(&s.gpu, steps, &s.cfg, &weights(4, true), &s.ws)
        .err()
        .unwrap();
    assert!(format!("{e:#}").contains("Compact"), "{e:#}");
}
