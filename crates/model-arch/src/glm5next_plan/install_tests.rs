// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Plan execution installed on a GLM text layer, without a device: under the served
//! levers the live policy is the instance's (the plan is fused under the switches the serve
//! runs), a changed lever changes the plan, the installed site launches what `forward_moe`
//! launches at the gated row counts and leaves every other row count to it, and a lever the
//! rules do not model refuses installation.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_circuit::{AvailableKernels, Mode};

use super::*;
use crate::glm5next_plan::compile_moe;
use crate::glm5next_plan::fixture::{plan, ranks, worlds};
use crate::glm5next_plan::fixture_site::{OUT, STREAM, X, site, site_at, weights};

/// 2026-10-23: Every rank's live policy under the served recipe's levers is the instance's: the
/// instance states the recipe, so the installed plan is the one the reports describe.
#[test]
fn the_served_levers_fuse_under_the_instance_policy() {
    let (_, _, policy) = ranks();
    for (world, rank) in worlds()
        .into_iter()
        .flat_map(|w| (0..w).map(move |r| (w, r)))
    {
        let s = site_at(rank, world);
        assert!(unmodelled(&s.k).is_empty(), "{:?}", unmodelled(&s.k));
        let live = live_policy(&policy, &s.k, &s.cfg).unwrap();
        assert_eq!(live.settings, policy.settings, "rank {rank}");
    }
}

/// 2026-10-23: A lever the policy carries reaches the plan: with the combine split off the live
/// policy says so and the fused plan's combine spreads a row over one block, not sixteen.
#[test]
fn a_changed_lever_changes_the_fused_plan() {
    let (c, rules, policy) = ranks();
    let mut s = site(0);
    s.k.levers.moe_combine_split = false;
    let live = live_policy(&policy, &s.k, &s.cfg).unwrap();
    assert_eq!(live.settings["glm_moe_combine_split"], "off");
    let avail = AvailableKernels::all_named_by(&rules);
    let p = metrale_circuit::fuse(&c[0], &rules, &avail, &live, Mode::Verify, 2).unwrap();
    let steps = compile_moe(&c[0], &p, 4).unwrap().steps;
    assert_eq!(steps.blend.blocks_per_row, 1);
    let served = plan(&c, &rules, &policy, 0, Mode::Verify, 2);
    assert_eq!(
        compile_moe(&c[0], &served, 4)
            .unwrap()
            .steps
            .blend
            .blocks_per_row,
        16
    );
}

/// 2026-10-23: Installed from the served plans (decode 1, verify 2 and the ungated verify 3),
/// layer 4 executes its routed groups at one and two rows and launches what `forward_moe`
/// launches there; at three rows, and after clearing, nothing runs from a plan.
#[test]
fn an_installed_site_runs_at_the_gated_rows_only() {
    let (c, rules, policy) = ranks();
    let s = site(0);
    let w = weights(4, false);
    let plans: Vec<_> = [(Mode::Decode, 1), (Mode::Verify, 2), (Mode::Verify, 3)]
        .into_iter()
        .map(|(m, r)| plan(&c, &rules, &policy, 0, m, r))
        .collect();
    let rp = RankPlans {
        circuit: &c[0],
        plans: &plans,
        policy: &policy,
    };
    let inst = InstalledMoe::default();
    let sites = inst
        .install(&s.gpu, 4, Some(rp), &s.k, &s.cfg, &w, &s.ws)
        .unwrap();
    // 2026-10-23: [gate] [top_k] at one row or [gate,top_k] at two, then gate/up, SwiGLU, down,
    // combine.
    assert_eq!(sites.len(), 6 + 5, "{sites:#?}");
    assert!(
        sites.iter().all(|l| l.starts_with("layer 4 n=")),
        "{sites:#?}"
    );
    assert_eq!(inst.sites(), sites);
    for rows in [1, 2] {
        crate::glm5next_mlp::forward::forward_moe(
            &s.gpu, &s.k, &s.cfg, &w, X, OUT, rows, &s.ws, true, STREAM,
        )
        .unwrap();
        let legacy = s.gpu.take();
        let ran = inst
            .run(&s.gpu, &s.k, &s.cfg, &w, &s.ws, X, OUT, rows, STREAM)
            .unwrap();
        assert!(ran, "{rows} rows run from the plan");
        assert_eq!(s.gpu.take(), legacy, "{rows} rows");
    }
    let run3 = inst.run(&s.gpu, &s.k, &s.cfg, &w, &s.ws, X, OUT, 3, STREAM);
    assert!(!run3.unwrap(), "verify 3 is not gated");
    assert!(
        s.gpu.take().is_empty(),
        "an uninstalled row count launches nothing"
    );
    assert!(
        inst.install(&s.gpu, 4, None, &s.k, &s.cfg, &w, &s.ws)
            .unwrap()
            .is_empty()
    );
    assert!(
        !inst
            .run(&s.gpu, &s.k, &s.cfg, &w, &s.ws, X, OUT, 1, STREAM)
            .unwrap()
    );
}

/// 2026-10-23: A lever the MoE rules do not model refuses installation, naming it, and installs
/// nothing.
#[test]
fn an_unmodelled_lever_refuses_installation() {
    let (c, rules, policy) = ranks();
    let mut s = site(0);
    s.k.levers.mlp.sweep_serial = false;
    let plans = [plan(&c, &rules, &policy, 0, Mode::Decode, 1)];
    let rp = RankPlans {
        circuit: &c[0],
        plans: &plans,
        policy: &policy,
    };
    let inst = InstalledMoe::default();
    let e = inst
        .install(&s.gpu, 4, Some(rp), &s.k, &s.cfg, &weights(4, false), &s.ws)
        .err()
        .unwrap();
    assert!(format!("{e:#}").contains("SWEEP_SERIAL=0"), "{e:#}");
    assert!(inst.sites().is_empty());
    // 2026-10-23: The dense segment replaces the shared expert's launches the rules name.
    let mut s = site(0);
    s.k.levers.dense.dense_seg = true;
    assert_eq!(
        unmodelled(&s.k),
        ["METRALE_GLM_DENSE_SEGMENT (the rules name the shared expert's separate launches)"]
    );
}

/// 2026-10-23: The boot's shapes for GLM TP=3 with one draft (`--num-drafts 1`): from the shapes
/// the GLM sites install (`plan_shapes`: `GATED` on a text layer, `DRAFT_GATED` on the draft
/// block) and the serve's verify widths, decode 1, verify 2 and draft 1, and every rank fuses
/// each. Three drafts (verify 2..=4) are refused naming verify 3 and the depth; the widths the
/// boot used to fuse whenever a drafter existed (2, 3, 4) do not all fuse on these rules.
#[test]
fn one_draft_fuses_only_the_covered_shapes_on_every_rank() {
    use metrale_model_layers::circuit_exec::rank_shapes::{
        rank_plan_shapes, speculative_verify_rows,
    };
    let (c, rules, policy) = ranks();
    let installed: Vec<(Mode, u64)> = GATED.iter().chain(&DRAFT_GATED).copied().collect();
    let shapes = rank_plan_shapes(&installed, &speculative_verify_rows(1)).unwrap();
    assert_eq!(
        shapes,
        [(Mode::Decode, 1), (Mode::Verify, 2), (Mode::Draft, 1)]
    );
    let avail = AvailableKernels::all_named_by(&rules);
    for (rank, circuit) in c.iter().enumerate() {
        for &(mode, rows) in &shapes {
            metrale_circuit::fuse(circuit, &rules, &avail, &policy, mode, rows)
                .unwrap_or_else(|e| panic!("rank {rank} {mode:?} {rows}: {e}"));
        }
    }
    let e = rank_plan_shapes(&installed, &speculative_verify_rows(3)).unwrap_err();
    assert_eq!(e.code(), metrale_error::ErrorCode::CircuitPlanUncovered);
    assert!(e.to_string().contains("verify at 3 rows"), "{e}");
    let fixed_widths = [2, 3, 4].iter().all(|&rows| {
        metrale_circuit::fuse(&c[0], &rules, &avail, &policy, Mode::Verify, rows).is_ok()
    });
    assert!(
        !fixed_widths,
        "the GLM rules cover every verify width up to 4"
    );
}
