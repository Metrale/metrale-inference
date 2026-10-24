// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The GLM emitters over the served TP=3 plans, without a device: the plans are the
//! ones the hardware report counts, every routed MoE site of every rank compiles at decode n=1
//! and verify n=2 into the steps the plan names, the done list is exactly what compiles, and a
//! plan the emitters were not written for is refused.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use std::collections::BTreeSet;

use metrale_circuit::Mode;

use super::fixture::*;
use super::*;
use crate::glm5next_mlp::forward::plan_moe::{ActStep, RouterStep, ScaleForm, SweepEntries};

/// 2026-10-23: The fixture's plans launch what kernels/circuits/plans/hw/
/// glm-5.3-flash-nvfp4-tp3--gb10.md counts per rank (decode n=1 1269, verify n=2 1392): the
/// plans compiled here are the reported ones, not plans of another availability.
#[test]
fn the_fixture_plans_are_the_reported_ones() {
    let (c, rules, policy) = ranks();
    for rank in 0..3 {
        for (mode, rows, want) in [(Mode::Decode, 1, 1269), (Mode::Verify, 2, 1392)] {
            let p = plan(&c, &rules, &policy, rank, mode, rows);
            assert_eq!(p.launches(), want, "rank {rank} {mode:?} {rows}");
        }
    }
}

/// 2026-10-23: Every routed MoE site compiles on every rank at both runs. Its steps are the
/// served launches: the 64-unit expert slice of ranks 0 and 1 (704 columns) takes the `_k64`
/// down quantizer storing 768-wide rows, rank 2's 640 the plain one storing 640; one row sweeps its own
/// slots, two rows the union the gate/up group builds and zero `expert_out` in the SwiGLU; the
/// router is one GEMV per row at one row and the batched entry at two; the combine spreads a
/// row over 16 blocks; compact scales everywhere; the shared expert is the one legacy site.
#[test]
fn every_moe_site_compiles_into_the_served_steps() {
    for (world, rank) in worlds()
        .into_iter()
        .flat_map(|w| (0..w).map(move |r| (w, r)))
    {
        let (c, rules, policy) = ranks_at(world).unwrap();
        let layers = moe_layers(&c[rank]);
        assert_eq!(
            layers.len(),
            42,
            "rank {rank}: 45 layers, the first 3 dense"
        );
        for (mode, rows) in RUNS {
            let p = plan(&c, &rules, &policy, rank, mode, rows);
            for &l in &layers {
                let site = compile_moe(&c[rank], &p, l)
                    .unwrap_or_else(|e| panic!("rank {rank} {mode:?} layer {l}: {e:#}"));
                assert_eq!(
                    site.legacy,
                    ["moe [shared_gate_up,shared_clamp,shared_act,shared_down]"]
                );
                let s = site.steps;
                let slice = mlp_config_at(rank, world).moe_intermediate;
                let k64 = !slice.is_multiple_of(128);
                if world == served_world() {
                    assert_eq!(slice, [704, 704, 640][rank], "the served split");
                }
                assert_eq!(s.moe_inter, slice, "world {world} rank {rank}");
                let stored = if k64 {
                    slice.next_multiple_of(128)
                } else {
                    slice
                };
                assert_eq!(s.down.k_store, stored, "world {world} rank {rank}");
                assert_eq!(s.down.quant.func.ends_with("_k64"), k64, "rank {rank}");
                assert_eq!(s.gate_up.k_store, 4096);
                assert_eq!(s.blend.blocks_per_row, 16);
                assert_eq!(
                    (s.gate_up.scales, s.down.scales),
                    (ScaleForm::Compact, ScaleForm::Compact)
                );
                assert_eq!(s.topk.block, 288);
                if rows == 1 {
                    assert!(matches!(s.router, RouterStep::PerRow(_)));
                    assert_eq!(s.down.entries, SweepEntries::OwnSlots);
                    assert!(matches!(s.act, ActStep::Plain(_)));
                    assert!(s.gate_up.union.is_none());
                } else {
                    assert!(matches!(s.router, RouterStep::Batched(_)));
                    assert_eq!(s.down.entries, SweepEntries::Union);
                    assert!(matches!(s.act, ActStep::Zero(_)));
                    assert!(s.gate_up.union.is_some());
                }
                assert!(!s.memsets_expert_out());
            }
        }
    }
}

/// 2026-10-23: [`DONE`] is exactly the sites the emitters compile over both runs (a dense layer,
/// a DSA and a KDA layer: MoE, mHC and the KDA and DSA mixers' own groups) and the draft (the MTP
/// block's DSA mixer): every compiled group's locals are listed, and every listed entry is
/// compiled by some run.
#[test]
fn the_done_list_is_what_compiles() {
    let (c, rules, policy) = ranks();
    let mut seen = BTreeSet::new();
    for (mode, rows) in RUNS.into_iter().chain([(Mode::Draft, 1)]) {
        let p = plan(&c, &rules, &policy, 0, mode, rows);
        let layers = [Some(0), Some(3), Some(4), None];
        for g in PlanGroup::all(&c[0], &p).filter(|g| layers.contains(&g.layer())) {
            let moe = moe_site_group(&g) && !g.locals().iter().all(|l| l.starts_with("shared_"));
            let mixer = kda::is_site_group(&g) || dsa::is_site_group(&g);
            if moe || mhc::is_site_group(&g) || mixer {
                seen.insert((g.block().to_string(), g.locals().join(",")));
            }
        }
    }
    let done: BTreeSet<(String, String)> = DONE
        .iter()
        .map(|(b, l)| (b.to_string(), l.join(",")))
        .collect();
    assert_eq!(seen, done);
}

fn verify_plan(rank: usize) -> (Vec<metrale_circuit::Circuit>, metrale_circuit::FusionPlan) {
    let (c, rules, policy) = ranks();
    let p = plan(&c, &rules, &policy, rank, Mode::Verify, 2);
    (c, p)
}

fn site_group(c: &metrale_circuit::Circuit, p: &metrale_circuit::FusionPlan, local: &str) -> usize {
    PlanGroup::all(c, p)
        .position(|g| g.layer() == Some(4) && g.block() == "moe" && g.locals().contains(&local))
        .unwrap()
}

/// 2026-10-23: A plan naming a sweep entry the emitter does not know is refused, naming it,
/// rather than launched on a guessed shape.
#[test]
fn an_unknown_sweep_entry_is_refused() {
    let (c, mut p) = verify_plan(0);
    let i = site_group(&c[0], &p, "experts_down");
    p.groups[i].kernels[1].func = "w4a4_gemv_mx8_moe_union_sweep_k32_cs".into();
    let e = compile_moe(&c[0], &p, 4).err().unwrap();
    assert!(format!("{e:#}").contains("k32_cs"), "{e:#}");
}

/// 2026-10-23: A site whose plan has no combine group is refused at compile, not run short.
#[test]
fn a_site_without_its_combine_is_refused() {
    let (c, mut p) = verify_plan(0);
    let i = site_group(&c[0], &p, "blend");
    p.groups.remove(i);
    let e = compile_moe(&c[0], &p, 4).err().unwrap();
    assert!(format!("{e:#}").contains("no combine group"), "{e:#}");
}

/// 2026-10-23: Rank 2's plan compiled over rank 0's circuit (704-column slices) is refused: its
/// plain down quantizer cannot store a K that is not a multiple of 128. The emitters read the
/// widths from the circuit the plan is compiled over, not from the plan's kernel names alone.
#[test]
fn a_plan_over_another_ranks_dims_is_refused() {
    let (c, p) = verify_plan(2);
    assert!(compile_moe(&c[2], &p, 4).is_ok());
    let e = compile_moe(&c[0], &p, 4).err().unwrap();
    assert!(format!("{e:#}").contains("704"), "{e:#}");
}

/// 2026-10-23: The emitters refuse a group of another block: the draft head's MoE (`mtp_moe`)
/// does not run on the main layers' site code.
#[test]
fn an_emitter_refuses_another_block() {
    let (c, rules, policy) = ranks();
    let p = plan(&c, &rules, &policy, 0, Mode::Draft, 1);
    let reg = registry().unwrap();
    let g = PlanGroup::all(&c[0], &p)
        .find(|g| g.block() == "mtp_moe" && g.group.emitter == "moe_router")
        .expect("the draft head routes its MoE through moe_router");
    let e = reg.emit(&g, &mut GlmBlock::default()).err().unwrap();
    assert!(format!("{e:#}").contains("`moe` block"), "{e:#}");
}

/// 2026-10-23: `glm_w4a16` refuses a six-projection group naming another row count's pipelined
/// entry and a group outside the KDA block (the shared expert's), naming each.
#[test]
fn the_w4a16_emitter_refuses_another_entry_and_block() {
    let (c, rules, policy) = ranks();
    let mut p = plan(&c, &rules, &policy, 0, Mode::Decode, 1);
    let i = PlanGroup::all(&c[0], &p)
        .position(|g| g.layer() == Some(4) && g.locals() == ["q", "k", "v", "f_a", "b", "g_a"])
        .unwrap();
    p.groups[i].kernels[0].func = "w4a16_gemv_pipe_m2".into();
    let e = compile_kda(&c[0], &p, 4).err().unwrap();
    assert!(format!("{e:#}").contains("w4a16_gemv_pipe_m1"), "{e:#}");
    let p = plan(&c, &rules, &policy, 0, Mode::Decode, 1);
    let g = PlanGroup::all(&c[0], &p)
        .find(|g| g.layer() == Some(4) && g.block() == "moe" && g.group.emitter == "glm_w4a16")
        .expect("the shared expert is a glm_w4a16 group");
    let e = registry()
        .unwrap()
        .emit(&g, &mut GlmBlock::default())
        .err()
        .unwrap();
    assert!(format!("{e:#}").contains("`kda` block"), "{e:#}");
}
