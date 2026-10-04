// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The dense 27B circuit with LoRA adapters (tests/common/lora.rs), fused and
//! pipeline-checked at the shapes LoRA phase 1 serves: the plans equal the checked-in LoRA
//! goldens, each adapter runs by the rule of legacy's route for its projection, the fused arms
//! legacy leaves under an adapter are left (the `lora_active` route key), and the draft head
//! stays unadapted. Regenerate the goldens with
//! `cargo test -p metrale-circuit --test lora -- --ignored regenerate`.
//!
//! Owner: metrale-circuit tests (FEATURES workstream).
//! Invariants: none beyond the types.

mod common;

use std::collections::BTreeSet;

use common::lora::{SHAPES, adapted};
use metrale_circuit::{FusionPlan, Instance, Loaded, Mode, OpKind, fuse};

fn plan(inst: &Instance, loaded: &Loaded, (mode, rows): (Mode, u64)) -> FusionPlan {
    let avail = common::available(inst, &loaded.rules);
    fuse(
        &loaded.circuit,
        &loaded.rules,
        &avail,
        &inst.policy,
        mode,
        rows,
    )
    .unwrap_or_else(|e| panic!("{} n={rows}: {e}", mode.name()))
}

#[test]
fn the_lora_plans_equal_their_goldens() {
    let dir = common::plans_dir();
    let want = common::lora::plans();
    let mut problems = Vec::new();
    for (name, text) in &want {
        match std::fs::read_to_string(dir.join(name)) {
            Ok(on_disk) if on_disk == *text => {}
            Ok(_) => problems.push(format!("{name}: stale")),
            Err(_) => problems.push(format!("{name}: missing")),
        }
    }
    let expected: BTreeSet<&str> = want.iter().map(|(n, _)| n.as_str()).collect();
    for entry in std::fs::read_dir(dir.join("lora"))
        .expect("plans/lora")
        .flatten()
    {
        let name = format!("lora/{}", entry.file_name().to_string_lossy());
        if !expected.contains(name.as_str()) {
            problems.push(format!("{name}: no LoRA shape produces it"));
        }
    }
    assert!(
        problems.is_empty(),
        "LoRA goldens: regenerate with `cargo test -p metrale-circuit --test lora -- --ignored \
         regenerate` and review the diff:\n{}",
        problems.join("\n")
    );
}

#[test]
#[ignore = "writes kernels/circuits/plans/lora/; run explicitly to regenerate"]
fn regenerate() {
    let dir = common::plans_dir().join("lora");
    std::fs::create_dir_all(&dir).expect("plans/lora");
    for entry in std::fs::read_dir(&dir).expect("plans/lora").flatten() {
        std::fs::remove_file(entry.path()).expect("remove stale LoRA plan");
    }
    for (name, text) in common::lora::plans() {
        std::fs::write(common::plans_dir().join(name), text).expect("write LoRA plan");
    }
}

#[test]
fn each_adapter_runs_by_its_projections_legacy_route() {
    let (inst, loaded) = adapted();
    let c = &loaded.circuit;
    for shape in SHAPES.into_iter().filter(|(m, _)| *m != Mode::Draft) {
        let p = plan(&inst, &loaded, shape);
        let mut seen = 0;
        for g in &p.groups {
            let lora: Vec<&str> = g
                .nodes
                .iter()
                .filter(|&&n| matches!(c.nodes[n].op, OpKind::LoraShrink | OpKind::LoraExpand))
                .map(|&n| c.nodes[n].local.as_str())
                .collect();
            if lora.is_empty() {
                continue;
            }
            seen += lora.len();
            assert_eq!(
                lora.len(),
                g.nodes.len(),
                "{shape:?}: {} mixes LoRA and others",
                g.rule
            );
            let attention = ["q", "k", "v", "o"]
                .iter()
                .any(|r| lora[0] == format!("{r}_lora_a"));
            let want = if attention { "lora_bgmv" } else { "lora_pair" };
            assert_eq!(g.emitter, want, "{shape:?}: {lora:?} by {}", g.rule);
        }
        let nodes = c
            .nodes
            .iter()
            .filter(|n| matches!(n.op, OpKind::LoraShrink | OpKind::LoraExpand))
            .count();
        assert_eq!(
            seen, nodes,
            "{shape:?}: every adapter node is in a LoRA group"
        );
    }
}

#[test]
fn under_lora_active_no_arm_legacy_leaves_is_selected() {
    let (inst, loaded) = adapted();
    let off: BTreeSet<&str> = loaded
        .rules
        .iter()
        .filter(|r| r.when.get("lora_active").map(String::as_str) == Some("off"))
        .map(|r| r.id.as_str())
        .collect();
    assert!(
        off.contains("ffn_mmq16_gate_up") && off.contains("w4a16_gemv_qg_1row"),
        "the arms legacy leaves under an adapter are gated: {off:?}"
    );
    let c = &loaded.circuit;
    // 2026-10-03: Beyond phase 1's widths too (5 and 8 rows plan, though the engine refuses to
    // serve them yet): there the NVFP4 MMQ arms would otherwise match an adapted FFN.
    let wider = [(Mode::MultiSeq, 5), (Mode::MultiSeq, 8)];
    for shape in SHAPES.into_iter().chain(wider) {
        let p = plan(&inst, &loaded, shape);
        for g in &p.groups {
            assert!(
                !off.contains(g.rule.as_str()),
                "{shape:?}: {} selected",
                g.rule
            );
            // 2026-10-03: The MMQ arms write gate|up without the weight's global scale, so no
            // adapted projection may run on them (legacy turns them off under an adapter).
            let adapted = g.nodes.iter().any(|&n| {
                c.nodes[n].outputs.iter().any(|&e| {
                    c.edges[e]
                        .consumers
                        .iter()
                        .any(|&r| c.nodes[r].op == OpKind::LoraExpand)
                })
            });
            assert!(
                !adapted || g.kernels.iter().all(|k| k.module != "nvfp4_mmq"),
                "{shape:?}: {} runs an adapted projection on the MMQ arm",
                g.rule
            );
            let locals: BTreeSet<&str> =
                g.nodes.iter().map(|&n| c.nodes[n].local.as_str()).collect();
            assert!(
                !(locals.contains("q") && locals.contains("q_split")),
                "{shape:?}: {} fuses q with its split; legacy splits after the q fold",
                g.rule
            );
            assert!(
                !(locals.contains("act") && locals.contains("down")),
                "{shape:?}: {} fuses the activation into down; the down adapter reads it",
                g.rule
            );
        }
    }
}

#[test]
fn the_draft_head_is_not_adapted() {
    let (inst, loaded) = adapted();
    let p = plan(&inst, &loaded, (Mode::Draft, 1));
    for g in &p.groups {
        for &n in &g.nodes {
            assert!(
                !matches!(
                    loaded.circuit.nodes[n].op,
                    OpKind::LoraShrink | OpKind::LoraExpand
                ),
                "draft group {} runs an adapter",
                g.rule
            );
        }
    }
}
