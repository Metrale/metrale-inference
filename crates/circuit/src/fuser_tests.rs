// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Fuser selection, legality and errors on the toy circuit.
//!
//! Owner: metrale-circuit.
//! Invariants: none beyond the types.

use super::*;
use crate::test_toy::{circuit, fused, fused_with, groups, plan, policy, rules};

const ACT_DOWN: &str = r#"{ op = "silu_mul" }, { op = "linear", role = "down" }"#;
const UP_ACT: &str = r#"{ op = "linear", role = "gate_up" }, { op = "silu_mul" }"#;

fn rule_of_node(c: &Circuit, p: &FusionPlan, id: &str) -> String {
    let n = c.node(id).expect("node");
    p.groups
        .iter()
        .find(|g| g.nodes.contains(&n))
        .map(|g| g.rule.clone())
        .expect("covered")
}

#[test]
fn every_node_lands_in_exactly_one_group_and_fused_edges_stay_inside() {
    let c = circuit(2);
    let r = rules(&fused("a_act_down", ACT_DOWN, 100));
    let p = plan(&c, &r, &policy(), 1);
    let mut seen = vec![0usize; c.nodes.len()];
    for g in &p.groups {
        for &n in &g.nodes {
            seen[n] += 1;
        }
    }
    assert!(seen.iter().all(|&k| k == 1), "{seen:?}");
    let a = c.edge("l1.ffn.a").unwrap();
    let owner = p
        .groups
        .iter()
        .position(|g| g.rule == "a_act_down" && g.nodes.contains(&c.node("l1.ffn.act").unwrap()));
    assert_eq!(p.edge_states[a], Some(EdgeState::Fused(owner.unwrap())));
    let gu = c.edge("l1.ffn.gu").unwrap();
    assert_eq!(p.edge_states[gu], Some(EdgeState::Materialized));
    let logits = c.edge("head.logits").unwrap();
    assert_eq!(p.edge_states[logits], Some(EdgeState::Materialized));
    assert_eq!(
        p.launches(),
        p.groups.len() as u64,
        "one kernel per toy group"
    );
}

#[test]
fn overlapping_rules_resolve_by_priority_and_a_swap_changes_the_plan() {
    let c = circuit(1);
    let first = rules(&(fused("a_act_down", ACT_DOWN, 100) + &fused("b_up_act", UP_ACT, 90)));
    let p1 = plan(&c, &first, &policy(), 1);
    assert_eq!(rule_of_node(&c, &p1, "l0.ffn.act"), "a_act_down");
    assert_eq!(rule_of_node(&c, &p1, "l0.ffn.up"), "up");

    let swapped = rules(&(fused("a_act_down", ACT_DOWN, 90) + &fused("b_up_act", UP_ACT, 100)));
    let p2 = plan(&c, &swapped, &policy(), 1);
    assert_eq!(rule_of_node(&c, &p2, "l0.ffn.act"), "b_up_act");
    assert_eq!(rule_of_node(&c, &p2, "l0.ffn.down"), "down");
    assert_ne!(groups(&c, &p1), groups(&c, &p2));
    assert_ne!(p1.digest, p2.digest);

    // 2026-09-28: Equal priority: the lower id wins, and the rule's file position does not.
    let tie_a = rules(&(fused("a_act_down", ACT_DOWN, 100) + &fused("b_up_act", UP_ACT, 100)));
    let tie_b = rules(&(fused("b_up_act", UP_ACT, 100) + &fused("a_act_down", ACT_DOWN, 100)));
    let (pa, pb) = (
        plan(&c, &tie_a, &policy(), 1),
        plan(&c, &tie_b, &policy(), 1),
    );
    assert_eq!(rule_of_node(&c, &pa, "l0.ffn.act"), "a_act_down");
    assert_eq!(groups(&c, &pa), groups(&c, &pb));
    assert_eq!(pa.digest, pb.digest);
}

#[test]
fn an_absent_kernel_is_never_selected() {
    let c = circuit(1);
    let r = rules(&(fused("a_act_down", ACT_DOWN, 100) + &fused("b_up_act", UP_ACT, 90)));
    let mut avail = AvailableKernels::all_named_by(&r);
    avail.kernels.retain(|k| k.func != "a_act_down");
    let p = fuse(&c, &r, &avail, &policy(), Mode::Decode, 1).unwrap();
    assert!(p.groups.iter().all(|g| g.rule != "a_act_down"));
    assert_eq!(rule_of_node(&c, &p, "l0.ffn.act"), "b_up_act");

    avail.kernels.retain(|k| k.func != "norm");
    assert_eq!(
        fuse(&c, &r, &avail, &policy(), Mode::Decode, 1),
        Err(FuseError::Uncovered {
            node: "l0.ffn.norm".into(),
            op: "rms_norm".into(),
            mode: "decode",
            rows: 1
        })
    );
}

#[test]
fn a_missing_capability_skips_the_rule() {
    let c = circuit(1);
    let mut text = fused("a_act_down", ACT_DOWN, 100);
    text = text.replace(
        "priority = 100",
        "priority = 100\nrequires = [\"w8a8_decode\"]",
    );
    let r = rules(&text);
    let mut avail = AvailableKernels::all_named_by(&r);
    let with = fuse(&c, &r, &avail, &policy(), Mode::Decode, 1).unwrap();
    assert_eq!(rule_of_node(&c, &with, "l0.ffn.act"), "a_act_down");
    avail.caps.clear();
    let without = fuse(&c, &r, &avail, &policy(), Mode::Decode, 1).unwrap();
    assert_eq!(rule_of_node(&c, &without, "l0.ffn.act"), "act");
}

#[test]
fn a_differs_rule_needs_its_lever() {
    let c = circuit(1);
    let text = fused_with(
        "z_fast",
        ACT_DOWN,
        200,
        (1, 128),
        "numerics = \"differs\"\nlever = \"fast\"",
    );
    let r = rules(&text);
    let off = plan(&c, &r, &policy(), 1);
    assert!(off.groups.iter().all(|g| g.rule != "z_fast"));
    let mut on = policy();
    on.opt_in_levers.insert("fast".into());
    let p = plan(&c, &r, &on, 1);
    assert_eq!(rule_of_node(&c, &p, "l0.ffn.down"), "z_fast");
    on.opt_in_levers = ["other".to_string()].into();
    assert!(
        plan(&c, &r, &on, 1)
            .groups
            .iter()
            .all(|g| g.rule != "z_fast")
    );
}

#[test]
fn rows_outside_the_range_skip_the_rule() {
    let c = circuit(1);
    let r = rules(&fused_with(
        "a_act_down",
        ACT_DOWN,
        100,
        (2, 4),
        "numerics = \"reference\"",
    ));
    for (rows, want) in [(1, "act"), (2, "a_act_down"), (4, "a_act_down"), (5, "act")] {
        let p = plan(&c, &r, &policy(), rows);
        assert_eq!(rule_of_node(&c, &p, "l0.ffn.act"), want, "rows {rows}");
    }
}

#[test]
fn when_settings_select_and_an_unstated_setting_is_an_error() {
    let c = circuit(1);
    let text = fused("a_act_down", ACT_DOWN, 100)
        .replace("priority = 100", "priority = 100\nwhen = { kv = \"fp8\" }");
    let r = rules(&text);
    assert_eq!(
        rule_of_node(&c, &plan(&c, &r, &policy(), 1), "l0.ffn.act"),
        "act"
    );
    let mut fp8 = policy();
    fp8.settings.insert("kv".into(), "fp8".into());
    assert_eq!(
        rule_of_node(&c, &plan(&c, &r, &fp8, 1), "l0.ffn.act"),
        "a_act_down"
    );
    let avail = AvailableKernels::all_named_by(&r);
    assert_eq!(
        fuse(&c, &r, &avail, &Policy::default(), Mode::Decode, 1),
        Err(FuseError::PolicyMissing {
            rule: "a_act_down".into(),
            key: "kv".into()
        })
    );
}

#[test]
fn an_escaping_intermediate_needs_keep_and_then_stays_materialized() {
    let c = circuit(3);
    // 2026-09-28: Layer i's add output is layer i+1's input: its norm AND its add read it.
    let pat = r#"{ op = "residual_add" }, { op = "rms_norm" }"#;
    let refused = plan(&c, &rules(&fused("x_cross", pat, 100)), &policy(), 1);
    assert!(refused.groups.iter().all(|g| g.rule != "x_cross"));
    let keep = r#"{ op = "residual_add", keep = true }, { op = "rms_norm" }"#;
    let p = plan(&c, &rules(&fused("x_cross", keep, 100)), &policy(), 1);
    let crossing: Vec<_> = groups(&c, &p)
        .into_iter()
        .filter(|(r, _)| r == "x_cross")
        .collect();
    assert_eq!(
        crossing,
        [
            (
                "x_cross".to_string(),
                vec!["l0.ffn.add".to_string(), "l1.ffn.norm".to_string()]
            ),
            (
                "x_cross".to_string(),
                vec!["l1.ffn.add".to_string(), "l2.ffn.norm".to_string()]
            ),
        ]
    );
    let y = c.edge("l0.ffn.y").unwrap();
    assert_eq!(p.edge_states[y], Some(EdgeState::Materialized));
}

#[test]
fn a_chain_with_an_outside_node_between_members_is_refused() {
    let c = circuit(1);
    let pat = r#"{ op = "rms_norm" }, { op = "residual_add", sibling = true }"#;
    let p = plan(&c, &rules(&fused("x_cycle", pat, 100)), &policy(), 1);
    assert!(p.groups.iter().all(|g| g.rule != "x_cycle"));
}

#[test]
fn a_reader_disagreeing_with_the_stored_format_is_an_error() {
    let c = circuit(1);
    let text = fused(
        "w_up",
        r#"{ op = "linear", role = "gate_up", writes = "f32" }"#,
        50,
    ) + &fused("r_act", r#"{ op = "silu_mul", input = "bf16" }"#, 50);
    let r = rules(&text);
    let err = fuse(
        &c,
        &r,
        &AvailableKernels::all_named_by(&r),
        &policy(),
        Mode::Decode,
        1,
    );
    assert_eq!(
        err,
        Err(FuseError::FormatConflict {
            edge: "l0.ffn.gu".into(),
            stored: "f32".into(),
            rule: "r_act".into(),
            expected: "bf16".into()
        })
    );
    let ok = rules(&fused(
        "w_up",
        r#"{ op = "linear", role = "gate_up", writes = "f32" }"#,
        50,
    ));
    let p = plan(&c, &ok, &policy(), 1);
    assert_eq!(p.edge_formats[c.edge("l0.ffn.gu").unwrap()], Format::F32);
}

#[test]
fn zero_rows_and_an_empty_section_are_errors() {
    let c = circuit(1);
    let r = rules("");
    let avail = AvailableKernels::all_named_by(&r);
    assert_eq!(
        fuse(&c, &r, &avail, &policy(), Mode::Decode, 0),
        Err(FuseError::ZeroRows)
    );
    assert_eq!(
        fuse(&c, &r, &avail, &policy(), Mode::Draft, 1),
        Err(FuseError::EmptySection("draft"))
    );
}

#[test]
fn per_row_and_chunked_rules_count_their_launches() {
    let c = circuit(1);
    let per_row = fused("p_norm", r#"{ op = "rms_norm" }"#, 50)
        .replace("repeat = \"once\"", "repeat = \"per_row\"");
    let chunk = fused("c_add", r#"{ op = "residual_add" }"#, 50)
        .replace("repeat = \"once\"", "repeat = \"chunk64\"");
    let r = rules(&(per_row + &chunk));
    let p = plan(&c, &r, &policy(), 96);
    let base = plan(&c, &rules(""), &policy(), 96);
    assert_eq!(p.launches(), base.launches() - 2 + 96 + 2);
}

#[test]
fn the_digest_is_stable_and_moves_with_every_rule_and_policy_field() {
    let c = circuit(2);
    let base_text = fused("a_act_down", ACT_DOWN, 100);
    let r = rules(&base_text);
    let d0 = plan(&c, &r, &policy(), 1).digest;
    assert_eq!(d0, plan(&c, &rules(&base_text), &policy(), 1).digest);
    assert_eq!(d0.len(), 64);

    let edits = [
        ("priority = 100", "priority = 99"),
        ("rows = [1, 128]", "rows = [1, 127]"),
        ("emitter = \"a_act_down\"", "emitter = \"other\""),
        ("repeat = \"once\"", "repeat = \"per_row\""),
        ("func = \"a_act_down\"", "func = \"a_act_down2\""),
        (
            "modes = [\"decode\", \"multi_seq\", \"verify\"]",
            "modes = [\"decode\"]",
        ),
        (
            "numerics = \"reference\"",
            "numerics = \"bit_identical\"\nmicrotest = \"t\"",
        ),
    ];
    for (from, to) in edits {
        let edited = rules(&base_text.replacen(from, to, 1));
        let mut avail = AvailableKernels::all_named_by(&edited);
        avail
            .kernels
            .extend(AvailableKernels::all_named_by(&r).kernels);
        let p = fuse(&c, &edited, &avail, &policy(), Mode::Decode, 1).unwrap();
        assert_ne!(p.digest, d0, "{to}");
    }
    let cite_only = rules(&base_text.replacen("cite = \"test\"", "cite = \"elsewhere:1\"", 1));
    assert_eq!(plan(&c, &cite_only, &policy(), 1).digest, d0);

    let mut other = policy();
    other.settings.insert("kv".into(), "fp8".into());
    assert_ne!(plan(&c, &r, &other, 1).digest, d0);
    let mut lever = policy();
    lever.opt_in_levers.insert("x".into());
    assert_ne!(plan(&c, &r, &lever, 1).digest, d0);
    assert_ne!(plan(&c, &r, &policy(), 2).digest, d0);
}
