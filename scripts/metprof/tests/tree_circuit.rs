// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the circuit-backed kernel tree over a synthetic plan and
//! synthetic graph steps: exact attribution, a reordered step, the sum
//! invariant, stable paths, the JSON round trip and refusals of edited JSON.

mod common;

use std::collections::BTreeMap;

use metprof::core::model::Kernel;
use metprof::core::tree::circuit::{Alignment, PlanGroup, PlanView, circuit_tree};
use metprof::core::tree::json::tree_from_json;
use metprof::core::tree::{Node, Tree};
use serde_json::{Map, Value, json};

fn group(
    index: u64,
    rule: &str,
    layer: &str,
    members: &[&str],
    ops: &[&str],
    funcs: &[&str],
    reps: u64,
) -> PlanGroup {
    PlanGroup {
        index,
        rule: rule.to_string(),
        layer: layer.to_string(),
        key: format!("{rule}[{}]", members.join(",")),
        members: members.iter().map(|s| s.to_string()).collect(),
        ops: ops.iter().map(|s| s.to_string()).collect(),
        kernels: funcs
            .iter()
            .map(|f| ("k".to_string(), f.to_string()))
            .collect(),
        reps,
    }
}

/// layer.0: norm+gemv fused (2 kernels), silu twice; layer.1: norm+gemv.
/// One step = rms_norm, gemv, silu, silu, rms_norm, gemv.
fn plan() -> PlanView {
    PlanView {
        recipe: "toy/recipe".into(),
        arch: "toy".into(),
        mode: "decode".into(),
        rows: 1,
        digest: "d1".into(),
        rules_digest: "r1".into(),
        groups: vec![
            group(
                1,
                "norm_gemv",
                "layer.0",
                &["norm", "qkv"],
                &["rms_norm", "gemv"],
                &["rms_norm", "gemv"],
                1,
            ),
            group(2, "act", "layer.0", &["act"], &["silu"], &["silu"], 2),
            group(
                4,
                "norm_gemv",
                "layer.1",
                &["norm", "qkv"],
                &["rms_norm", "gemv"],
                &["rms_norm", "gemv"],
                1,
            ),
        ],
    }
}

/// `k::gemv` is captured under a different short name.
fn names() -> BTreeMap<String, String> {
    BTreeMap::from([("k::gemv".to_string(), "gemv_sw".to_string())])
}

/// Graph steps of (name, duration us); each step one correlation id, back to
/// back, node ids from 1 (0 reads as no graph). `eager` adds launches outside
/// any graph before the steps.
fn stream(steps: &[&[(&str, i64)]], eager: &[(&str, i64)]) -> Vec<Kernel> {
    let mut t = 0;
    let mut out = Vec::new();
    let mut push = |name: &str, dur: i64, corr: i64, node: Option<i64>, t: &mut i64| {
        out.push(Kernel {
            start: *t,
            end: *t + dur * 1000,
            stream: 7,
            correlation: corr,
            graph_node: node,
            name: name.to_string(),
            demangled: format!("void {name}<int>(float*)"),
            grid: (1, 1, 1),
            launch: common::LAUNCH0,
        });
        *t += dur * 1000 + 500;
    };
    for (i, &(n, d)) in eager.iter().enumerate() {
        push(n, d, 900 + i as i64, None, &mut t);
    }
    for (s, step) in steps.iter().enumerate() {
        for (j, &(n, d)) in step.iter().enumerate() {
            push(n, d, s as i64 + 1, Some(j as i64 + 1), &mut t);
        }
    }
    out
}

const STEP_A: &[(&str, i64)] = &[
    ("rms_norm", 2),
    ("gemv_sw", 10),
    ("silu", 3),
    ("silu", 4),
    ("rms_norm", 5),
    ("gemv_sw", 20),
];
const STEP_B: &[(&str, i64)] = &[
    ("rms_norm", 1),
    ("gemv_sw", 11),
    ("silu", 6),
    ("silu", 7),
    ("rms_norm", 8),
    ("gemv_sw", 30),
];

fn build(ks: &[Kernel]) -> (Tree, Alignment) {
    circuit_tree(&plan(), &names(), ks, Map::new()).expect("circuit tree")
}

fn node<'t>(t: &'t Tree, path: &str) -> &'t Node {
    t.paths()
        .into_iter()
        .find(|(p, _)| p == path)
        .map(|(_, n)| n)
        .unwrap_or_else(|| panic!("no node {path}"))
}

const L0: &str = "toy/decode/layer.0/norm_gemv[norm,qkv]";

#[test]
fn each_launch_lands_on_its_planned_slot() {
    let (t, al) = build(&stream(&[STEP_A, STEP_B], &[("prefill_x", 50)]));
    // gemv of layer.0 is step position 1 in both steps: 10 + 11 us.
    let g = node(&t, &format!("{L0}/k::gemv")).total();
    assert_eq!(
        (g.n, g.sum, g.min, g.max),
        (2, 21_000, Some(10_000), Some(11_000))
    );
    let n0 = node(&t, &format!("{L0}/k::rms_norm")).total();
    assert_eq!((n0.n, n0.sum), (2, 3_000));
    // The two silu reps of one group share one leaf: 3 + 4 + 6 + 7.
    let s = node(&t, "toy/decode/layer.0/act[act]/k::silu").total();
    assert_eq!((s.n, s.sum), (4, 20_000));
    let g1 = node(&t, "toy/decode/layer.1/norm_gemv[norm,qkv]/k::gemv").total();
    assert_eq!((g1.n, g1.sum), (2, 50_000));
    // The eager launch is outside every step: unattributed by full name.
    assert_eq!(t.unattributed.len(), 1);
    assert_eq!(t.unattributed["void prefill_x<int>(float*)"].sum, 50_000);
    assert_eq!(
        (
            al.steps,
            al.launches_per_step,
            al.matched,
            al.missed,
            al.unmatched,
            al.outside_steps
        ),
        (2, 6, 12, 0, 0, 1)
    );
    // Join keys and declared fusion on the group node, none on a single-op group.
    let g0 = node(&t, L0);
    let j = g0.circuit.as_ref().expect("join key");
    assert_eq!(
        (j.group, j.rule.as_str(), j.plan_digest.as_str()),
        (1, "norm_gemv", "d1")
    );
    assert_eq!(g0.fuses, vec!["rms_norm", "gemv"]);
    assert_eq!(g0.members, vec!["norm", "qkv"]);
    assert!(node(&t, "toy/decode/layer.0/act[act]").fuses.is_empty());
    let all: i64 = stream(&[STEP_A, STEP_B], &[("prefill_x", 50)])
        .iter()
        .map(Kernel::dur)
        .sum();
    assert_eq!(t.total_ns(), all, "nothing dropped");
    assert!((t.coverage().unwrap() - 107_000.0 / 157_000.0).abs() < 1e-12);
}

#[test]
fn a_reordered_step_is_unattributed_and_fails_the_gate() {
    // Step B runs silu before the gemv of layer.0: the alignment cannot match
    // both in order, so one launch is unattributed and its slot is missed.
    let swapped: &[(&str, i64)] = &[
        ("rms_norm", 1),
        ("silu", 6),
        ("gemv_sw", 11),
        ("silu", 7),
        ("rms_norm", 8),
        ("gemv_sw", 30),
    ];
    let (t, al) = build(&stream(&[STEP_A, swapped], &[]));
    assert_eq!((al.matched, al.missed, al.unmatched), (11, 1, 1));
    assert_eq!(t.unattributed.len(), 1);
    let loose = t.unattributed.values().next().unwrap();
    assert_eq!(loose.n, 1);
    assert_eq!(t.attributed().n + loose.n, 12);
    let err = t.gate(0.99).unwrap_err().0;
    assert!(err.contains("below [tree] coverage_min 0.99"), "{err}");
    assert!(t.gate(0.5).is_ok());
}

#[test]
fn a_launch_the_plan_does_not_name_is_unattributed() {
    let extra: &[(&str, i64)] = &[
        ("rms_norm", 2),
        ("gemv_sw", 10),
        ("silu", 3),
        ("dbg_copy", 9),
        ("silu", 4),
        ("rms_norm", 5),
        ("gemv_sw", 20),
    ];
    let (t, al) = build(&stream(&[extra], &[]));
    assert_eq!((al.matched, al.missed, al.unmatched), (6, 0, 1));
    assert_eq!(t.unattributed["void dbg_copy<int>(float*)"].sum, 9_000);
    assert_eq!(
        node(&t, "toy/decode/layer.0/act[act]/k::silu").total().sum,
        7_000
    );
}

fn check_sums(prefix: &str, n: &Node) {
    let kids: i64 = n.kernels.values().map(|c| c.total().sum).sum();
    assert_eq!(n.total().sum, n.own.sum + kids, "{prefix}");
    if !n.kernels.is_empty() {
        assert_eq!(
            n.own.n, 0,
            "{prefix}: an interior node has no launches of its own"
        );
    }
    for (k, c) in &n.kernels {
        check_sums(&format!("{prefix}/{k}"), c);
    }
}

fn check_json_sums(place: &str, v: &Value) {
    let ns = |x: &Value| (x.as_f64().unwrap() * 1000.0).round() as i64;
    let kids = v["kernels"].as_object().unwrap();
    let sum: i64 = kids.values().map(|c| ns(&c["total_time"])).sum();
    assert_eq!(ns(&v["total_time"]), ns(&v["self_time"]) + sum, "{place}");
    let launches: u64 = kids.values().map(|c| c["launches"].as_u64().unwrap()).sum();
    if !kids.is_empty() {
        assert_eq!(v["launches"].as_u64().unwrap(), launches, "{place}");
    }
    for (k, c) in kids {
        check_json_sums(&format!("{place}/{k}"), c);
    }
}

#[test]
fn every_node_is_its_own_time_plus_its_children() {
    let (t, _) = build(&stream(&[STEP_A, STEP_B], &[("x", 1)]));
    for (k, n) in &t.kernels {
        check_sums(k, n);
    }
    let v = t.to_json().unwrap();
    let mut root_sum = 0;
    for (k, n) in v["kernels"].as_object().unwrap() {
        check_json_sums(k, n);
        root_sum += (n["total_time"].as_f64().unwrap() * 1000.0).round() as i64;
    }
    let loose: i64 = v["unattributed"]
        .as_object()
        .unwrap()
        .values()
        .map(|l| (l["total_time"].as_f64().unwrap() * 1000.0).round() as i64)
        .sum();
    assert_eq!(
        (v["total_time"].as_f64().unwrap() * 1000.0).round() as i64,
        root_sum + loose
    );
}

#[test]
fn paths_are_the_plan_whatever_the_capture() {
    // A second capture where layer.1 never ran and the durations differ:
    // the same paths in the same order, with a zero-launch leaf kept.
    let short: &[(&str, i64)] = &[("rms_norm", 9), ("gemv_sw", 9), ("silu", 9), ("silu", 9)];
    let (a, _) = build(&stream(&[STEP_A], &[]));
    let (b, al) = build(&stream(&[short, short], &[]));
    let pa: Vec<String> = a.paths().into_iter().map(|(p, _)| p).collect();
    let pb: Vec<String> = b.paths().into_iter().map(|(p, _)| p).collect();
    assert_eq!(pa, pb);
    assert_eq!(
        node(&b, "toy/decode/layer.1/norm_gemv[norm,qkv]/k::gemv")
            .total()
            .n,
        0
    );
    assert_eq!(al.missed, 4);
    assert_eq!(b.coverage().unwrap(), 1.0);
    // A zero-launch leaf serializes with null min/max/mean and reads back.
    let v = b.to_json().unwrap();
    assert!(v["kernels"]["toy"]["kernels"]["decode"]["kernels"]["layer.1"]["kernels"]["norm_gemv[norm,qkv]"]["mean_us"].is_null());
    assert_eq!(tree_from_json(&v).unwrap(), b);
}

#[test]
fn eager_segments_are_refused() {
    let ks = stream(&[], &[("rms_norm", 2), ("gemv_sw", 3)]);
    let err = circuit_tree(&plan(), &names(), &ks, Map::new())
        .unwrap_err()
        .0;
    assert!(err.contains("no graph-node kernels"), "{err}");
}

#[test]
fn json_round_trips_exactly() {
    let (mut t, _) = build(&stream(&[STEP_A, STEP_B], &[("prefill_x", 50)]));
    t.provenance.insert("capture_sha256".into(), json!("ab"));
    let v = t.to_json().unwrap();
    let text = serde_json::to_string_pretty(&v).unwrap();
    let back = tree_from_json(&serde_json::from_str(&text).unwrap()).unwrap();
    assert_eq!(back, t);
    assert_eq!(back.to_json().unwrap(), v);
}

fn tampered(edit: impl Fn(&mut Value)) -> String {
    let (t, _) = build(&stream(&[STEP_A, STEP_B], &[("prefill_x", 50)]));
    let mut v = t.to_json().unwrap();
    edit(&mut v);
    tree_from_json(&v).unwrap_err().0
}

#[test]
fn edited_json_is_refused() {
    fn leaf(v: &mut Value) -> &mut Value {
        &mut v["kernels"]["toy"]["kernels"]["decode"]["kernels"]["layer.0"]["kernels"]["norm_gemv[norm,qkv]"]
            ["kernels"]["k::gemv"]
    }
    let e = tampered(|v| leaf(v)["total_time"] = json!(99.0));
    assert!(!e.is_empty());
    let e = tampered(|v| leaf(v)["launches"] = json!(3));
    assert!(!e.is_empty());
    let e = tampered(|v| v["coverage"] = json!(1.0));
    assert!(e.contains("coverage"), "{e}");
    let e = tampered(|v| v["schema"] = json!("metprof.kernel_tree/2"));
    assert!(e.contains("schema"), "{e}");
    let e = tampered(|v| v["time_unit"] = json!("ms"));
    assert!(!e.is_empty());
    let e = tampered(|v| {
        v["kernels"]["toy"]
            .as_object_mut()
            .unwrap()
            .insert("extra".into(), json!(1));
    });
    assert!(e.contains("unknown key"), "{e}");
    let e =
        tampered(|v| v["unattributed"]["void prefill_x<int>(float*)"]["total_time"] = json!(1.0));
    assert!(!e.is_empty());
}
