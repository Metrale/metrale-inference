// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The event graph, the max-plus evaluation and the capture check, on the
//! hand-built captures of [`super::fixture`] (expected values computed by hand there).

use alloc::collections::BTreeMap;

use metrale_error::ErrorCode;

use super::capture::parse_capture;
use super::dag::build;
use super::fixture::{diamond, event_wait};
use super::graph::{CriticalGraph, EdgeKind, node_of};
use super::maxplus::{
    critical_chain, fit_latency, latest, longest, step_us, weights, work_in_context,
};

fn node(g: &CriticalGraph, c: &super::capture::Capture, name: &str, nth: usize) -> usize {
    (0..c.ops.len())
        .filter(|&i| c.ops[i].name() == name && c.ops[i].start() >= 0)
        .nth(nth)
        .unwrap_or(usize::MAX)
        .min(g.nodes.len())
}

fn kinds_into(g: &CriticalGraph, i: usize) -> Vec<EdgeKind> {
    let s = u32::try_from(2 * i).unwrap();
    let mut v: Vec<EdgeKind> = g
        .incoming(s)
        .iter()
        .chain(g.incoming(s + 1))
        .map(|&k| g.edges[k as usize].kind)
        .collect();
    v.sort();
    v
}

fn model_step(c: &super::capture::Capture) -> f64 {
    let g = build(c, 1_000, 3_000).unwrap();
    let w = weights(&g, &work_in_context(&g), Some(&fit_latency(&g)));
    step_us(&g, &longest(&g, &w))
}

#[test]
fn replay_and_model_reproduce_the_diamond_step() {
    let c = diamond(4, 11_000);
    let g = build(&c, 1_000, 3_000).unwrap();
    let work = work_in_context(&g);
    assert!((c.step_us() - 23.0).abs() < 1e-9);
    assert!((step_us(&g, &longest(&g, &weights(&g, &work, None))) - 23.0).abs() < 1e-9);
    assert!(
        (model_step(&c) - 23.0).abs() < 1e-9,
        "model {}",
        model_step(&c)
    );
    assert!(g.dropped.is_empty());
}

#[test]
fn edges_are_stream_order_pdl_fork_and_join() {
    let c = diamond(3, 11_000);
    let g = build(&c, 1_000, 3_000).unwrap();
    let work = work_in_context(&g);
    let a2 = node(&g, &c, "a2", 1);
    assert_eq!(
        kinds_into(&g, a2),
        vec![EdgeKind::Duration, EdgeKind::PdlStart, EdgeKind::PdlWait]
    );
    assert!(
        (work[a2] - 3.0).abs() < 1e-9,
        "a PDL successor's work is the part after its predecessor: {}",
        work[a2]
    );
    let b1 = node(&g, &c, "b1", 1);
    assert_eq!(
        kinds_into(&g, b1),
        vec![EdgeKind::Duration, EdgeKind::Join],
        "first on its stream: a fork"
    );
    let src = g
        .incoming(u32::try_from(2 * b1).unwrap())
        .iter()
        .map(|&k| node_of(g.edges[k as usize].src))
        .collect::<Vec<_>>();
    assert_eq!(
        c.ops[src[0]].name(),
        "step",
        "the fork is from the latest-finishing node, not a1 (still running)"
    );
    let a3 = node(&g, &c, "a3", 1);
    let src = node_of(g.edges[g.incoming(u32::try_from(2 * a3).unwrap())[0] as usize].src);
    assert_eq!(c.ops[src].name(), "b1");
}

#[test]
fn a_mid_stream_gap_is_a_residency_wait_not_a_join() {
    let c = diamond(3, 13_500);
    let g = build(&c, 1_000, 3_000).unwrap();
    let a2 = node(&g, &c, "a2", 1);
    assert_eq!(
        kinds_into(&g, a2),
        vec![EdgeKind::Duration, EdgeKind::Residency]
    );
    let c = diamond(3, 12_000);
    let g = build(&c, 1_000, 3_000).unwrap();
    let a2 = node(&g, &c, "a2", 1);
    assert_eq!(
        kinds_into(&g, a2),
        vec![EdgeKind::Duration, EdgeKind::GraphStream],
        "0.5 µs <= tau: explained by its stream"
    );
}

#[test]
fn the_critical_chain_and_the_subgradient() {
    let c = diamond(4, 11_000);
    let g = build(&c, 1_000, 3_000).unwrap();
    let lat = fit_latency(&g);
    let mut work = work_in_context(&g);
    let w = weights(&g, &work, Some(&lat));
    let at = longest(&g, &w);
    let chain = critical_chain(&g, &w, &at);
    let on: BTreeMap<&str, usize> = chain
        .iter()
        .filter(|&&k| g.edges[k as usize].kind == EdgeKind::Duration)
        .fold(BTreeMap::new(), |mut m, &k| {
            *m.entry(c.ops[node_of(g.edges[k as usize].dst)].name())
                .or_default() += 1;
            m
        });
    assert_eq!(on.get("b1"), Some(&4));
    assert_eq!(on.get("a1"), None);
    let late = latest(&g, &w, &at);
    let b1 = node(&g, &c, "b1", 1);
    assert!((late[2 * b1] - at[2 * b1]).abs() < 1e-9, "b1 has no slack");
    let a1 = node(&g, &c, "a1", 1);
    assert!(late[2 * a1].is_infinite(), "a1 feeds no anchor");
    let base = step_us(&g, &at);
    for i in 0..c.ops.len() {
        if c.ops[i].name() == "a1" && c.ops[i].start() >= 0 {
            work[i] = 0.0;
        }
    }
    assert!(
        (step_us(&g, &longest(&g, &weights(&g, &work, Some(&lat)))) - base).abs() < 1e-9,
        "subgradient 0 off the chain"
    );
    for i in 0..c.ops.len() {
        if c.ops[i].name() == "b1" && c.ops[i].start() >= 0 {
            work[i] -= 5.0;
        }
    }
    let shrunk = step_us(&g, &longest(&g, &weights(&g, &work, Some(&lat))));
    assert!(
        (base - shrunk - 5.0).abs() < 1e-9,
        "subgradient 1 on the chain: {base} -> {shrunk}"
    );
}

#[test]
fn dropping_the_event_wait_breaks_the_step() {
    let with = event_wait(4, true);
    assert!((with.step_us() - 26.0).abs() < 1e-9);
    let g = build(&with, 1_000, 3_000).unwrap();
    assert_eq!(
        g.edges.iter().filter(|e| e.kind == EdgeKind::Event).count(),
        2 * 5,
        "two waits per step"
    );
    assert!(
        (model_step(&with) - 26.0).abs() < 1e-6,
        "with the record/wait pair: {}",
        model_step(&with)
    );
    let without = event_wait(4, false);
    assert!(
        (without.step_us() - 26.0).abs() < 1e-9,
        "the measured step is the same"
    );
    let m = model_step(&without);
    assert!(
        m < 26.0 - 10.0,
        "without the wait the model cannot see why `m` waited: {m}"
    );
}

#[test]
fn a_capture_that_cannot_be_a_window_is_refused() {
    let c = diamond(2, 11_000);
    let mut v: serde_json::Value = serde_json::from_str(&serde_json::to_string(&serde_json::json!({
        "schema": c.schema, "source": c.source, "step_kernel": c.step_kernel, "anchors": c.anchors,
        "fields": c.fields, "groups": c.groups.iter().map(|g| serde_json::json!([g.0, g.1, g.2, g.3])).collect::<Vec<_>>(),
        "ops": c.ops.iter().map(|o| serde_json::json!([match o.0 { super::capture::OpKind::Kernel => "k", super::capture::OpKind::Memcpy => "m", super::capture::OpKind::Memset => "s" }, o.1, o.2, o.3, o.4, o.5, o.6, o.7, o.8, o.9, o.10, o.11, o.12])).collect::<Vec<_>>(),
        "api": [], "record": [], "sync": []
    })).unwrap()).unwrap();
    assert!(
        parse_capture(&v.to_string()).is_ok(),
        "the round trip itself parses"
    );
    v["anchors"] = serde_json::json!([0, 23000]);
    let e = parse_capture(&v.to_string()).unwrap_err();
    assert_eq!(e.code(), ErrorCode::CriticalCaptureInvalid);
    v["anchors"] = serde_json::json!(c.anchors);
    v["schema"] = serde_json::json!("critical-trace/v0");
    assert_eq!(
        parse_capture(&v.to_string()).unwrap_err().code(),
        ErrorCode::CriticalCaptureInvalid
    );
    v["schema"] = serde_json::json!("critical-trace/v1");
    v["fields"]["op"] = serde_json::json!(["kind"]);
    assert_eq!(
        parse_capture(&v.to_string()).unwrap_err().code(),
        ErrorCode::CriticalCaptureInvalid
    );
}
