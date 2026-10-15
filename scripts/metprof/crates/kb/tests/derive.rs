// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: a run and its kernel tree as KB records and edges (PLAN W1.1).
//! The fixtures are synthetic: a families tree three levels deep under `moe`
//! (one level past the sample cap) and a circuit tree.

mod common;

use std::collections::BTreeSet;

use common::run;
use metprof_kb::derive::{Graph, derive, run_key};
use metprof_kb::ids::{hex_id, node_id};
use metprof_kb::port::RunRecord;
use metprof_kb::schema::{Body, Kind, MAX_SAMPLE_DEPTH, NodeRef, Relation, Variant};
use serde_json::Value;

fn families() -> RunRecord {
    run(include_str!("fixtures/families_tree.json"))
}

/// Edges as (from kind, from key, relation, to kind, to key); sample keys
/// lose their run prefix so the expectation reads as tree paths.
fn edges(g: &Graph, run_hex: &str) -> BTreeSet<(String, String, String, String, String)> {
    let short = |n: &NodeRef| n.key.trim_start_matches(&format!("{run_hex}‖")).to_string();
    g.edges
        .iter()
        .map(|e| {
            (
                e.from.kind.label().to_string(),
                short(&e.from),
                e.rel.name().to_string(),
                e.to.kind.label().to_string(),
                short(&e.to),
            )
        })
        .collect()
}

fn hex(r: &RunRecord) -> String {
    hex_id(node_id(&NodeRef::new(
        Kind::ProfileRun,
        run_key(r).expect("key"),
    )))
}

#[test]
fn a_families_tree_gives_the_exact_record_and_edge_set() {
    let r = families();
    let g = derive(&r, MAX_SAMPLE_DEPTH).expect("derives");
    let h = hex(&r);
    let run = run_key(&r).expect("key");
    let e = |a: &str, b: &str, rel: &str, c: &str, d: &str| {
        (
            a.to_string(),
            b.to_string(),
            rel.to_string(),
            c.to_string(),
            d.to_string(),
        )
    };
    let s = "KernelSample";
    let mut want = BTreeSet::from([
        e(
            "ProfileRun",
            &run,
            "ON_HARDWARE",
            "HardwareClass",
            "class-a",
        ),
        e("ProfileRun", &run, "IN_CAMPAIGN", "Campaign", "campaign-a"),
        e("ProfileRun", &run, "ON_BOX", "Box", "box-a"),
        e(s, "attn", "PARENT_OF", s, "attn/flash_fwd"),
        e(
            s,
            "attn/flash_fwd",
            "PARENT_OF",
            s,
            "attn/flash_fwd/flash_fwd_k",
        ),
        e(s, "moe", "PARENT_OF", s, "moe/experts"),
        e(s, "moe/experts", "PARENT_OF", s, "moe/experts/grouped_gemm"),
        e(s, "attn/flash_fwd", "FUSES", "Kernel", "family:rope"),
        e(s, "moe/experts", "FUSES", "Kernel", "family:topk"),
        e(
            s,
            "attn/flash_fwd",
            "SAMPLE_OF",
            "Kernel",
            "family:flash_fwd",
        ),
        e(
            s,
            "attn/flash_fwd/flash_fwd_k",
            "SAMPLE_OF",
            "Kernel",
            "family:flash_fwd",
        ),
        e(
            s,
            "moe/experts/grouped_gemm",
            "SAMPLE_OF",
            "Kernel",
            "family:grouped_gemm",
        ),
        e(
            "KernelSymbol",
            "flash_fwd_k",
            "SYMBOL_OF",
            "Kernel",
            "family:flash_fwd",
        ),
        e(
            "KernelSymbol",
            "gg_kernel_a",
            "SYMBOL_OF",
            "Kernel",
            "family:grouped_gemm",
        ),
        e(
            "KernelSymbol",
            "gg_kernel_b",
            "SYMBOL_OF",
            "Kernel",
            "family:grouped_gemm",
        ),
        e(
            "KernelSymbol",
            "mystery_kernel(int)",
            "SYMBOL_OF",
            "Kernel",
            "family:unclassified",
        ),
    ]);
    for p in [
        "attn",
        "attn/flash_fwd",
        "attn/flash_fwd/flash_fwd_k",
        "moe",
        "moe/experts",
        "moe/experts/grouped_gemm",
    ] {
        want.insert(e("ProfileRun", &run, "HAS_SAMPLE", s, p));
    }
    assert_eq!(edges(&g, &h), want);

    let records: BTreeSet<(String, String, bool)> = g
        .records
        .iter()
        .map(|r| (r.kind.label().to_string(), r.key.clone(), r.is_stub()))
        .collect();
    let rec = |k: &str, key: &str, stub: bool| (k.to_string(), key.to_string(), stub);
    let mut want_records = BTreeSet::from([
        rec("ProfileRun", &run, false),
        rec("HardwareClass", "class-a", true),
        rec("Campaign", "campaign-a", true),
        rec("Box", "box-a", true),
        rec("Kernel", "family:rope", true),
        rec("Kernel", "family:topk", true),
        rec("Kernel", "family:flash_fwd", true),
        rec("Kernel", "family:grouped_gemm", true),
        rec("Kernel", "family:unclassified", true),
        rec("KernelSymbol", "flash_fwd_k", false),
        rec("KernelSymbol", "gg_kernel_a", false),
        rec("KernelSymbol", "gg_kernel_b", false),
        rec("KernelSymbol", "mystery_kernel(int)", false),
    ]);
    for w in want.iter().filter(|w| w.2 == "HAS_SAMPLE") {
        want_records.insert(rec(s, &format!("{h}‖{}", w.4), false));
    }
    assert_eq!(records, want_records);
    assert!(g.records.iter().all(|r| r.private), "run data is private");
    // Derivation is a function of the run.
    assert_eq!(derive(&r, MAX_SAMPLE_DEPTH).expect("again"), g);
}

fn sample_of(g: &Graph, path_suffix: &str) -> metprof_kb::schema::KernelSample {
    let r = g
        .records
        .iter()
        .find(|r| r.kind == Kind::KernelSample && r.key.ends_with(&format!("‖{path_suffix}")))
        .expect("sample present");
    match r.body.as_ref().expect("complete") {
        Body::KernelSample(k) => (**k).clone(),
        _ => panic!("not a sample"),
    }
}

#[test]
fn the_depth_cap_is_honoured() {
    let r = families();
    for cap in [0, 1, 2, 3] {
        let g = derive(&r, cap).expect("derives");
        let depths: Vec<u64> = g
            .records
            .iter()
            .filter_map(|r| match &r.body {
                Some(Body::KernelSample(k)) => Some(k.depth),
                _ => None,
            })
            .collect();
        assert_eq!(depths.iter().max().copied(), Some(cap as u64), "cap {cap}");
        let want = [2, 4, 6, 8][cap];
        assert_eq!(depths.len(), want, "cap {cap}");
        // The symbol table does not depend on the cap.
        let symbols = g
            .records
            .iter()
            .filter(|r| r.kind == Kind::KernelSymbol)
            .count();
        assert_eq!(symbols, 4, "cap {cap}");
    }
}

#[test]
fn totals_add_up() {
    let r = families();
    let g = derive(&r, 3).expect("derives");
    let h = hex(&r);
    // Top-level shares plus the unattributed share cover the run.
    let top: f64 = g
        .edges
        .iter()
        .filter(|e| e.rel == Relation::HAS_SAMPLE && !e.to.key.contains('/'))
        .map(|e| f64::from(e.weight))
        .sum();
    assert!((top + 10.0 / 100.0 - 1.0).abs() < 1e-6, "top {top}");
    // Every parent's total is the sum of its children's.
    for parent in g
        .edges
        .iter()
        .filter(|e| e.rel == Relation::PARENT_OF)
        .map(|e| &e.from)
    {
        let p = parent.key.trim_start_matches(&format!("{h}‖"));
        let kids: f64 = g
            .edges
            .iter()
            .filter(|e| e.rel == Relation::PARENT_OF && &e.from == parent)
            .map(|e| sample_of(&g, e.to.key.trim_start_matches(&format!("{h}‖"))).total_time)
            .sum();
        assert!((sample_of(&g, p).total_time - kids).abs() < 1e-9, "{p}");
    }
    assert_eq!(
        sample_of(&g, "moe/experts/grouped_gemm/gg_kernel_b").total_time,
        20.0
    );
}

#[test]
fn a_circuit_tree_links_its_plan_ops_and_kernels() {
    let r = run(include_str!("fixtures/circuit_tree.json"));
    let h = hex(&r);
    let capped = edges(&derive(&r, MAX_SAMPLE_DEPTH).expect("derives"), &h);
    // The plan is reached from below the cap; no group-level edge is.
    assert!(capped.iter().any(|e| e.2 == "PLANNED_BY" && e.4 == "d0d0"));
    assert!(!capped.iter().any(|e| e.2 == "FUSES" || e.2 == "SAMPLE_OF"));
    let deep = edges(&derive(&r, 4).expect("derives"), &h);
    let group = "arch0/decode/layer0/norm_matmul[0,1]";
    for op in ["op:norm", "op:matmul"] {
        let want = (
            "KernelSample".into(),
            group.into(),
            "FUSES".into(),
            "Kernel".into(),
            op.into(),
        );
        assert!(deep.contains(&want), "{op}");
    }
    let leaf = format!("{group}/mod_a::fused_norm_matmul");
    let want = (
        "KernelSample".into(),
        leaf,
        "SAMPLE_OF".into(),
        "Kernel".into(),
        "mod_a::fused_norm_matmul".into(),
    );
    assert!(deep.contains(&want));
    assert!(
        !deep.iter().any(|e| e.2 == "SYMBOL_OF"),
        "a circuit tree has no symbols"
    );
}

#[test]
fn a_leaf_sample_carries_its_launch_identities_and_quantiles() {
    let r = families();
    let g = derive(&r, 3).expect("derives");
    let tree = r.files[0].json.as_ref().expect("tree");
    let leaf = &tree["kernels"]["moe"]["kernels"]["experts"]["kernels"]["grouped_gemm"]["kernels"]
        ["gg_kernel_a"];
    let want: Vec<Variant> = serde_json::from_value(leaf["variants"].clone()).expect("variants");
    let s = sample_of(&g, "moe/experts/grouped_gemm/gg_kernel_a");
    assert_eq!(s.variants, want);
    let ids: BTreeSet<&str> = s.variants.iter().map(|v| v.kernel_id.as_str()).collect();
    assert_eq!(ids.len(), 2, "two launch configurations, two kernel ids");
    assert_eq!(
        (s.p10_us, s.median_us, s.p90_us),
        (Some(8.4), Some(10.0), Some(11.6))
    );
    let interior = sample_of(&g, "moe/experts");
    assert!(interior.variants.is_empty());
    assert_eq!(interior.median_us, Some(10.0));
}

#[test]
fn a_tree_with_misplaced_variants_or_a_missing_quantile_is_refused() {
    let ok = families();
    derive(&ok, 3).expect("the fixture derives");
    let leaf = "/kernels/attn/kernels/flash_fwd/kernels/flash_fwd_k";
    let interior = "/kernels/attn/kernels/flash_fwd";
    type Edit = fn(&mut Value, &str, &str);
    let edits: [(&str, Edit); 3] = [
        ("a leaf without variants", |t, leaf, _| {
            t.pointer_mut(leaf)
                .and_then(Value::as_object_mut)
                .expect("leaf")
                .remove("variants");
        }),
        ("an interior node with variants", |t, _, interior| {
            t.pointer_mut(interior)
                .and_then(Value::as_object_mut)
                .expect("node")
                .insert("variants".into(), Value::Array(vec![]));
        }),
        ("a node without its p90_us key", |t, _, interior| {
            t.pointer_mut(interior)
                .and_then(Value::as_object_mut)
                .expect("node")
                .remove("p90_us");
        }),
    ];
    for (what, edit) in edits {
        let mut r = ok.clone();
        edit(r.files[0].json.as_mut().expect("tree"), leaf, interior);
        assert!(derive(&r, 3).is_err(), "{what} must be refused");
    }
}
