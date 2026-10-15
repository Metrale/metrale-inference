// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: kernel identity and the per-launch distribution in the kernel
//! tree: variants split by launch configuration, the identity hashes against
//! values computed independently (`sha256sum`), exact quantiles on leaves and
//! parents, the JSON round trip, and refusal of edited identity fields.

mod common;

use common::{LAUNCH0, assert_close, assert_err, f};
use metprof::core::config::parse_toml;
use metprof::core::families::{classify, table_from_toml};
use metprof::core::model::{Kernel, Launch};
use metprof::core::tree::config::Resolved;
use metprof::core::tree::families::families_tree;
use metprof::core::tree::json::tree_from_json;
use metprof::core::tree::{Tree, span};
use serde_json::{Map, Value, json};

const TABLE: &str = r#"
schema = 1
coverage_min = 0.5
[api]
host_sync_regex = "^cudaStreamSynchronize"
[[rule]]
family = "moe_gu"
group = "moe"
class = "compute"
name_regex = "^grouped_gu_"
"#;

/// The launch configuration of `nsys_small.sql`'s gemm_b row.
const WIDE: Launch = Launch {
    block: (128, 1, 1),
    registers_per_thread: 64,
    static_shared_bytes: 0,
    dynamic_shared_bytes: 16384,
};

/// `printf '%s' 'grid=4,2,1;block=128,1,1;registers_per_thread=64;static_shared_bytes=0;dynamic_shared_bytes=16384' | sha256sum`
const WIDE_CONFIG_SHA: &str = "9e86340b2dee215e39ec453b3a6c9b9167eb6e2115334df0755bcd271933b482";
/// The same with `printf 'grouped_gu_a(int)\n%s'`.
const WIDE_KERNEL_ID: &str = "8adcccc8766ee47e9a40ade9bdcb597d8f448b459b814ff64d055de43d759773";

fn k(name: &str, start_us: i64, dur_us: i64, grid: (i64, i64, i64), launch: Launch) -> Kernel {
    Kernel {
        start: start_us * 1000,
        end: (start_us + dur_us) * 1000,
        stream: 1,
        correlation: start_us,
        graph_node: None,
        name: name.to_string(),
        demangled: format!("{name}(int)"),
        grid,
        launch,
    }
}

/// grouped_gu_a: 1, 2, 3 us (two launch configurations, two grids);
/// grouped_gu_b: 10 us; an unclassified kernel at two configurations.
fn window() -> Vec<Kernel> {
    vec![
        k("grouped_gu_a", 0, 1, (1, 1, 1), LAUNCH0),
        k("grouped_gu_a", 10, 3, (4, 2, 1), WIDE),
        k("grouped_gu_b", 20, 10, (1, 1, 1), LAUNCH0),
        k("grouped_gu_a", 40, 2, (1, 1, 1), LAUNCH0),
        k("stray", 50, 4, (1, 1, 1), LAUNCH0),
        k("stray", 60, 6, (4, 2, 1), WIDE),
    ]
}

fn tree() -> Tree {
    let doc = parse_toml(TABLE.as_bytes()).unwrap();
    let t = table_from_toml(&doc).unwrap();
    let ks = window();
    let labels = classify(&ks, &t);
    let all: Vec<usize> = (0..ks.len()).collect();
    families_tree(&ks, &labels, &all, &t, &Resolved::default(), Map::new()).unwrap()
}

fn json_tree() -> Value {
    tree().to_json().unwrap()
}

fn leaf(v: &Value, short: &str) -> Value {
    v["kernels"]["moe"]["kernels"]["moe_gu"]["kernels"][short].clone()
}

fn quantiles(v: &Value) -> [f64; 3] {
    [f(&v["p10_us"]), f(&v["median_us"]), f(&v["p90_us"])]
}

#[test]
fn variants_split_a_leaf_by_launch_configuration_and_add_up_to_it() {
    let v = json_tree();
    let a = leaf(&v, "grouped_gu_a");
    let vars = a["variants"].as_array().unwrap();
    assert_eq!(vars.len(), 2, "{a}");
    // First seen first; launches and times per identity.
    assert_eq!(vars[0]["grid"], json!([1, 1, 1]));
    assert_eq!(vars[0]["block"], json!([128, 1, 1]));
    assert_eq!(
        (&vars[0]["launches"], &vars[0]["total_time"]),
        (&json!(2), &json!(3.0))
    );
    assert_eq!(
        (&vars[1]["launches"], &vars[1]["total_time"]),
        (&json!(1), &json!(3.0))
    );
    assert_eq!(a["launches"], json!(3));
    assert_eq!(a["total_time"], json!(6.0));
    // Interior nodes carry no variants; a single-identity leaf has one.
    assert!(
        v["kernels"]["moe"]["kernels"]["moe_gu"]
            .get("variants")
            .is_none()
    );
    assert_eq!(
        leaf(&v, "grouped_gu_b")["variants"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let stray = &v["unattributed"]["stray(int)"];
    assert_eq!(stray["variants"].as_array().unwrap().len(), 2);
    assert_eq!(stray["variants"][1]["demangled"], json!("stray(int)"));
}

#[test]
fn identity_hashes_match_an_independent_sha256() {
    let v = json_tree();
    let wide = &leaf(&v, "grouped_gu_a")["variants"][1];
    assert_eq!(
        (
            &wide["registers_per_thread"],
            &wide["static_shared_bytes"],
            &wide["dynamic_shared_bytes"]
        ),
        (&json!(64), &json!(0), &json!(16384))
    );
    assert_eq!(wide["config_sha256"], json!(WIDE_CONFIG_SHA));
    assert_eq!(wide["kernel_id"], json!(WIDE_KERNEL_ID));
    // Same configuration, other name: same config hash, other kernel id.
    let stray = &v["unattributed"]["stray(int)"]["variants"][1];
    assert_eq!(stray["config_sha256"], json!(WIDE_CONFIG_SHA));
    assert_ne!(stray["kernel_id"], json!(WIDE_KERNEL_ID));
}

#[test]
fn quantiles_are_exact_on_leaves_and_span_children_on_parents() {
    let v = json_tree();
    // [1, 2, 3] us: h = 2p, linear between closest ranks.
    let a = quantiles(&leaf(&v, "grouped_gu_a"));
    for (x, want) in a.iter().zip([1.2, 2.0, 2.8]) {
        assert_close(*x, want);
    }
    assert_eq!(quantiles(&leaf(&v, "grouped_gu_b")), [10.0; 3]);
    // The family node pools its leaves' launches: [1, 2, 3, 10], h = 3p.
    let gu = quantiles(&v["kernels"]["moe"]["kernels"]["moe_gu"]);
    for (x, want) in gu.iter().zip([1.3, 2.5, 7.9]) {
        assert_close(*x, want);
    }
    assert_eq!(quantiles(&v["kernels"]["moe"]), gu);
    // Variant quantiles are over that variant's launches only: [1, 2].
    let first = quantiles(&leaf(&v, "grouped_gu_a")["variants"][0]);
    for (x, want) in first.iter().zip([1.1, 1.5, 1.9]) {
        assert_close(*x, want);
    }
    let stray = quantiles(&v["unattributed"]["stray(int)"]);
    for (x, want) in stray.iter().zip([4.2, 5.0, 5.8]) {
        assert_close(*x, want);
    }
}

#[test]
fn the_json_reads_back_to_the_same_json() {
    let v = json_tree();
    let back = tree_from_json(&v).unwrap();
    assert_eq!(back.to_json().unwrap(), v);
}

fn edited(edit: impl FnOnce(&mut Value)) -> Value {
    let mut v = json_tree();
    edit(&mut v);
    v
}

fn gu_a(v: &mut Value) -> &mut Value {
    &mut v["kernels"]["moe"]["kernels"]["moe_gu"]["kernels"]["grouped_gu_a"]
}

#[test]
fn edited_identity_and_quantile_fields_are_refused() {
    let cases: Vec<(Value, &str)> = vec![
        (
            edited(|v| gu_a(v)["variants"][1]["block"] = json!([64, 1, 1])),
            "config_sha256 is not the SHA-256",
        ),
        (
            edited(|v| gu_a(v)["variants"][1]["kernel_id"] = json!(WIDE_CONFIG_SHA)),
            "kernel_id is not the SHA-256",
        ),
        (
            edited(|v| {
                v["unattributed"]["stray(int)"]["variants"][0]["demangled"] = json!("other(int)")
            }),
            "kernel_id is not the SHA-256",
        ),
        (
            edited(|v| gu_a(v)["variants"][1]["launches"] = json!(2)),
            "do not add up",
        ),
        (
            edited(|v| {
                gu_a(v)["variants"].as_array_mut().unwrap().pop();
            }),
            "do not add up",
        ),
        (
            edited(|v| gu_a(v)["p10_us"] = json!(2.9)),
            "p10_us <= median_us <= p90_us",
        ),
        (
            edited(|v| gu_a(v)["median_us"] = Value::Null),
            "median_us must be a finite number",
        ),
        (
            edited(|v| gu_a(v)["p90_us"] = json!(3.5)),
            r"outside \[min_us, max_us\]",
        ),
        (
            edited(|v| {
                gu_a(v).as_object_mut().unwrap().remove("variants");
            }),
            "variants must be present exactly on a leaf",
        ),
        (
            edited(|v| v["kernels"]["moe"]["variants"] = json!([])),
            "variants must be present exactly on a leaf",
        ),
        (
            edited(|v| {
                let b = &mut v["kernels"]["moe"]["kernels"]["moe_gu"]["kernels"]["grouped_gu_b"];
                b["variants"][0]["p10_us"] = json!(9.0);
            }),
            "single variant's quantiles differ",
        ),
    ];
    for (v, pattern) in cases {
        assert_err(tree_from_json(&v), pattern);
    }
}

#[test]
fn an_unsealed_tree_is_not_written() {
    let mut t = tree_from_json(&json_tree()).unwrap();
    t.kernels["moe"].quantiles = None;
    assert_err(
        t.to_json(),
        "moe.* has 4 launch\\(es\\) but quantiles None; the tree was not sealed",
    );
    let mut t = tree_from_json(&json_tree()).unwrap();
    t.unattributed["stray(int)"].quantiles = None;
    assert_err(t.to_json(), "the tree was not sealed");
}

#[test]
fn the_window_is_the_first_start_to_the_last_end_of_the_segment() {
    let ks = window();
    assert_eq!(span(&ks, &[0, 1, 2, 3, 4, 5]), Some((0, 66_000)));
    assert_eq!(span(&ks, &[2, 3]), Some((20_000, 42_000)));
    assert_eq!(span(&ks, &[]), None);
}
