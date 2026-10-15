// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the family-table kernel tree with a model overlay, the `[tree]`
//! config refusals, and baseline/diff by node path including fused-away and
//! new nodes.

mod common;

use metprof::core::config::parse_toml;
use metprof::core::families::{FamilyTable, classify, table_from_toml};
use metprof::core::model::Kernel;
use metprof::core::tree::config::{Resolved, TreeConfig, tree_config};
use metprof::core::tree::diff::{apply_baseline, tree_diff};
use metprof::core::tree::families::families_tree;
use metprof::core::tree::json::tree_from_json;
use metprof::core::tree::{Source, Tree};
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
[[rule]]
family = "moe_down"
group = "moe"
class = "compute"
name_regex = "^grouped_down"
[[rule]]
family = "router_gemm"
group = "router"
class = "compute"
name_regex = "^router"
[[rule]]
family = "norm"
group = "misc"
class = "compute"
name_regex = "^rms_norm"
"#;

const TREE: &str = r#"
[tree]
coverage_min = 0.9
[tree.model.m]
fuses = { moe_gu = ["gate_gemm", "up_gemm"] }
[[tree.model.m.node]]
path = "layer/moe"
families = ["^moe_", "^router_"]
fuses = ["router", "experts"]
"#;

fn doc(extra: &str) -> toml::Table {
    parse_toml(format!("{TABLE}{extra}").as_bytes()).expect("toml")
}

fn table() -> FamilyTable {
    table_from_toml(&doc("")).expect("table")
}

fn config(extra: &str) -> TreeConfig {
    tree_config(&doc(extra)).expect("tree config")
}

fn k(name: &str, start_us: i64, dur_us: i64) -> Kernel {
    Kernel {
        start: start_us * 1000,
        end: (start_us + dur_us) * 1000,
        stream: 1,
        correlation: start_us,
        graph_node: None,
        name: name.to_string(),
        demangled: format!("{name}(int)"),
        grid: (1, 1, 1),
        launch: common::LAUNCH0,
    }
}

fn window() -> Vec<Kernel> {
    vec![
        k("rms_norm_a", 0, 2),
        k("router_gemm_x", 10, 3),
        k("grouped_gu_mx", 20, 40),
        k("grouped_down", 70, 20),
        k("dbg_fill", 100, 5),
        k("grouped_gu_mx", 110, 44),
    ]
}

fn tree(kernels: &[Kernel], overlay: &Resolved) -> Tree {
    let t = table();
    let labels = classify(kernels, &t);
    let all: Vec<usize> = (0..kernels.len()).collect();
    families_tree(kernels, &labels, &all, &t, overlay, Map::new()).expect("tree")
}

fn paths(t: &Tree) -> Vec<String> {
    t.paths().into_iter().map(|(p, _)| p).collect()
}

#[test]
fn the_overlay_places_families_and_states_fuses() {
    let cfg = config(TREE);
    let r = cfg.models["m"].resolve(&table()).unwrap();
    let t = tree(&window(), &r);
    assert_eq!(t.source, Source::Families);
    assert_eq!(
        paths(&t),
        [
            "misc",
            "misc/norm",
            "misc/norm/rms_norm_a",
            "layer",
            "layer/moe",
            "layer/moe/router_gemm",
            "layer/moe/router_gemm/router_gemm_x",
            "layer/moe/moe_gu",
            "layer/moe/moe_gu/grouped_gu_mx",
            "layer/moe/moe_down",
            "layer/moe/moe_down/grouped_down",
        ]
    );
    let get = |p: &str| {
        t.paths()
            .into_iter()
            .find(|(q, _)| q == p)
            .unwrap()
            .1
            .clone()
    };
    assert_eq!(get("layer/moe").fuses, vec!["router", "experts"]);
    assert!(get("layer").fuses.is_empty());
    let gu = get("layer/moe/moe_gu");
    assert_eq!(gu.fuses, vec!["gate_gemm", "up_gemm"]);
    // The fused GEMM is one leaf with its own time; the absorbed GEMMs get none.
    assert_eq!(gu.kernels.len(), 1);
    let s = gu.total();
    assert_eq!(
        (s.n, s.sum, s.min, s.max),
        (2, 84_000, Some(40_000), Some(44_000))
    );
    assert_eq!(get("layer/moe").total().sum, 107_000);
    assert_eq!(t.unattributed["dbg_fill(int)"].sum, 5_000);
    assert_eq!(t.total_ns(), 114_000);
    assert!(t.gate(0.95).is_ok());
    assert!(t.gate(0.99).is_err());
    assert_eq!(tree_from_json(&t.to_json().unwrap()).unwrap(), t);
}

#[test]
fn without_an_overlay_families_sit_under_their_groups() {
    let t = tree(&window(), &Resolved::default());
    assert_eq!(
        paths(&t)
            .iter()
            .filter(|p| !p.contains('/'))
            .collect::<Vec<_>>(),
        ["misc", "router", "moe"]
    );
    assert!(t.paths().iter().all(|(_, n)| n.fuses.is_empty()));
}

#[test]
fn a_segment_only_counts_its_launches() {
    let ks = window();
    let t = table();
    let labels = classify(&ks, &t);
    let tr = families_tree(&ks, &labels, &[5, 2], &t, &Resolved::default(), Map::new()).unwrap();
    assert_eq!(
        paths(&tr),
        ["moe", "moe/moe_gu", "moe/moe_gu/grouped_gu_mx"]
    );
    assert_eq!(tr.total_ns(), 84_000);
}

fn refused(extra: &str) -> String {
    match tree_config(&doc(extra)) {
        Err(e) => e.0,
        Ok(c) => match c
            .models
            .values()
            .map(|o| o.resolve(&table()))
            .find_map(Result::err)
        {
            Some(e) => e.0,
            None => panic!("accepted: {extra}"),
        },
    }
}

#[test]
fn bad_tree_config_is_refused() {
    let e = tree_config(&doc("")).unwrap_err().0;
    assert!(e.contains("[tree] is required"), "{e}");
    let e = refused("[tree]\n");
    assert!(e.contains("coverage_min"), "{e}");
    for bad in ["0", "1.5", "-0.1", "\"0.9\""] {
        let e = refused(&format!("[tree]\ncoverage_min = {bad}\n"));
        assert!(
            e.contains("coverage_min must be a number in (0, 1]"),
            "{bad}: {e}"
        );
    }
    let e = refused("[tree]\ncoverage_min = 0.9\ncoverage = 1\n");
    assert!(e.contains("unknown key"), "{e}");
    let e = refused(
        "[tree]\ncoverage_min = 0.9\n[tree.model.m]\n[[tree.model.m.node]]\npath = \"x\"\nfamilies = [\"^nope_\"]\n",
    );
    assert!(e.contains("matches no declared family"), "{e}");
    let e = refused(
        "[tree]\ncoverage_min = 0.9\n[tree.model.m]\n[[tree.model.m.node]]\npath = \"a\"\nfamilies = [\"^moe_\"]\n\
         [[tree.model.m.node]]\npath = \"b\"\nfamilies = [\"_gu$\"]\n",
    );
    assert!(e.contains("claimed by both"), "{e}");
    let e = refused("[tree]\ncoverage_min = 0.9\n[tree.model.m]\nfuses = { moe_up = [\"x\"] }\n");
    assert!(e.contains("not a declared family"), "{e}");
    let e = refused("[tree]\ncoverage_min = 0.9\n[tree.model.m]\nfuses = { moe_gu = [] }\n");
    assert!(e.contains("non-empty strings"), "{e}");
    let e = refused(
        "[tree]\ncoverage_min = 0.9\n[tree.model.m]\n[[tree.model.m.node]]\npath = \"a//b\"\nfamilies = [\"^moe_\"]\n",
    );
    assert!(e.contains("empty segment"), "{e}");
    let e = refused(
        "[tree]\ncoverage_min = 0.9\n[tree.model.m]\n[[tree.model.m.node]]\npath = \"a\"\nfamilies = [\"(\"]\n",
    );
    assert!(e.contains("not a valid regex"), "{e}");
    let e = refused("[tree]\ncoverage_min = 0.9\n[tree.circuit_names]\nnot_qualified = \"x\"\n");
    assert!(e.contains("module::func"), "{e}");
}

#[test]
fn the_shipped_config_has_a_valid_tree_section() {
    let shipped = parse_toml(include_bytes!("../config/families.toml")).unwrap();
    let t = table_from_toml(&shipped).unwrap();
    let cfg = tree_config(&shipped).unwrap();
    let glm = cfg.models["glm"].resolve(&t).unwrap();
    assert_eq!(glm.family_fuses["moe_grouped_gu"], ["gate_gemm", "up_gemm"]);
    assert_eq!(glm.claim["moe_grouped_gu"], ["moe"]);
    assert_eq!(glm.claim["router_gemm"], ["moe"]);
    assert_eq!(glm.claim["dsa_topk"], ["attention"]);
}

fn overlay_tree(ks: &[Kernel]) -> Tree {
    tree(ks, &config(TREE).models["m"].resolve(&table()).unwrap())
}

#[test]
fn a_baseline_marks_every_node_and_what_was_fused_away() {
    let old = overlay_tree(&window());
    // The newer build has no separate down GEMM (fused into a new kernel) and
    // a faster norm.
    let new_ks = vec![
        k("rms_norm_a", 0, 1),
        k("router_gemm_x", 10, 3),
        k("grouped_gu_mx", 20, 40),
        k("grouped_gu_down", 70, 15),
    ];
    let mut new = overlay_tree(&new_ks);
    apply_baseline(&mut new, &old).unwrap();
    let v = new.to_json().unwrap();
    assert_eq!(v["baseline_time"], json!(114.0));
    assert_eq!(v["delta"], json!(59.0 - 114.0));
    assert_eq!(
        v["fused_away"],
        json!([{"path": "layer/moe/moe_down", "baseline_time": 20.0}])
    );
    assert_eq!(v["new"], json!(["layer/moe/moe_gu/grouped_gu_down"]));
    let norm = &v["kernels"]["misc"]["kernels"]["norm"]["kernels"]["rms_norm_a"];
    assert_eq!(
        (norm["baseline_time"].clone(), norm["delta"].clone()),
        (json!(2.0), json!(-1.0))
    );
    let fresh =
        &v["kernels"]["layer"]["kernels"]["moe"]["kernels"]["moe_gu"]["kernels"]["grouped_gu_down"];
    assert!(fresh["baseline_time"].is_null() && fresh["delta"].is_null());
    // A tree with a baseline reads back exactly.
    assert_eq!(tree_from_json(&v).unwrap(), new);
    let d = tree_diff(&old, &new).unwrap();
    assert_eq!(
        d["fused_away"],
        json!([{"path": "layer/moe/moe_down", "fuses": [], "total_time": 20.0}])
    );
    assert_eq!(
        d["new"][0]["path"],
        json!("layer/moe/moe_gu/grouped_gu_down")
    );
    let rows = d["nodes"].as_array().unwrap();
    let deltas: Vec<f64> = rows
        .iter()
        .map(|r| r["delta"].as_f64().unwrap().abs())
        .collect();
    assert!(
        deltas.windows(2).all(|w| w[0] >= w[1]),
        "sorted by |delta|: {deltas:?}"
    );
    assert_eq!(rows[0]["path"], json!("layer"));
    let status = |p: &str| rows.iter().find(|r| r["path"] == json!(p)).unwrap()["status"].clone();
    assert_eq!(status("layer/moe/moe_down"), json!("fused_away"));
    assert_eq!(
        status("layer/moe/moe_down/grouped_down"),
        json!("fused_away")
    );
    assert_eq!(status("layer/moe/moe_gu/grouped_gu_down"), json!("new"));
    assert_eq!(status("misc/norm"), json!("both"));
    assert_eq!(d["delta"], json!(59.0 - 114.0));
}

#[test]
fn trees_of_different_sources_are_not_compared() {
    let a = overlay_tree(&window());
    let mut b: Value = a.to_json().unwrap();
    b["source"] = json!("circuit");
    let b = tree_from_json(&b).unwrap();
    assert!(
        tree_diff(&a, &b)
            .unwrap_err()
            .0
            .contains("different sources")
    );
    let mut c = a.clone();
    assert!(apply_baseline(&mut c, &b).is_err());
}
