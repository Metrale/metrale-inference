// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the family table: validation (no defaults, no unknown keys),
//! first-match classification, call-site rules, the coverage gate, and the
//! ordering invariants of the shipped table.

mod common;

use common::{FAMILIES_TOML, SHIPPED_FAMILIES, assert_close, assert_err};
use metprof::core::config::parse_toml;
use metprof::core::families::{FamilyTable, classify, coverage_gate, table_from_toml};
use metprof::core::model::Kernel;
use metprof::core::union::Class;
use toml::{Table, Value};

fn kern(name: &str, start: i64, stream: i64) -> Kernel {
    Kernel {
        start,
        end: start + 1,
        stream,
        correlation: 0,
        graph_node: None,
        name: name.to_string(),
        demangled: name.to_string(),
        grid: (1, 1, 1),
        launch: common::LAUNCH0,
    }
}

fn doc() -> Table {
    parse_toml(FAMILIES_TOML.as_bytes()).unwrap()
}

fn rules(d: &mut Table) -> &mut Vec<Value> {
    d.get_mut("rule").and_then(Value::as_array_mut).unwrap()
}

fn rule(d: &mut Table, i: usize) -> &mut Table {
    rules(d)[i].as_table_mut().unwrap()
}

fn refuse(d: &Table, pattern: &str) {
    assert_err(table_from_toml(d), pattern);
}

fn rule_of(family: &str, group: &str, class: &str, re: &str, next: Option<&[&str]>) -> Value {
    let mut t = Table::new();
    t.insert("family".into(), family.into());
    t.insert("group".into(), group.into());
    t.insert("class".into(), class.into());
    t.insert("name_regex".into(), re.into());
    if let Some(n) = next {
        t.insert(
            "next_on_stream_family".into(),
            Value::Array(n.iter().map(|s| Value::from(*s)).collect()),
        );
    }
    Value::Table(t)
}

fn shipped() -> FamilyTable {
    table_from_toml(&parse_toml(SHIPPED_FAMILIES).unwrap()).unwrap()
}

#[test]
fn valid_table_loads() {
    let t = table_from_toml(&doc()).unwrap();
    assert_eq!(t.families(), vec!["comm", "gemm", "norm"]);
    assert_eq!(t.class("comm").unwrap(), Class::Comm);
}

#[test]
fn missing_top_level_key() {
    let mut d = doc();
    d.remove("coverage_min");
    refuse(&d, "missing required key.*coverage_min");
}

#[test]
fn missing_rule_key() {
    let mut d = doc();
    rule(&mut d, 1).remove("class");
    refuse(&d, "#2: missing required key.*class");
}

#[test]
fn unknown_rule_key_is_not_ignored() {
    let mut d = doc();
    rule(&mut d, 0).insert("name_regx".into(), "x".into());
    refuse(&d, "unknown key.*name_regx");
}

#[test]
fn bad_class() {
    let mut d = doc();
    rule(&mut d, 0).insert("class".into(), "memcpy".into());
    refuse(&d, "class must be one of");
}

#[test]
fn bad_regex() {
    let mut d = doc();
    rule(&mut d, 0).insert("name_regex".into(), "nccl(".into());
    refuse(&d, "not a valid regex");
}

#[test]
fn family_declared_twice_with_different_class() {
    let mut d = doc();
    let mut dup = rule(&mut d, 1).clone();
    dup.insert("class".into(), "comm".into());
    dup.insert("name_regex".into(), "^other".into());
    rules(&mut d).push(Value::Table(dup));
    refuse(&d, "family 'gemm' is declared");
}

#[test]
fn coverage_min_range_and_type() {
    for bad in [
        Value::from(0),
        Value::from(1.5),
        Value::from(true),
        Value::from("0.9"),
    ] {
        let mut d = doc();
        d.insert("coverage_min".into(), bad);
        refuse(&d, "coverage_min");
    }
}

#[test]
fn schema_and_empty_rules() {
    let mut d = doc();
    d.insert("schema".into(), 2.into());
    refuse(&d, "schema");
    d.insert("schema".into(), 1.into());
    d.insert("rule".into(), Value::Array(Vec::new()));
    refuse(&d, "at least one");
}

#[test]
fn call_site_references_are_validated() {
    for (reference, msg) in [
        ("ghost", "undeclared family 'ghost'"),
        ("gemm", "its own family 'gemm'"),
    ] {
        let mut d = doc();
        rule(&mut d, 1).insert(
            "next_on_stream_family".into(),
            Value::Array(vec![reference.into()]),
        );
        refuse(&d, msg);
    }
    let mut d = doc();
    rule(&mut d, 1).insert("next_on_stream_family".into(), "norm".into());
    refuse(&d, "non-empty list");
}

#[test]
fn call_site_cycle_is_refused() {
    let mut d = doc();
    rule(&mut d, 1).insert(
        "next_on_stream_family".into(),
        Value::Array(vec!["norm".into()]),
    );
    rule(&mut d, 2).insert(
        "next_on_stream_family".into(),
        Value::Array(vec!["gemm".into()]),
    );
    refuse(&d, "cycle: gemm -> norm -> gemm");
}

#[test]
fn missing_api_key() {
    let mut d = doc();
    d.insert("api".into(), Value::Table(Table::new()));
    refuse(&d, "host_sync_regex");
}

#[test]
fn first_match_wins() {
    let mut d = doc();
    rules(&mut d).insert(
        1,
        rule_of("gemm_small", "dense", "compute", "^gemm_small", None),
    );
    let t = table_from_toml(&d).unwrap();
    let got = classify(&[kern("gemm_small_x", 0, 1), kern("gemm_big", 0, 1)], &t);
    assert_eq!(got, vec![Some("gemm_small"), Some("gemm")]);
}

#[test]
fn unmatched_is_none() {
    let t = table_from_toml(&doc()).unwrap();
    assert_eq!(classify(&[kern("mystery", 0, 1)], &t), vec![None]);
}

fn router_table() -> FamilyTable {
    let mut d = doc();
    let head = [
        rule_of("topk", "router", "compute", "^topk$", None),
        rule_of(
            "router_reduce",
            "router",
            "compute",
            "^reduce$",
            Some(&["topk"]),
        ),
        rule_of(
            "router_gemm",
            "router",
            "compute",
            "^gemm",
            Some(&["topk", "router_reduce"]),
        ),
    ];
    for (i, r) in head.into_iter().enumerate() {
        rules(&mut d).insert(i, r);
    }
    rules(&mut d).push(rule_of("reduce", "dense", "compute", "^reduce$", None));
    table_from_toml(&d).unwrap()
}

fn names(got: Vec<Option<&str>>) -> Vec<&str> {
    got.into_iter().map(|f| f.unwrap_or("-")).collect()
}

#[test]
fn call_site_rule_needs_the_next_kernel_on_the_same_stream() {
    let ks = [
        kern("gemm", 0, 1),
        kern("norm", 1, 2),
        kern("topk", 2, 1), // router call
        kern("gemm", 3, 1),
        kern("topk", 4, 2), // topk on another stream
        kern("gemm", 5, 1), // last on its stream
    ];
    let t = router_table();
    assert_eq!(
        names(classify(&ks, &t)),
        vec!["router_gemm", "norm", "topk", "gemm", "topk", "gemm"]
    );
}

#[test]
fn call_site_chain_follows_the_split_k_reduce_but_stops_at_the_table() {
    let ks = [
        kern("gemm", 0, 1),
        kern("gemm", 1, 1),
        kern("reduce", 2, 1),
        kern("topk", 3, 1),
        kern("reduce", 4, 1),
        kern("norm_x", 5, 1),
    ];
    let t = router_table();
    // the first gemm precedes a router GEMM, not a topk/reduce: it stays dense
    assert_eq!(
        names(classify(&ks, &t)),
        vec![
            "gemm",
            "router_gemm",
            "router_reduce",
            "topk",
            "reduce",
            "norm"
        ]
    );
}

#[test]
fn order_of_input_does_not_matter() {
    let ks = [
        kern("gemm", 1, 1),
        kern("reduce", 2, 1),
        kern("topk", 3, 1),
        kern("gemm", 0, 1),
    ];
    let t = router_table();
    assert_eq!(
        names(classify(&ks, &t)),
        vec!["router_gemm", "router_reduce", "topk", "gemm"]
    );
}

#[test]
fn coverage_gate_threshold() {
    let t = table_from_toml(&doc()).unwrap();
    assert_close(coverage_gate(99.0, 100.0, &t).unwrap(), 0.99);
    assert_err(coverage_gate(98.9, 100.0, &t), "below coverage_min");
    assert!(coverage_gate(0.0, 0.0, &t).is_err());
}

#[test]
fn shipped_table_ordering_invariants() {
    let t = shipped();
    let cases = [
        ("glm5next_hc_pre_norm_tok4", "mhc_pre_norm_tok4"),
        ("glm5next_hc_pre_norm", "mhc_pre_norm"),
        ("glm5next_hc_post_rank_sum", "mhc_post_rank_sum"),
        ("w4a4_gemv_mx16_moe_union_sweep_k64", "moe_union_sweep"),
        ("w4a4_gemv_mx8_moe_slots_sweep", "moe_slots_sweep"),
        ("w4a4_gemv_mx32", "dense_gemv"),
        ("w4a4_grouped_gu_m32", "moe_grouped_gu"),
        ("w4a4_grouped_mx_m128_k64", "moe_grouped"),
        ("ncclDevKernel_SendRecv", "nccl_sendrecv"),
        ("ncclDevKernel_AllReduce_Sum_bf16_RING_LL", "nccl_allreduce"),
        ("dense_gemv_bf16_fp32out", "dense_gemv"),
    ];
    let ks: Vec<Kernel> = cases
        .iter()
        .enumerate()
        .map(|(i, (n, _))| kern(n, i as i64, i as i64))
        .collect();
    let want: Vec<&str> = cases.iter().map(|(_, f)| *f).collect();
    assert_eq!(names(classify(&ks, &t)), want);
    assert_eq!(t.class("nccl_allreduce").unwrap(), Class::Comm);
}

#[test]
fn shipped_table_router_call_site() {
    let t = shipped();
    let ns = [
        "nvjet_tss_a",
        "nvjet_tss_b",
        "splitKreduce_kernel",
        "glm5next_router_topk_rank",
        "dense_gemv_bf16_fp32out",
        "glm5next_router_topk_rank",
        "nvjet_tss_c",
        "splitKreduce_kernel",
        "rms_norm_x",
    ];
    let ks: Vec<Kernel> = ns
        .iter()
        .enumerate()
        .map(|(i, n)| kern(n, i as i64, 1))
        .collect();
    assert_eq!(
        names(classify(&ks, &t)),
        vec![
            "nvjet",
            "router_gemm",
            "router_splitk_reduce",
            "router_topk",
            "router_gemm",
            "router_topk",
            "nvjet",
            "vendor_gemm_other",
            "norm"
        ]
    );
}
