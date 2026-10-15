// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: capture A vs capture B: per-unit normalization, only-in lists,
//! kernel rows by (name, grid), and the `diff` command end to end.

mod common;

use common::{Capture, FAMILIES_TOML, Mem, assert_close, f};
use metprof::core::diff::{Side, capture_diff};
use metprof::core::model::Kernel;
use serde_json::{Map, Value, json};

const MS: f64 = 1_000_000.0;

fn k(start_ms: f64, dur_ms: f64, name: &str, grid: (i64, i64, i64)) -> Kernel {
    Kernel {
        start: (start_ms * MS).round() as i64,
        end: ((start_ms + dur_ms) * MS).round() as i64,
        stream: 1,
        correlation: 0,
        graph_node: None,
        name: name.to_string(),
        demangled: name.to_string(),
        grid,
        launch: common::LAUNCH0,
    }
}

const G1: (i64, i64, i64) = (1, 1, 1);

/// A: two requests, each one 4 ms gemm (grid 1) and one 2 ms norm.
/// B: one request; the gemm runs at another grid and a new kernel appears.
fn kernels() -> (Vec<Kernel>, Vec<Kernel>) {
    let a = vec![
        k(0.0, 4.0, "gemm", G1),
        k(4.0, 2.0, "norm", G1),
        k(100.0, 4.0, "gemm", G1),
        k(104.0, 2.0, "norm", G1),
    ];
    let b = vec![
        k(0.0, 3.0, "gemm", (2, 1, 1)),
        k(3.0, 2.0, "norm", G1),
        k(5.0, 5.0, "fused", G1),
    ];
    (a, b)
}

fn diff(top: i64) -> Map<String, Value> {
    let (a, b) = kernels();
    let sa = Side {
        kernels: &a,
        labels: vec!["dense", "misc", "dense", "misc"],
        units: 2,
    };
    let sb = Side {
        kernels: &b,
        labels: vec!["dense", "misc", "fused"],
        units: 1,
    };
    capture_diff(&sa, &sb, top).unwrap()
}

fn family<'a>(r: &'a Map<String, Value>, name: &str) -> &'a Value {
    r["families"]
        .as_array()
        .unwrap()
        .iter()
        .find(|x| x["family"] == name)
        .unwrap()
}

#[test]
fn every_ms_and_launch_column_is_per_unit() {
    let r = diff(10);
    let t = &r["totals"];
    assert_eq!(
        (f(&t["a_kernels"]), f(&t["b_kernels"]), f(&t["d_kernels"])),
        (2.0, 3.0, 1.0)
    );
    assert_close(f(&t["a_sum_ms"]), 6.0);
    assert_close(f(&t["d_u_all_ms"]), 4.0);
    let dense = family(&r, "dense");
    assert_close(f(&dense["a_ms"]), 4.0);
    assert_close(f(&dense["d_ms"]), -1.0);
    assert_eq!(f(&dense["a_launches"]), 1.0);
    // medians are per launch and never divided by the unit count
    assert_close(f(&dense["a_median_us"]), 4000.0);
    assert_close(f(&dense["d_median_us"]), -1000.0);
    assert!(family(&r, "fused")["a_median_us"].is_null());
    assert!(family(&r, "fused")["d_median_us"].is_null());
}

#[test]
fn only_in_is_by_name_and_kernel_rows_by_name_and_grid() {
    let r = diff(10);
    assert_eq!(r["only_in_a"], json!([]));
    assert_eq!(
        r["only_in_b"],
        json!([{"kernel": "fused", "ms": 5.0, "launches": 1.0}])
    );
    let row = |name: &str, grid: Value| {
        r["kernels"]
            .as_array()
            .unwrap()
            .iter()
            .find(|x| x["kernel"] == name && x["grid"] == grid)
            .unwrap()
            .clone()
    };
    assert_close(f(&row("gemm", json!([1, 1, 1]))["d_ms"]), -4.0);
    assert_close(f(&row("gemm", json!([2, 1, 1]))["d_ms"]), 3.0);
}

#[test]
fn kernel_rows_sort_by_absolute_delta_and_top_cuts_only_them() {
    let r = diff(1);
    let ks: Vec<(Value, Value)> = r["kernels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| (x["kernel"].clone(), x["grid"].clone()))
        .collect();
    assert_eq!(ks, vec![(json!("fused"), json!([1, 1, 1]))]); // +5 outranks -4
    let fams: Vec<&str> = r["families"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x["family"].as_str().unwrap())
        .collect();
    assert_eq!(fams.len(), 3);
    assert_eq!(fams[..2], ["fused", "dense"]);
}

#[test]
fn bad_sides() {
    let (a, b) = kernels();
    let sb = Side {
        kernels: &b,
        labels: vec!["dense", "misc", "fused"],
        units: 1,
    };
    let zero_units = Side {
        kernels: &a,
        labels: vec!["dense", "misc", "dense", "misc"],
        units: 0,
    };
    let empty = Side {
        kernels: &[],
        labels: vec![],
        units: 1,
    };
    for bad in [zero_units, empty] {
        assert!(capture_diff(&bad, &sb, 1).is_err());
    }
}

fn mem() -> Mem {
    Mem::new()
        .file("f.toml", FAMILIES_TOML)
        .capture(
            "a.sqlite",
            Capture::new()
                .kernel("gemm_a", 0.0, 100.0)
                .kernel("gemm_a", 5000.0, 5100.0),
        )
        .capture(
            "b.sqlite",
            Capture::new()
                .kernel("gemm_a", 0.0, 80.0)
                .kernel("norm_new", 80.0, 90.0),
        )
        .file("a.env", "LEVER=0\nPORT=1\nSAME=x\n")
        .file("b.env", "LEVER=1\nPORT=2\nSAME=x\n")
}

fn cli<'a>(extra: &[&'a str]) -> Vec<&'a str> {
    let mut v = vec![
        "diff",
        "a.sqlite",
        "b.sqlite",
        "--device",
        "0",
        "--families",
        "f.toml",
        "--top",
        "5",
        "--json",
    ];
    v.extend_from_slice(extra);
    v
}

#[test]
fn per_request_normalizes_each_side_by_its_asserted_count() {
    let r = mem().json(&cli(&[
        "--per",
        "request",
        "--gap-ms",
        "1",
        "--expect-a",
        "2",
        "--expect-b",
        "1",
    ]));
    let gemm = r["families"]
        .as_array()
        .unwrap()
        .iter()
        .find(|x| x["family"] == "gemm")
        .unwrap();
    assert_close(f(&gemm["a_ms"]), 0.1);
    assert_close(f(&gemm["d_ms"]), -0.02);
    let only: Vec<&str> = r["only_in_b"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x["kernel"].as_str().unwrap())
        .collect();
    assert_eq!(only, vec!["norm_new"]);
}

#[test]
fn a_wrong_request_count_names_the_side() {
    let err = mem().refused(&cli(&[
        "--per",
        "request",
        "--gap-ms",
        "1",
        "--expect-a",
        "1",
        "--expect-b",
        "1",
    ]));
    assert!(err.contains("side A: request count"), "{err}");
}

#[test]
fn request_arguments_are_all_or_nothing() {
    for extra in [
        &["--per", "request", "--gap-ms", "1", "--expect-a", "2"][..],
        &["--per", "window", "--gap-ms", "1"][..],
    ] {
        mem().refused(&cli(extra));
    }
}

#[test]
fn the_coverage_gate_applies_to_each_side() {
    let m = mem().capture(
        "u.sqlite",
        Capture::new()
            .kernel("gemm_a", 0.0, 10.0)
            .kernel("mystery", 10.0, 100.0),
    );
    let err = m.refused(&[
        "diff",
        "a.sqlite",
        "u.sqlite",
        "--device",
        "0",
        "--families",
        "f.toml",
        "--top",
        "5",
        "--per",
        "window",
    ]);
    assert!(err.contains("side B:"), "{err}");
    assert!(err.contains("coverage"), "{err}");
}

#[test]
fn env_pair_must_differ_in_one_lever() {
    let m = mem();
    let r = m.json(&cli(&[
        "--per", "window", "--env", "a.env", "b.env", "--ignore", "^PORT$",
    ]));
    assert_eq!(r["lever"], json!({"key": "LEVER", "a": "0", "b": "1"}));
    let err = m.refused(&cli(&["--per", "window", "--env", "a.env", "b.env"]));
    assert!(err.contains("found 2 differing keys"), "{err}");
    let err = m.refused(&cli(&["--per", "window", "--ignore", "^PORT$"]));
    assert!(err.contains("--ignore applies to --env only"), "{err}");
}
