// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `segment requests|decode|spec` end to end through `cli::run`.

mod common;

use common::{Capture, FAMILIES_TOML, K, K0, Mem, assert_close, f};

/// Three launches of a 2-node decode graph.
fn decode_capture() -> Capture {
    let mut cap = Capture::new();
    for (n, t) in [0.0, 100.0, 200.0].into_iter().enumerate() {
        let k = |node| K {
            corr: Some(10 + n as i64),
            node: Some(node),
            ..K0
        };
        cap = cap.kernel_with("gemm_a", t, t + 40.0, k(1)).kernel_with(
            "norm_b",
            t + 40.0,
            t + 50.0,
            k(2),
        );
    }
    cap
}

fn mem(c: Capture) -> Mem {
    Mem::new()
        .file("f.toml", FAMILIES_TOML)
        .capture("c.sqlite", c)
}

const BASE: [&str; 5] = ["c.sqlite", "--device", "0", "--families", "f.toml"];

fn argv<'a>(mode: &'a str, extra: &[&'a str]) -> Vec<&'a str> {
    let mut v = vec!["segment", mode];
    v.extend(BASE);
    v.extend_from_slice(extra);
    v
}

#[test]
fn requests_counts_and_never_prints_a_step_column() {
    let cap = Capture::new()
        .kernel("gemm_a", 0.0, 100.0)
        .kernel("norm_b", 101.0, 110.0)
        .kernel("gemm_a", 5000.0, 5100.0);
    let m = mem(cap);
    let a = argv(
        "requests",
        &["--gap-ms", "1", "--expect", "2", "--top", "5", "--json"],
    );
    let out = m.run(&a);
    assert_eq!(out.code, 0, "{}", out.stderr);
    let r: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(r["requests"], 2);
    assert_close(f(&r["max_intra_request_gap_ms"]), 0.001);
    assert_close(f(&r["min_inter_request_gap_ms"]), 4.89);
    let gemm = r["families_per_request"]
        .as_array()
        .unwrap()
        .iter()
        .find(|x| x["family"] == "gemm")
        .unwrap();
    assert_close(f(&gemm["ms_per_request"]), 0.1);
    assert!(!out.stdout.contains("step"));
}

#[test]
fn requests_wrong_count_is_an_error() {
    let m = mem(Capture::new()
        .kernel("gemm_a", 0.0, 100.0)
        .kernel("gemm_a", 5000.0, 5100.0));
    let err = m.refused(&argv(
        "requests",
        &["--gap-ms", "1", "--expect", "1", "--top", "5"],
    ));
    assert!(err.contains("into 2 segments, expected 1"), "{err}");
}

#[test]
fn decode_per_step_numbers() {
    let m = mem(decode_capture().kernel("gemm_eager", 60.0, 70.0));
    let r = m.json(&argv("decode", &["--min-graph-share", "0.5", "--top", "5"]));
    assert_eq!(
        (&r["steps"], &r["nodes_per_step"], &r["graph_launches"]),
        (&3.into(), &2.into(), &3.into())
    );
    assert_close(f(&r["graph_share"]), 150.0 / 160.0);
    let rows = r["families_per_step"].as_array().unwrap();
    let fam = |n: &str| rows.iter().find(|x| x["family"] == n).unwrap();
    assert_close(f(&fam("gemm")["ms_per_step"]), 0.04); // the eager gemm is not in any step
    assert_close(f(&fam("norm")["launches_per_step"]), 1.0);
}

#[test]
fn decode_refuses_a_window_the_graph_does_not_dominate() {
    let m = mem(decode_capture().kernel("gemm_prefill", 300.0, 1000.0));
    let err = m.refused(&argv("decode", &["--min-graph-share", "0.5", "--top", "5"]));
    assert!(err.contains("below --min-graph-share"), "{err}");
}

#[test]
fn spec_refuses_a_capture_without_the_node_column() {
    let m = mem(decode_capture().without_node_column());
    let err = m.refused(&argv("spec", &[]));
    assert!(err.contains("graph-trace=node"), "{err}");
}

#[test]
fn spec_end_to_end() {
    let m = mem(decode_capture()
        .copy(1, 55.0, 60.0)
        .call("cudaStreamSynchronize", 52.0, 58.0));
    let r = m.json(&argv("spec", &[]));
    assert_eq!(r["steps"], 2);
    assert_close(f(&r["medians"]["post_ms"]), 0.05);
    assert_close(f(&r["means"]["post_gpu_idle_ms"]), (0.045 + 0.05) / 2.0);
    assert_close(f(&r["means"]["n_htod"]), 0.5);
    assert_close(f(&r["means"]["n_host_sync"]), 0.5);
}
