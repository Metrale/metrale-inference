// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: segmentation (requests by host gap, pinned decode steps,
//! per-unit tables) and the speculative-decode step decomposition.

mod common;

use std::collections::HashMap;

use common::{assert_close, assert_err};
use metprof::core::model::{ApiCall, Copy, Kernel};
use metprof::core::segment::{
    Unit, graph_share, per_unit_rows, pinned_steps, split_requests, step_stats,
};
use metprof::core::spec::spec_steps;
use regex::Regex;

const MS: f64 = 1_000_000.0;

fn sync() -> Regex {
    Regex::new("^(cu|cuda)(Stream|Ctx|Event|Device)Synchronize").unwrap()
}

fn k(start: i64, end: i64) -> Kernel {
    kn(start, end, "gemm", 0, None)
}

fn kn(start: i64, end: i64, name: &str, corr: i64, node: Option<i64>) -> Kernel {
    Kernel {
        start,
        end,
        stream: 1,
        correlation: corr,
        graph_node: node,
        name: name.to_string(),
        demangled: name.to_string(),
        grid: (1, 1, 1),
        launch: common::LAUNCH0,
    }
}

fn call(start: i64, end: i64, name: &str, corr: i64) -> ApiCall {
    ApiCall {
        start,
        end,
        name: name.to_string(),
        correlation: corr,
    }
}

fn copy(start: i64, end: i64, kind: &str) -> Copy {
    Copy {
        start,
        end,
        kind: kind.to_string(),
        nbytes: 8,
    }
}

/// Three requests: islands 0-10 and 12-20 (a 2 ns host gap inside), then
/// 50-60, then 100-130.
fn requests() -> Vec<Kernel> {
    vec![
        k(0, 10),
        k(5, 8),
        k(12, 20),
        k(50, 60),
        k(100, 120),
        k(110, 130),
    ]
}

#[test]
fn membership_spans_and_the_largest_intra_gap() {
    let segs = split_requests(&requests(), 30, 3).unwrap();
    let spans: Vec<(i64, i64)> = segs.iter().map(|s| (s.start, s.end)).collect();
    assert_eq!(spans, vec![(0, 20), (50, 60), (100, 130)]);
    let members: Vec<Vec<usize>> = segs.iter().map(|s| s.kernels.clone()).collect();
    assert_eq!(members, vec![vec![0, 1, 2], vec![3], vec![4, 5]]);
    assert_eq!(segs[0].max_intra_gap, 2);
}

#[test]
fn a_gap_equal_to_the_threshold_is_a_boundary() {
    assert_eq!(split_requests(&requests(), 40, 2).unwrap().len(), 2); // 60 -> 100 is exactly 40
    assert_eq!(split_requests(&requests(), 41, 1).unwrap().len(), 1);
}

#[test]
fn wrong_count_is_an_error_that_lists_the_segments() {
    let msg = split_requests(&requests(), 2, 3).unwrap_err().0;
    assert!(msg.contains("into 4 segments, expected 3"), "{msg}");
    assert!(msg.contains("#3:"), "{msg}");
}

#[test]
fn bad_split_arguments() {
    for (gap, expect) in [(0, 1), (5, 0)] {
        assert!(split_requests(&requests(), gap, expect).is_err());
    }
    assert!(split_requests(&[], 5, 1).is_err());
}

/// The decode graph: 3 nodes, 2 launches; a small graph launched 4 times
/// (argmax-like) must not be the step.
fn decode_window() -> Vec<Kernel> {
    let mut ks = Vec::new();
    for (n, t) in [0, 100].into_iter().enumerate() {
        let c = 10 + n as i64;
        ks.push(kn(t, t + 10, "gemm", c, Some(1)));
        ks.push(kn(t + 12, t + 40, "gemm", c, Some(2)));
        ks.push(kn(t + 20, t + 30, "gemm", c, Some(3)));
    }
    for (n, t) in [50, 60, 150, 160].into_iter().enumerate() {
        ks.push(kn(t, t + 2, "argmax", 20 + n as i64, Some(1)));
    }
    ks.push(kn(70, 90, "eager", 0, None));
    ks
}

#[test]
fn the_largest_graph_is_the_step_even_when_a_smaller_one_launches_more() {
    let ks = decode_window();
    let ps = pinned_steps(&ks).unwrap();
    assert_eq!((ps.steps.len(), ps.nodes, ps.graph_launches), (2, 3, 6));
    assert!(ps.steps[0].iter().all(|&i| ks[i].correlation == 10));
}

#[test]
fn span_runs_to_the_latest_end_and_idle_is_span_minus_union() {
    let ks = decode_window();
    let st = step_stats(&ks, &pinned_steps(&ks).unwrap()).unwrap();
    assert_close(st.span_median_ms, 40.0 / MS); // the last-started kernel ends at 30, not 40
    assert_close(st.sum_per_step_ms, 48.0 / MS);
    assert_close(st.idle_median_ms, 2.0 / MS);
}

#[test]
fn graph_share_of_the_busy_union() {
    let ks = decode_window();
    assert_close(
        graph_share(&ks, &pinned_steps(&ks).unwrap()).unwrap(),
        76.0 / (76.0 + 8.0 + 20.0),
    );
}

#[test]
fn no_graph_kernels() {
    assert_err(
        pinned_steps(&[k(0, 1), kn(2, 3, "gemm", 0, Some(0))]),
        "graph-trace=node",
    );
}

#[test]
fn every_column_is_divided_by_the_unit_count() {
    let m = MS as i64;
    let ks = [
        kn(0, 4 * m, "a", 0, None),
        kn(10 * m, 12 * m, "a", 0, None),
        kn(20 * m, 21 * m, "b", 0, None),
    ];
    let rows = per_unit_rows(
        &ks,
        &["fa", "fa", "fb"],
        &[vec![0], vec![1], vec![2]],
        Unit::Request,
        false,
        10,
    )
    .unwrap();
    let by: HashMap<&str, _> = rows
        .iter()
        .map(|r| (r["family"].as_str().unwrap(), r))
        .collect();
    assert_close(by["fa"]["ms_per_request"].as_f64().unwrap(), 2.0);
    // (4, 2, 0): a unit without it counts 0
    assert_close(by["fa"]["median_ms_per_request"].as_f64().unwrap(), 2.0);
    assert_eq!(by["fa"]["requests_with_it"], 2);
    assert_close(by["fb"]["median_ms_per_request"].as_f64().unwrap(), 0.0);
    assert!(
        !rows
            .iter()
            .any(|r| r.keys().any(|key| key.contains("step")))
    );
}

#[test]
fn by_kernel_key_keeps_grids_apart() {
    let mut a1 = kn(0, 1, "a", 0, None);
    let mut a2 = kn(2, 3, "a", 0, None);
    a1.grid = (1, 1, 1);
    a2.grid = (2, 1, 1);
    let rows = per_unit_rows(&[a1, a2], &["f", "f"], &[vec![0, 1]], Unit::Step, true, 10).unwrap();
    let mut grids: Vec<String> = rows.iter().map(|r| r["grid"].to_string()).collect();
    grids.sort();
    assert_eq!(grids, vec!["[1,1,1]", "[2,1,1]"]);
}

/// Verify graph (2 nodes) at t = 0, 100, 200, 300, each launched by a host
/// call 5 or 30 ns earlier; a drafter graph (1 node) and eager kernels run in
/// the post windows.
fn spec_window(with_calls: bool) -> (Vec<Kernel>, Vec<Copy>, Vec<ApiCall>) {
    let (mut ks, mut calls) = (Vec::new(), Vec::new());
    for (n, (t, lead)) in [(0, 5), (100, 30), (200, 5), (300, 30)]
        .into_iter()
        .enumerate()
    {
        let c = n as i64 + 1;
        ks.push(kn(t, t + 30, "gemm", c, Some(1)));
        ks.push(kn(t + 30, t + 70, "gemm", c, Some(2)));
        if with_calls {
            calls.push(call(t - lead, t - lead + 1, "cudaGraphLaunch", c));
        }
    }
    ks.push(kn(75, 80, "drafter", 50, Some(1)));
    ks.push(kn(82, 85, "eager", 0, None));
    ks.push(kn(175, 190, "eager2", 0, None));
    let copies = vec![
        copy(85, 88, "HtoD"),
        copy(90, 92, "DtoH"),
        copy(95, 120, "HtoD"),
    ];
    if with_calls {
        calls.push(call(86, 94, "cuStreamSynchronize", 0));
        calls.push(call(97, 99, "cudaEventRecord", 0));
        calls.push(call(150, 160, "cudaStreamSynchronize_v3020", 0));
    }
    (ks, copies, calls)
}

fn columns(with_calls: bool) -> (usize, usize, usize, HashMap<&'static str, (f64, f64)>) {
    let (ks, copies, calls) = spec_window(with_calls);
    let r = spec_steps(&ks, &copies, &calls, &sync()).unwrap();
    let cols = r
        .columns
        .iter()
        .map(|(c, med, mean)| (*c, (*med, *mean)))
        .collect();
    (r.steps, r.verify_nodes, r.graph_launches, cols)
}

#[test]
fn spec_decomposition() {
    let (steps, nodes, launches, c) = columns(true);
    assert_eq!((steps, nodes, launches), (3, 2, 5));
    // steps run launch call to launch call: -5 -> 70 -> 195 -> 270
    assert_close(c["step_ms"].0, 75.0 / MS);
    assert_close(c["step_ms"].1, 275.0 / 3.0 / MS);
    assert_close(c["verify_gpu_ms"].0, 70.0 / MS);
    assert_close(c["post_ms"].0, 30.0 / MS);
    // post 1 (70-100): drafter 75-80, eager 82-85, copies 85-88, 90-92, 95-100 (clipped) = 18 busy -> idle 12
    // post 2 (170-200): eager2 175-190 -> idle 15; post 3 (270-300): nothing -> idle 30
    assert_close(c["post_gpu_idle_ms"].0, 15.0 / MS);
    assert_close(c["post_gpu_idle_ms"].1, 57.0 / 3.0 / MS);
    // both blocking syncs (86, 150) fall in step 2 [70, 195); the event record is not a sync
    assert_close(c["n_host_sync"].0, 0.0);
    assert_close(c["n_host_sync"].1, 2.0 / 3.0);
    assert_close(c["host_sync_ms"].1, 18.0 / 3.0 / MS);
    assert_close(c["n_htod"].1, 2.0 / 3.0);
    assert_close(c["n_dtoh"].1, 1.0 / 3.0);
    // the verify kernels never count as eager or other-graph work
    assert_close(c["n_eager"].1, 2.0 / 3.0);
    assert_close(c["n_other_graph"].1, 1.0 / 3.0);
}

#[test]
fn launch_time_falls_back_to_the_kernel_start() {
    let (_, _, _, c) = columns(false);
    assert_close(c["step_ms"].0, 100.0 / MS);
    assert_close(c["n_host_sync"].1, 0.0);
}

#[test]
fn spec_refusals() {
    assert_err(
        spec_steps(&[k(0, 1)], &[], &[], &sync()),
        "graph-trace=node",
    );
    assert_err(
        spec_steps(&[kn(0, 1, "gemm", 1, Some(1))], &[], &[], &sync()),
        "at least 2 launches",
    );
}
