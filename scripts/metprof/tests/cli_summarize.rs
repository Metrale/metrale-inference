// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `summarize` and `families` end to end through `cli::run`.

mod common;

use common::{Capture, FAMILIES_TOML, K0, Mem, assert_close, f};

fn mem(c: Capture) -> Mem {
    Mem::new()
        .file("f.toml", FAMILIES_TOML)
        .capture("c.sqlite", c)
}

#[test]
fn end_to_end_numbers() {
    let cap = Capture::new()
        .kernel("gemm_a", 0.0, 100.0)
        .kernel_with("nccl_send", 50.0, 150.0, common::K { stream: 9, ..K0 })
        .kernel("norm_x", 3000.0, 3010.0)
        .kernel_with(
            "gemm_on_other_device",
            0.0,
            5000.0,
            common::K { device: 1, ..K0 },
        );
    let r = mem(cap).json(&[
        "summarize",
        "c.sqlite",
        "--device",
        "0",
        "--families",
        "f.toml",
        "--gap-ms",
        "1",
        "--top",
        "2",
    ]);
    let expect = [
        ("span_ms", 3.01),
        ("sum_ms", 0.21),
        ("u_all_ms", 0.16),
        ("u_compute_ms", 0.11),
        ("u_comm_ms", 0.1),
        ("exposed_comm_ms", 0.05),
        ("idle_ms", 2.85),
        ("gaps", 1.0),
        ("gaps_ms", 2.85),
        ("kernels", 3.0),
    ];
    for (k, v) in expect {
        assert!((f(&r[k]) - v).abs() < 1e-9, "{k}: {} != {v}", r[k]);
    }
    let names: Vec<&str> = r["top_kernels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["kernel"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["gemm_a", "nccl_send"]);
    assert_close(f(&r["top_kernels"][0]["union_ms"]), 0.075);
}

#[test]
fn unclassified_busy_time_fails_the_coverage_gate() {
    let m = mem(Capture::new()
        .kernel("gemm_a", 0.0, 98.0)
        .kernel("mystery", 98.0, 100.0));
    let err = m.refused(&[
        "summarize",
        "c.sqlite",
        "--device",
        "0",
        "--families",
        "f.toml",
        "--gap-ms",
        "1",
        "--top",
        "1",
    ]);
    assert!(err.contains("below coverage_min"), "{err}");
}

#[test]
fn thresholds_have_no_defaults() {
    let m = mem(Capture::new().kernel("gemm_a", 0.0, 1.0));
    let err = m.refused(&[
        "summarize",
        "c.sqlite",
        "--device",
        "0",
        "--families",
        "f.toml",
        "--top",
        "1",
    ]);
    assert!(err.contains("required: --gap-ms"), "{err}");
}

#[test]
fn json_flag_works_on_either_side_of_the_subcommand() {
    let cap = Capture::new()
        .kernel_with(
            "gemm_a",
            0.0,
            100.0,
            common::K {
                grid: (4, 1, 1),
                ..K0
            },
        )
        .kernel("norm_b", 100.0, 150.0);
    let m = mem(cap);
    let tail = [
        "c.sqlite",
        "--device",
        "0",
        "--families",
        "f.toml",
        "--grid-filter",
        "gemm:x>=4",
    ];
    let mut a_argv = vec!["--json", "families"];
    a_argv.extend(tail);
    let mut b_argv = vec!["families"];
    b_argv.extend(tail);
    b_argv.push("--json");
    let a = m.run(&a_argv);
    let b = m.run(&b_argv);
    assert_eq!(a.code, 0, "{}", a.stderr);
    assert_eq!(a, b);
    let r: serde_json::Value = serde_json::from_str(&a.stdout).unwrap();
    let fams: Vec<&str> = r["families"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x["family"].as_str().unwrap())
        .collect();
    assert_eq!(fams, vec!["gemm", "norm"]);
    assert_eq!(r["families"][0]["grid_filter"], "x>=4");
}
