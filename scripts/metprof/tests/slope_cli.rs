// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: `metprof slope` end to end through the real `cli::run` over
//! three synthetic decode captures: the tree written with its per-node
//! slope, the summary, order independence, the run record and its KB
//! samples, `tree-diff` reading the file back, and every refusal.

mod common;

use common::tree::{PROV, RULES, written};
use common::{Capture, K, K0, Mem};
use metprof::core::provenance::sha256_hex;
use metprof::core::slope::Point;
use metprof_kb::derive::derive;
use metprof_kb::port::Arg;
use metprof_kb::schema::{Body, Kind};
use serde_json::{Value, json};

const SLOPE: &str = "[tree]\ncoverage_min = 0.9\n[slope]\nflat_band_ms_per_row = 0.001\n\
                     [slope.cost_driver]\ngemm = \"weight_bytes\"\n";

/// Gemm time per step at each width: shared weight bytes, so it grows slower
/// than the rows.
fn gemm_us(rows: u64) -> f64 {
    match rows {
        1 => 100.0,
        2 => 140.0,
        _ => 160.0,
    }
}

/// Three pinned decode steps 1000 us apart, each one graph launch: one gemm,
/// one 50 us norm per row and one unclassified 1 us kernel.
fn capture(rows: u64) -> Capture {
    let mut c = Capture::new();
    for step in 0..3i64 {
        let s = 1000.0 * step as f64;
        let node = |n: i64| K {
            corr: Some(100 + step),
            node: Some(n),
            ..K0
        };
        c = c.kernel_with("gemm_a", s, s + gemm_us(rows), node(1));
        for r in 0..rows as i64 {
            let t = s + 200.0 + 60.0 * r as f64;
            c = c.kernel_with("norm_b", t, t + 50.0, node(2 + r));
        }
        c = c.kernel_with("odd", s + 500.0, s + 501.0, node(99));
    }
    c
}

const NAMES: [(u64, &str, &[u8]); 3] = [
    (1, "/w1.sqlite", b"one"),
    (2, "/w2.sqlite", b"two"),
    (4, "/w4.sqlite", b"four"),
];

fn mem_with(cfg: &str) -> Mem {
    let mut m = Mem::new()
        .bytes("/bin/serve", b"")
        .file("/cfg.toml", &format!("{RULES}{cfg}"))
        .file("/prov.toml", PROV);
    for (rows, path, bytes) in NAMES {
        m = m.capture(path, capture(rows)).bytes(path, bytes);
    }
    m
}

fn mem() -> Mem {
    mem_with(SLOPE)
}

const AT: [&str; 3] = ["1=/w1.sqlite", "2=/w2.sqlite", "4=/w4.sqlite"];

fn argv<'a>(at: &[&'a str], extra: &[&'a str]) -> Vec<&'a str> {
    let mut v = vec!["slope"];
    for a in at {
        v.extend(["--at", a]);
    }
    v.extend([
        "--device",
        "0",
        "--families",
        "/cfg.toml",
        "--min-graph-share",
        "0.9",
        "--provenance",
        "/prov.toml",
        "--out",
        "/t.json",
    ]);
    v.extend_from_slice(extra);
    v
}

fn ms_per_step(rows: u64, us_per_step: f64) -> f64 {
    Point::of(rows, 3, (us_per_step * 3.0 * 1000.0).round() as i64, 3).ms_per_step
}

#[test]
fn slope_writes_the_tree_with_a_slope_on_every_node() {
    let v = written(&mem().run(&argv(&AT, &[])));
    assert_eq!(v["schema"], json!("metprof.kernel_tree/1"));
    assert_eq!(v["provenance"]["segment"], json!("decode_steps"));
    assert_eq!(
        v["provenance"]["capture_sha256"],
        json!(sha256_hex(b"four")),
        "the widest capture names the tree"
    );
    let widths: Vec<Value> = NAMES
        .iter()
        .map(|(rows, _, b)| {
            json!({"rows": rows, "steps": 3, "graph_share": 1.0, "capture_sha256": sha256_hex(b)})
        })
        .collect();
    assert_eq!(v["slope"]["widths"], json!(widths));
    assert_eq!(v["slope"]["flat_band_ms_per_row"], json!(0.001));
    let gemm = &v["kernels"]["dense"]["kernels"]["gemm"];
    assert_eq!(gemm["total_time"], json!(480.0), "the widest width's tree");
    let s = &gemm["slope"];
    assert_eq!(s["class"], json!("FALLING"));
    assert_eq!(s["cost_driver"], json!("weight_bytes"));
    let got: Vec<&Value> = s["widths"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| &w["ms_per_step"])
        .collect();
    let want: Vec<Value> = [1, 2, 4]
        .iter()
        .map(|r| json!(ms_per_step(*r, gemm_us(*r))))
        .collect();
    assert_eq!(got, want.iter().collect::<Vec<_>>());
    let norm = &v["kernels"]["misc"]["kernels"]["norm"]["slope"];
    assert_eq!(norm["class"], json!("FLAT"));
    assert_eq!(norm["cost_driver"], json!("undeclared"));
    let lpr: Vec<&Value> = norm["widths"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| &w["launches_per_row"])
        .collect();
    assert_eq!(lpr, [&json!(1.0), &json!(1.0), &json!(1.0)]);
    let leaf = &v["kernels"]["dense"]["kernels"]["gemm"]["kernels"]["gemm_a"]["slope"];
    assert!(
        leaf.get("cost_driver").is_none(),
        "a kernel is not a family"
    );
    assert_eq!(leaf["class"], json!("FALLING"));
    // 151, 241, 361 us: marginals 90 then 60 us per row.
    assert_eq!(v["slope"]["step"]["class"], json!("FALLING"));
}

#[test]
fn the_summary_lists_widths_the_step_and_families_widest_first() {
    let s = mem().json(&argv(&AT, &[]));
    assert_eq!(s["out"], json!("/t.json"));
    assert_eq!(s["segment"], json!("decode_steps"));
    assert_eq!(s["step_class"], json!("FALLING"));
    let rows: Vec<&Value> = s["widths"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| &w["rows"])
        .collect();
    assert_eq!(rows, [&json!(1), &json!(2), &json!(4)]);
    let fams: Vec<(&Value, &Value, &Value)> = s["families"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| (&f["family"], &f["class"], &f["cost_driver"]))
        .collect();
    // norm costs 200 us at 4 rows, gemm 160 us.
    assert_eq!(
        fams,
        [
            (&json!("norm"), &json!("FLAT"), &json!("undeclared")),
            (&json!("gemm"), &json!("FALLING"), &json!("weight_bytes")),
        ]
    );
    let g = &s["families"][1];
    assert_eq!(g["path"], json!("dense/gemm"));
    assert_eq!(g["ms_per_step@4"], json!(ms_per_step(4, 160.0)));
    assert!(g.get("ms_per_row@1-2").is_some() && g.get("ms_per_row@2-4").is_some());
    assert_eq!(g["launches_per_row@4"], json!(0.25));
    assert_eq!(g["absent_at"], json!([]));
    let text = mem().run(&argv(&AT, &[])).stdout;
    for want in [
        "FALLING",
        "weight_bytes",
        "undeclared",
        "dense/gemm",
        "(whole step)",
    ] {
        assert!(text.contains(want), "{want}: {text}");
    }
}

#[test]
fn the_order_of_the_widths_does_not_change_the_file() {
    let first = mem().run(&argv(&AT, &[])).files.remove(0).1;
    for at in [[AT[2], AT[0], AT[1]], [AT[1], AT[2], AT[0]]] {
        let out = mem().run(&argv(&at, &[]));
        assert_eq!(out.code, 0, "{}", out.stderr);
        assert_eq!(out.files[0].1, first, "{at:?}");
    }
}

#[test]
fn the_run_record_keeps_rows_and_digests_and_the_kb_keeps_each_slope() {
    let m = mem();
    let out = m.run(&argv(&AT, &[]));
    let tree = written(&out);
    let [r] = <[_; 1]>::try_from(m.records()).expect("one record");
    for (rows, path, bytes) in NAMES {
        let at = r
            .args
            .iter()
            .position(|a| *a == Arg::Word(format!("{rows}=")))
            .expect("rows recorded");
        assert_eq!(r.args[at - 1], Arg::Word("--at".into()));
        assert_eq!(
            r.args[at + 1],
            Arg::File {
                sha256: sha256_hex(bytes),
                basename: path.trim_start_matches('/').into(),
            }
        );
    }
    assert_eq!(r.files[0].json.as_ref(), Some(&tree));
    assert_eq!(
        r.stamps.capture_sha256, None,
        "three captures, no one stamp"
    );
    let g = derive(&r, 3).expect("derives");
    let sample = |path: &str| {
        g.records
            .iter()
            .filter(|x| x.kind == Kind::KernelSample)
            .find_map(|x| match &x.body {
                Some(Body::KernelSample(k)) if k.node_path == path => Some(k.clone()),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no sample {path}"))
    };
    assert_eq!(
        sample("dense/gemm").slope.as_ref(),
        Some(&tree["kernels"]["dense"]["kernels"]["gemm"]["slope"])
    );
    assert_eq!(
        sample("misc/norm/norm_b").slope.as_ref(),
        Some(&tree["kernels"]["misc"]["kernels"]["norm"]["kernels"]["norm_b"]["slope"])
    );
}

#[test]
fn tree_diff_reads_a_slope_tree_and_refuses_an_edited_one() {
    let bytes = mem().run(&argv(&AT, &[])).files.remove(0).1;
    let mut v: Value = serde_json::from_slice(&bytes).unwrap();
    let m = mem().bytes("/old.json", &bytes).bytes("/new.json", &bytes);
    let d = m.json(&["tree-diff", "/old.json", "/new.json"]);
    assert!(d.is_object(), "{d}");
    v["kernels"]["dense"]["kernels"]["gemm"]["slope"]["class"] = json!("RISING");
    let m = m.bytes("/bad.json", &serde_json::to_vec(&v).unwrap());
    let e = m.refused(&["tree-diff", "/old.json", "/bad.json"]);
    assert!(e.contains("is not what its per-width points imply"), "{e}");
}

#[test]
fn widths_that_cannot_give_a_slope_are_refused() {
    let cases: [(Vec<&str>, &str); 6] = [
        (vec![AT[0], AT[1]], "at least 3 widths"),
        (vec![AT[0], AT[1], AT[1]], "width 2 rows is given twice"),
        (
            vec![AT[0], AT[1], "x=/w4.sqlite"],
            "argument --at: 'x=/w4.sqlite': ROWS must be a whole number",
        ),
        (vec![AT[0], AT[1], "0=/w4.sqlite"], "at least 1"),
        (vec![AT[0], AT[1], "/w4.sqlite"], "expected ROWS=PATH"),
        (vec![AT[0], AT[1], "8=/w8.sqlite"], "--at 8=/w8.sqlite"),
    ];
    for (at, want) in cases {
        let out = mem().run(&argv(&at, &[]));
        assert_eq!(out.code, 2, "{at:?}");
        assert!(out.files.is_empty());
        assert!(out.stderr.contains(want), "{want}: {}", out.stderr);
    }
    let e = mem().refused(&argv(&[], &[]));
    assert!(e.contains("--at") && e.contains("required"), "{e}");
}

#[test]
fn config_and_windows_that_cannot_give_a_slope_are_refused() {
    let e = mem_with("[tree]\ncoverage_min = 0.9\n").refused(&argv(&AT, &[]));
    assert!(e.contains("[slope] is required"), "{e}");
    let e = mem_with(&SLOPE.replace("gemm = ", "attn = ")).refused(&argv(&AT, &[]));
    assert!(e.contains("'attn' is not a declared family"), "{e}");
    let e = mem().refused(
        &argv(&AT, &[])
            .into_iter()
            .map(|a| if a == "0.9" { "1.5" } else { a })
            .collect::<Vec<_>>(),
    );
    assert!(
        e.contains("--at 1=/w1.sqlite") && e.contains("--min-graph-share 1.5"),
        "{e}"
    );
    let e = mem_with(&SLOPE.replace("0.9", "0.999")).refused(&argv(&AT, &[]));
    assert!(
        e.contains("--at 1=/w1.sqlite") && e.contains("coverage_min"),
        "{e}"
    );
}
