// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `metprof tree` and `tree-diff` end to end through the real
//! `cli::run`: the file written, provenance digests, segments, the coverage
//! gate, baselines and every refusal. The circuit cases plan with the repo's
//! own `kernels/` texts and check the plan against the checked-in golden.

mod common;

use common::tree::{PROV, RULES, SHA_ABC, SHA_EMPTY, TREE, argv, mem, mem_with, written};
use common::{Capture, K, K0, Mem};
use metprof::adapters::circuit::{KernelsDir, plan_texts};
use metprof::adapters::files::FsFiles;
use metprof::core::tree::circuit::plan_view;
use metprof::ports::{CircuitTexts, Files};
use serde_json::{Value, json};

#[test]
fn tree_writes_the_file_with_its_provenance() {
    let out = mem().run(&argv(&["--model", "m"]));
    let v = written(&out);
    assert_eq!(v["schema"], json!("metprof.kernel_tree/1"));
    assert_eq!(v["time_unit"], json!("us"));
    assert_eq!(v["source"], json!("families"));
    let p = v["provenance"].as_object().unwrap();
    let keys: Vec<&str> = p.keys().map(String::as_str).collect();
    assert_eq!(
        keys,
        [
            "capture_sha256",
            "binary_sha256",
            "git_sha",
            "git_dirty",
            "env_sha256",
            "recipe",
            "isl",
            "osl",
            "concurrency",
            "rank",
            "device",
            "segment",
            "metprof_version",
            "window"
        ]
    );
    assert_eq!(p["capture_sha256"], json!(SHA_ABC));
    assert_eq!(p["binary_sha256"], json!(SHA_EMPTY));
    assert_eq!(p["git_sha"], json!("abc1234"));
    assert_eq!(p["env_sha256"], json!("unknown"));
    assert_eq!(p["osl"], json!("unknown"));
    assert_eq!(p["isl"], json!(1024));
    assert_eq!(p["segment"], json!("all"));
    assert_eq!(p["window"], json!({"start_ns": 0, "end_ns": 1_010_000}));
    let blk = &v["kernels"]["blk"];
    assert_eq!(blk["kernels"]["gemm"]["fuses"], json!(["a", "b"]));
    assert_eq!(blk["total_time"], json!(20.0));
    assert_eq!(
        blk["kernels"]["gemm"]["kernels"]["gemm_a"]["launches"],
        json!(2)
    );
    assert_eq!(
        v["kernels"]["misc"]["kernels"]["norm"]["total_time"],
        json!(5.0)
    );
    let odd = &v["unattributed"]["void odd<4>()"];
    assert_eq!(
        (&odd["total_time"], &odd["launches"]),
        (&json!(1.0), &json!(1))
    );
    assert_eq!(odd["variants"][0]["demangled"], json!("void odd<4>()"));
    assert_eq!(v["unattributed"].as_object().unwrap().len(), 1);
    assert_eq!(v["total_time"], json!(26.0));
    assert!(out.stdout.contains("out: /t.json"), "{}", out.stdout);
    let s = mem().json(&argv(&[]));
    assert_eq!(
        (s["coverage"].clone(), s["launches"].clone()),
        (json!(25.0 / 26.0), json!(4))
    );
}

#[test]
fn one_request_is_a_segment() {
    let out = mem().run(
        &argv(&[])
            .into_iter()
            .take(11)
            .chain(["--segment", "1", "--gap-ms", "0.5", "--expect", "2"])
            .collect::<Vec<_>>(),
    );
    let v = written(&out);
    assert_eq!(v["provenance"]["segment"], json!("request.1"));
    let w = json!({"start_ns": 1_000_000, "end_ns": 1_010_000});
    assert_eq!(v["provenance"]["window"], w);
    assert_eq!(v["total_time"], json!(10.0));
    let e = mem().refused(
        &argv(&[])
            .into_iter()
            .take(11)
            .chain(["--segment", "2", "--gap-ms", "0.5", "--expect", "2"])
            .collect::<Vec<_>>(),
    );
    assert!(e.contains("the window has 2 request(s)"), "{e}");
    let e = mem().refused(
        &argv(&[])
            .into_iter()
            .take(11)
            .chain(["--segment", "1"])
            .collect::<Vec<_>>(),
    );
    assert!(e.contains("requires --gap-ms and --expect"), "{e}");
    let e = mem().refused(
        &argv(&[])
            .into_iter()
            .take(11)
            .chain(["--segment", "last"])
            .collect::<Vec<_>>(),
    );
    assert!(e.contains("\"all\" or a request index"), "{e}");
    let e = mem().refused(&argv(&["--gap-ms", "0.5"]));
    assert!(e.contains("takes no --gap-ms"), "{e}");
}

#[test]
fn the_gate_refuses_and_writes_nothing() {
    let out = mem_with(&TREE.replace("0.9", "0.99"), PROV).run(&argv(&[]));
    assert_eq!(out.code, 2);
    assert!(out.files.is_empty());
    assert!(
        out.stderr.contains("below [tree] coverage_min 0.99"),
        "{}",
        out.stderr
    );
    let e = mem_with("", PROV).refused(&argv(&[]));
    assert!(e.contains("/cfg.toml: families: [tree] is required"), "{e}");
}

#[test]
fn provenance_is_complete_or_refused() {
    let cases = [
        (PROV.replace("rank = 0\n", ""), "missing required key"),
        (
            format!("{PROV}binary_sha256 = \"{SHA_ABC}\"\n"),
            "exactly one of binary and binary_sha256",
        ),
        (
            PROV.replace("env_sha256 = \"unknown\"", "env_sha256 = \"ABC\""),
            "64 lower-case hex",
        ),
        (
            PROV.replace("isl = 1024", "isl = -1"),
            "'isl' has an invalid value",
        ),
        (
            PROV.replace("git_dirty = false", "git_dirty = \"no\""),
            "'git_dirty' has an invalid value",
        ),
        (format!("{PROV}host = \"x\"\n"), "unknown key"),
        (
            PROV.replace("/bin/serve", "/bin/missing"),
            "provenance binary: cannot read /bin/missing",
        ),
    ];
    for (prov, want) in cases {
        let e = mem_with(TREE, &prov).refused(&argv(&[]));
        assert!(e.contains(want), "{want}: {e}");
    }
}

#[test]
fn flags_that_do_not_fit_are_refused() {
    let e = mem().refused(&argv(&["--circuit", "qwen3_5", "--mode", "decode"]));
    assert!(e.contains("missing --rows, --policy, --kernels-dir"), "{e}");
    let e = mem().refused(&argv(&[
        "--circuit",
        "a",
        "--mode",
        "decode",
        "--rows",
        "1",
        "--policy",
        "p",
        "--kernels-dir",
        "/k",
        "--model",
        "m",
    ]));
    assert!(e.contains("--model is a family-table overlay"), "{e}");
    let e = mem().refused(&argv(&["--model", "zz"]));
    assert!(e.contains("no [tree.model.zz]"), "{e}");
    let e = mem().refused(&["tree", "--capture", "/cap.sqlite", "--device", "0"]);
    assert!(e.contains("required"), "{e}");
}

#[test]
fn a_baseline_and_a_diff_read_back_the_written_tree() {
    let old = mem().run(&argv(&["--model", "m"])).files.remove(0).1;
    let m = mem().bytes("/old.json", &old);
    let v = written(&m.run(&argv(&["--model", "m", "--baseline", "/old.json"])));
    assert_eq!(v["baseline_time"], json!(26.0));
    assert_eq!(v["delta"], json!(0.0));
    assert_eq!(v["fused_away"], json!([]));
    // Without the overlay the same capture has other paths: all new, all fused away.
    let flat = written(&mem().run(&argv(&[])));
    let m = m.bytes("/flat.json", &serde_json::to_vec(&flat).unwrap());
    let d = m.json(&["tree-diff", "/old.json", "/flat.json"]);
    assert_eq!(
        d["fused_away"],
        json!([{"path": "blk", "fuses": [], "total_time": 20.0}])
    );
    assert_eq!(d["new"][0]["path"], json!("dense"));
    let out = m.run(&["tree-diff", "/old.json", "/flat.json", "--out", "/d.json"]);
    assert_eq!(out.code, 0, "{}", out.stderr);
    let file: Value = serde_json::from_slice(&out.files[0].1).unwrap();
    assert_eq!(file["nodes"], d["nodes"]);
    assert!(
        m.run(&["tree-diff", "/old.json", "/flat.json"])
            .files
            .is_empty()
    );
    let mut bad: Value = serde_json::from_slice(&old).unwrap();
    bad["total_time"] = json!(27.0);
    let m = m.bytes("/bad.json", &serde_json::to_vec(&bad).unwrap());
    let e = m.refused(&argv(&["--baseline", "/bad.json"]));
    assert!(e.starts_with("metprof: error: /bad.json: "), "{e}");
    let e = m.refused(&["tree-diff", "/old.json", "/bad.json"]);
    assert!(e.contains("/bad.json"), "{e}");
}

const KERNELS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels");
const RECIPE: &str = "qwen3.8/qwen3.8-27b-nvfp4-unsloth";
const GOLDEN: &str = include_str!("../../../kernels/circuits/plans/qwen3_5-decode-n1.txt");

/// A header value of the checked-in golden plan (`digest: X`, `launches: N`).
fn golden(key: &str) -> String {
    let pat = format!("{key}: ");
    GOLDEN
        .lines()
        .take(12)
        .flat_map(|l| l.split("  "))
        .find_map(|f| f.strip_prefix(&pat))
        .unwrap_or_else(|| panic!("golden has no {key}"))
        .to_string()
}

#[test]
fn the_offline_plan_is_the_golden_plan() {
    let dir = KernelsDir {
        files: &FsFiles,
        root: KERNELS,
    };
    let plan = plan_view(
        &plan_texts(&dir, RECIPE, "qwen3_5").unwrap(),
        RECIPE,
        "qwen3_5",
        "decode",
        1,
    )
    .unwrap();
    assert_eq!(plan.digest, golden("digest"));
    let launches: u64 = plan
        .groups
        .iter()
        .map(|g| g.reps * g.kernels.len() as u64)
        .sum();
    assert_eq!(launches.to_string(), golden("launches"));
    let e = plan_texts(&dir, RECIPE, "dense_gqa").unwrap_err().0;
    assert!(e.contains("not 'dense_gqa'"), "{e}");
    let e = plan_view(
        &plan_texts(&dir, RECIPE, "qwen3_5").unwrap(),
        RECIPE,
        "qwen3_5",
        "verify_batch",
        2,
    )
    .unwrap_err()
    .0;
    assert!(e.contains("verify_batch"), "{e}");
}

#[test]
fn kernels_dir_reads_only_plain_names() {
    let dir = KernelsDir {
        files: &FsFiles,
        root: KERNELS,
    };
    for bad in ["../x", "a/b", "", ".."] {
        assert!(
            dir.circuit(bad).unwrap_err().0.contains("not a plain name"),
            "{bad}"
        );
        assert!(dir.block(bad).is_err(), "{bad}");
    }
    assert!(
        dir.fusions("gb10/../x")
            .unwrap_err()
            .0
            .contains("not a plain name")
    );
    assert!(
        dir.circuit("no_such_arch")
            .unwrap_err()
            .0
            .contains("cannot read")
    );
}

/// The repo's `kernels/` from disk, everything else from memory.
struct Repo<'a>(&'a Mem);

impl Files for Repo<'_> {
    fn read(&self, path: &str) -> std::io::Result<Vec<u8>> {
        if path.starts_with(KERNELS) {
            FsFiles.read(path)
        } else {
            self.0.read(path)
        }
    }
}

#[test]
fn a_capture_that_runs_the_plan_is_attributed_in_full() {
    let dir = KernelsDir {
        files: &FsFiles,
        root: KERNELS,
    };
    let plan = plan_view(
        &plan_texts(&dir, RECIPE, "qwen3_5").unwrap(),
        RECIPE,
        "qwen3_5",
        "decode",
        1,
    )
    .unwrap();
    let mut cap = Capture::new();
    let mut t = 0.0;
    for step in 0..2 {
        // Graph node ids start at 1: a node id of 0 reads as no graph.
        let mut node = 1;
        for g in &plan.groups {
            for _ in 0..g.reps {
                for (_, func) in &g.kernels {
                    let k = K {
                        corr: Some(step),
                        node: Some(node),
                        ..K0
                    };
                    cap = cap.kernel_with(func, t, t + 1.0 + (node % 7) as f64, k);
                    t += 9.0;
                    node += 1;
                }
            }
        }
    }
    let m = mem().capture("/cap.sqlite", cap).file(
        "/cfg.toml",
        &format!("{RULES}[tree]\ncoverage_min = 0.99\n"),
    );
    let out = m.run_with(
        &argv(&[
            "--circuit",
            "qwen3_5",
            "--mode",
            "decode",
            "--rows",
            "1",
            "--policy",
            RECIPE,
            "--kernels-dir",
            KERNELS,
        ]),
        &Repo(&m),
    );
    let v = written(&out);
    assert_eq!(v["source"], json!("circuit"));
    assert_eq!(v["coverage"], json!(1.0));
    assert_eq!(v["unattributed"], json!({}));
    let c = &v["provenance"]["circuit"];
    assert_eq!(c["plan_digest"], json!(golden("digest")));
    assert_eq!(c["launches_per_step"].to_string(), golden("launches"));
    assert_eq!(
        (
            c["aligned_steps"].clone(),
            c["missed"].clone(),
            c["unmatched"].clone()
        ),
        (json!(2), json!(0), json!(0))
    );
    let layers = v["kernels"]["qwen3_5"]["kernels"]["decode"]["kernels"]
        .as_object()
        .unwrap();
    assert!(
        layers.contains_key("layer.0") && layers.contains_key("layer.63"),
        "{:?}",
        layers.keys().collect::<Vec<_>>()
    );
    let groups = layers["layer.0"]["kernels"].as_object().unwrap();
    let fused = groups
        .values()
        .find(|g| g["members"].as_array().is_some_and(|m| m.len() >= 2))
        .expect("a fused group");
    assert_eq!(
        fused["fuses"].as_array().unwrap().len(),
        fused["members"].as_array().unwrap().len()
    );
    assert_eq!(fused["circuit"]["plan_digest"], json!(golden("digest")));
}
