// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the synthetic `metprof tree` setup shared by the tree and KB
//! tests: a family table with its [tree] section, a provenance file, a
//! two-request capture and the base argv.

use metprof::cli::Outcome;
use serde_json::Value;

use super::{Capture, K, K0, Mem};

pub const RULES: &str = r#"
schema = 1
coverage_min = 0.5
[api]
host_sync_regex = "^cudaStreamSynchronize"
[[rule]]
family = "gemm"
group = "dense"
class = "compute"
name_regex = "^gemm"
[[rule]]
family = "norm"
group = "misc"
class = "compute"
name_regex = "^norm"
"#;

pub const TREE: &str = "[tree]\ncoverage_min = 0.9\n[tree.model.m]\nfuses = { gemm = [\"a\", \"b\"] }\n\
                    [[tree.model.m.node]]\npath = \"blk\"\nfamilies = [\"^gemm$\"]\n";

pub const PROV: &str = "binary = \"/bin/serve\"\ngit_sha = \"ABC1234\"\ngit_dirty = false\nenv_sha256 = \"unknown\"\n\
                    recipe = \"r/x\"\nisl = 1024\nosl = \"unknown\"\nconcurrency = 1\nrank = 0\n";

/// sha256("abc") and sha256(""), the FIPS 180-2 vectors.
pub const SHA_ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
pub const SHA_EMPTY: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

pub fn named(d: &'static str) -> K {
    K {
        demangled: Some(d),
        ..K0
    }
}

/// Two requests 969 us apart: gemm 10 + norm 5 + an unclassified 1, then gemm 10.
pub fn capture() -> Capture {
    Capture::new()
        .kernel("gemm_a", 0.0, 10.0)
        .kernel("norm_x", 20.0, 25.0)
        .kernel_with("odd", 30.0, 31.0, named("void odd<4>()"))
        .kernel("gemm_a", 1000.0, 1010.0)
}

pub fn mem_with(tree: &str, prov: &str) -> Mem {
    Mem::new()
        .capture("/cap.sqlite", capture())
        .bytes("/cap.sqlite", b"abc")
        .bytes("/bin/serve", b"")
        .file("/cfg.toml", &format!("{RULES}{tree}"))
        .file("/prov.toml", prov)
}

pub fn mem() -> Mem {
    mem_with(TREE, PROV)
}

pub const BASE: [&str; 13] = [
    "tree",
    "--capture",
    "/cap.sqlite",
    "--device",
    "0",
    "--config",
    "/cfg.toml",
    "--provenance",
    "/prov.toml",
    "--out",
    "/t.json",
    "--segment",
    "all",
];

pub fn argv<'a>(extra: &[&'a str]) -> Vec<&'a str> {
    let mut v = BASE.to_vec();
    v.extend_from_slice(extra);
    v
}

pub fn written(out: &Outcome) -> Value {
    assert_eq!(out.code, 0, "{}", out.stderr);
    assert_eq!(out.files.len(), 1);
    assert_eq!(out.files[0].0, "/t.json");
    serde_json::from_slice(&out.files[0].1).expect("tree JSON")
}
