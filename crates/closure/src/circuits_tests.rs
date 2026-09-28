// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Which circuit files enter a target's closure, and that editing one moves it.
//!
//! Owner: metrale-closure.
//! Invariants: none beyond the types.

use super::*;
use crate::{ClosureInputs, hash};

fn tree(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("metrale-circuits-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    for d in [
        "kernels/circuits/precision",
        "kernels/circuits/plans",
        "kernels/gb10/common",
        "kernels/hopper/common",
    ] {
        std::fs::create_dir_all(dir.join(d)).unwrap();
    }
    for (p, s) in [
        ("kernels/circuits/qwen3_5.toml", "a = 1\n"),
        ("kernels/circuits/INSTANCES.toml", "b = 1\n"),
        ("kernels/circuits/precision/dense.toml", "c = 1\n"),
        ("kernels/circuits/plans/qwen3_5-decode-n1.txt", "plan\n"),
        ("kernels/circuits/ROUTING-AUDIT.md", "doc\n"),
        ("kernels/gb10/common/FUSIONS.toml", "schema = 1\n"),
        ("kernels/gb10/common/norm.cu", "// k\n"),
    ] {
        std::fs::write(dir.join(p), s).unwrap();
    }
    dir
}

#[test]
fn every_circuit_toml_and_the_hardware_rules_are_listed_and_nothing_else() {
    let root = tree("list");
    let rel: Vec<String> = circuit_configs(&root, "gb10")
        .iter()
        .map(|p| p.strip_prefix(&root).unwrap().display().to_string())
        .collect();
    assert_eq!(
        rel,
        [
            "kernels/circuits/INSTANCES.toml",
            "kernels/circuits/precision/dense.toml",
            "kernels/circuits/qwen3_5.toml",
            "kernels/gb10/common/FUSIONS.toml",
        ]
    );
    // 2026-09-28: A hardware without its own FUSIONS.toml still hashes the circuits.
    assert_eq!(circuit_configs(&root, "hopper").len(), 3);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn editing_a_circuit_file_moves_the_closure_hash() {
    let root = tree("hash");
    let src = root.join("kernels/gb10/common/norm.cu");
    let inputs = || ClosureInputs {
        sources: vec![src.clone()],
        configs: circuit_configs(&root, "gb10"),
        flags: Vec::new(),
        arch: "sm_121f".into(),
        compiler: "nvcc".into(),
    };
    let before = hash(&root, &inputs()).unwrap();
    std::fs::write(
        root.join("kernels/circuits/precision/dense.toml"),
        "c = 2\n",
    )
    .unwrap();
    let after = hash(&root, &inputs()).unwrap();
    assert_ne!(before, after);
    std::fs::write(
        root.join("kernels/circuits/plans/qwen3_5-decode-n1.txt"),
        "other\n",
    )
    .unwrap();
    assert_eq!(
        hash(&root, &inputs()).unwrap(),
        after,
        "a rendered plan is not an input"
    );
    let _ = std::fs::remove_dir_all(&root);
}
