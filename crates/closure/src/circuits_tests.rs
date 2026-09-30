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
        "kernels/b300/common",
        "kernels/b200",
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
        (
            "kernels/gb10/HARDWARE.toml",
            "[hardware]\narch = \"sm_121f\"\n",
        ),
        // 2026-09-30: hopper declares no rules and inherits no tree.
        (
            "kernels/hopper/HARDWARE.toml",
            "[hardware]\narch = \"sm_90a\"\n",
        ),
        // 2026-09-30: b300 plans with gb10's rules through FUSIONS `inherits`, without
        // inheriting gb10's sources.
        (
            "kernels/b300/HARDWARE.toml",
            "[hardware]\narch = \"sm_103a\"\n",
        ),
        (
            "kernels/b300/common/FUSIONS.toml",
            "schema = 1\ninherits = \"gb10\"\n",
        ),
        // 2026-09-30: b200 declares no rules and takes its HARDWARE.toml parent's.
        (
            "kernels/b200/HARDWARE.toml",
            "[hardware]\narch = \"sm_100a\"\ninherits = \"gb10\"\n",
        ),
    ] {
        std::fs::write(dir.join(p), s).unwrap();
    }
    dir
}

#[test]
fn every_circuit_toml_and_the_hardware_rules_are_listed_and_nothing_else() {
    let root = tree("list");
    let rel: Vec<String> = circuit_configs(&root, "gb10")
        .unwrap()
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
    assert_eq!(circuit_configs(&root, "hopper").unwrap().len(), 3);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn editing_a_circuit_file_moves_the_closure_hash() {
    let root = tree("hash");
    let src = root.join("kernels/gb10/common/norm.cu");
    let inputs = || ClosureInputs {
        sources: vec![src.clone()],
        configs: circuit_configs(&root, "gb10").unwrap(),
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

fn rel(root: &Path, files: &[PathBuf]) -> Vec<String> {
    files
        .iter()
        .map(|p| p.strip_prefix(root).unwrap().display().to_string())
        .collect()
}

#[test]
fn the_rule_chain_follows_fusions_and_hardware_inheritance() {
    let root = tree("chain");
    assert_eq!(
        rel(&root, &fusions_chain(&root, "gb10").unwrap()),
        ["kernels/gb10/common/FUSIONS.toml"]
    );
    assert_eq!(
        rel(&root, &fusions_chain(&root, "b300").unwrap()),
        [
            "kernels/gb10/common/FUSIONS.toml",
            "kernels/b300/common/FUSIONS.toml"
        ]
    );
    assert_eq!(
        rel(&root, &fusions_chain(&root, "b200").unwrap()),
        ["kernels/gb10/common/FUSIONS.toml"]
    );
    assert!(fusions_chain(&root, "hopper").unwrap().is_empty());
    let _ = std::fs::remove_dir_all(&root);
}

/// 2026-09-30: The D6 defect: a child class's closure did not move when the parent's rules did.
/// Checked for both inheritance routes, and against a class that inherits nothing.
#[test]
fn editing_the_parent_fusions_moves_every_inheriting_class_closure() {
    let root = tree("parent");
    let src = root.join("kernels/gb10/common/norm.cu");
    let closure = |hw: &str| {
        hash(
            &root,
            &ClosureInputs {
                sources: vec![src.clone()],
                configs: circuit_configs(&root, hw).unwrap(),
                flags: Vec::new(),
                arch: "sm_121f".into(),
                compiler: "nvcc".into(),
            },
        )
        .unwrap()
    };
    let before: Vec<String> = ["b300", "b200", "hopper"].map(closure).to_vec();
    std::fs::write(
        root.join("kernels/gb10/common/FUSIONS.toml"),
        "schema = 1\n# a rule edit\n",
    )
    .unwrap();
    let after: Vec<String> = ["b300", "b200", "hopper"].map(closure).to_vec();
    assert_ne!(
        before[0], after[0],
        "b300 inherits gb10's rules through FUSIONS"
    );
    assert_ne!(
        before[1], after[1],
        "b200 inherits gb10's rules through HARDWARE"
    );
    assert_eq!(before[2], after[2], "hopper here inherits no rules");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_rule_chain_that_does_not_resolve_is_an_error_not_a_shorter_list() {
    let root = tree("broken");
    std::fs::write(
        root.join("kernels/gb10/common/FUSIONS.toml"),
        "schema = 1\ninherits = \"b300\"\n",
    )
    .unwrap();
    let cycle = fusions_chain(&root, "b300").unwrap_err().to_string();
    assert!(cycle.contains("cycle"), "{cycle}");
    std::fs::write(
        root.join("kernels/b300/common/FUSIONS.toml"),
        "schema = 1\ninherits = 3\n",
    )
    .unwrap();
    let shape = fusions_chain(&root, "b300").unwrap_err().to_string();
    assert!(shape.contains("not a class name"), "{shape}");
    std::fs::write(
        root.join("kernels/b300/common/FUSIONS.toml"),
        "schema = 1\ninherits = \"nowhere\"\n",
    )
    .unwrap();
    assert!(
        fusions_chain(&root, "b300").is_err(),
        "a missing parent class"
    );
    let _ = std::fs::remove_dir_all(&root);
}
