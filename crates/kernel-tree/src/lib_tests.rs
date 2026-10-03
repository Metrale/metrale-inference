// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The embedded tree is the repository's kernels/ text, byte for byte; it unpacks
//! under its own digest, is verified on every call, and is unpacked again when anything under it
//! differs.
//!
//! Owner: kernel tree (build embedding).
//! Invariants: none beyond the types.

use std::path::{Path, PathBuf};

use super::{SHA256, check_path, files, materialize, materialize_tree};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// 2026-10-03: A fresh scratch directory per test.
fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("metrale-kernel-tree-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn toy(tag: &str) -> Vec<(String, Vec<u8>)> {
    vec![
        (
            "kernels/DEVICES.toml".into(),
            format!("# {tag}\n").into_bytes(),
        ),
        ("kernels/gb10/HARDWARE.toml".into(), b"[memory]\n".to_vec()),
    ]
}

const D1: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const D2: &str = "2222222222222222222222222222222222222222222222222222222222222222";

/// 2026-10-02: Path A: every embedded file equals the checkout's, and the files the memory model
/// reads first are present.
#[test]
fn the_embedded_files_are_the_checkouts() {
    let all = files().unwrap();
    for must in [
        "kernels/DEVICES.toml",
        "kernels/circuits/COPIES.toml",
        "kernels/gb10/HARDWARE.toml",
        "kernels/gb10/common/KERNEL_FAMILIES.toml",
    ] {
        assert!(all.iter().any(|(p, _)| p == must), "{must} is not embedded");
    }
    for (rel, bytes) in &all {
        assert_eq!(&std::fs::read(repo().join(rel)).unwrap(), bytes, "{rel}");
    }
    assert!(
        all.windows(2).all(|w| w[0].0 < w[1].0),
        "sorted, no duplicates"
    );
}

/// 2026-10-03: The real tree unpacks under its own digest and reads back the checkout's bytes.
#[test]
fn the_embedded_tree_unpacks_under_its_digest() {
    let dir = scratch("embedded");
    let root = materialize(&dir).unwrap();
    assert_eq!(root, dir.join(SHA256));
    let devices = std::fs::read(root.join("kernels/DEVICES.toml")).unwrap();
    assert_eq!(
        devices,
        std::fs::read(repo().join("kernels/DEVICES.toml")).unwrap()
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

/// 2026-10-03: Path B: a planted edit, an extra file and a missing file are each detected and
/// replaced by the embedded bytes; an intact tree is reused as it is.
#[test]
fn a_tampered_tree_is_detected_and_replaced() {
    let dir = scratch("tamper");
    let want = toy("v1");
    let root = materialize_tree(&dir, D1, &want).unwrap();
    let devices = root.join("kernels/DEVICES.toml");
    let hw = root.join("kernels/gb10/HARDWARE.toml");
    let reused = |p: &Path| std::fs::metadata(p).unwrap().modified().unwrap();
    let before = reused(&devices);
    assert_eq!(materialize_tree(&dir, D1, &want).unwrap(), root);
    assert_eq!(reused(&devices), before, "an intact tree is not rewritten");

    std::fs::write(&devices, b"# planted\n").unwrap();
    materialize_tree(&dir, D1, &want).unwrap();
    assert_eq!(std::fs::read(&devices).unwrap(), b"# v1\n");

    let extra = root.join("kernels/gb10/rogue/MODEL.toml");
    std::fs::create_dir_all(extra.parent().unwrap()).unwrap();
    std::fs::write(&extra, b"x").unwrap();
    materialize_tree(&dir, D1, &want).unwrap();
    assert!(!extra.exists(), "an extra file is removed");

    std::fs::remove_file(&hw).unwrap();
    materialize_tree(&dir, D1, &want).unwrap();
    assert_eq!(std::fs::read(&hw).unwrap(), b"[memory]\n");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// 2026-10-03: Two trees with different digests live side by side; neither touches the other.
#[test]
fn two_digests_coexist() {
    let dir = scratch("coexist");
    let r1 = materialize_tree(&dir, D1, &toy("v1")).unwrap();
    let r2 = materialize_tree(&dir, D2, &toy("v2")).unwrap();
    assert_ne!(r1, r2);
    assert_eq!(materialize_tree(&dir, D1, &toy("v1")).unwrap(), r1);
    assert_eq!(
        std::fs::read(r1.join("kernels/DEVICES.toml")).unwrap(),
        b"# v1\n"
    );
    assert_eq!(
        std::fs::read(r2.join("kernels/DEVICES.toml")).unwrap(),
        b"# v2\n"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

/// 2026-10-03: An interrupted unpack (a partial directory left under its temporary name) never
/// looks complete: the next call unpacks in full under the final name. A partial tree under the
/// final name (a crash that left one there anyway) fails verification and is replaced.
#[test]
fn an_interrupted_unpack_never_looks_complete() {
    let dir = scratch("interrupt");
    let want = toy("v1");
    let partial = dir.join(format!("{D1}.partial-{}", std::process::id() + 1));
    std::fs::create_dir_all(partial.join("kernels")).unwrap();
    std::fs::write(partial.join("kernels/DEVICES.toml"), b"# v1\n").unwrap();
    assert!(!dir.join(D1).exists(), "nothing under the final name");
    let root = materialize_tree(&dir, D1, &want).unwrap();
    assert_eq!(
        std::fs::read(root.join("kernels/gb10/HARDWARE.toml")).unwrap(),
        b"[memory]\n"
    );

    std::fs::remove_file(root.join("kernels/gb10/HARDWARE.toml")).unwrap();
    let again = materialize_tree(&dir, D1, &want).unwrap();
    assert_eq!(
        std::fs::read(again.join("kernels/gb10/HARDWARE.toml")).unwrap(),
        b"[memory]\n"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

/// 2026-10-02: A path that would escape the tree, or a digest that is not one, is refused.
#[test]
fn escaping_paths_and_bad_digests_are_refused() {
    for bad in ["", "/etc/passwd", "../x", "kernels/../../x", "./kernels"] {
        assert!(check_path(bad).is_err(), "`{bad}` accepted");
    }
    assert!(check_path("kernels/gb10/HARDWARE.toml").is_ok());
    let dir = scratch("digest");
    for bad in ["", "../x", "zz", &"g".repeat(64)] {
        assert!(
            materialize_tree(&dir, bad, &toy("v1")).is_err(),
            "`{bad}` accepted"
        );
    }
    let escaping = vec![("../escape".to_string(), b"x".to_vec())];
    assert!(materialize_tree(&dir, D1, &escaping).is_err());
    assert!(!dir.join("escape").exists());
    let _ = std::fs::remove_dir_all(&dir);
}
