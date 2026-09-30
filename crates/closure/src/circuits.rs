// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The circuit files every target of a hardware hashes as configs: each
//! `kernels/circuits/**/*.toml` (the op graphs, precision tables and instances) and the
//! hardware's `kernels/<hw>/common/FUSIONS.toml`. They decide which kernel runs for which op,
//! so a record's closure must move when they do.
//!
//! Owner: metrale-closure (shared by the kernels build script and the bench gate, so both
//! hash the same list).
//! Invariants:
//! - The list is sorted and holds only files that exist; `kernels/circuits/plans/` holds
//!   rendered `.txt` plans, which are derived and not listed.

use std::path::{Path, PathBuf};

/// 2026-09-28: The circuit configs of `hardware` under `root`.
pub fn circuit_configs(root: &Path, hardware: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    tomls_under(&root.join("kernels").join("circuits"), &mut out);
    let fusions = root
        .join("kernels")
        .join(hardware)
        .join("common")
        .join("FUSIONS.toml");
    if fusions.is_file() {
        out.push(fusions);
    }
    out.sort();
    out
}

/// 2026-09-28: The directories [`circuit_configs`] reads, for `cargo:rerun-if-changed`, so a
/// circuit file added or removed reruns the build script too.
pub fn circuit_dirs(root: &Path) -> Vec<PathBuf> {
    vec![root.join("kernels").join("circuits")]
}

fn tomls_under(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            tomls_under(&path, out);
        } else if path.extension().is_some_and(|e| e == "toml") {
            out.push(path);
        }
    }
}

#[cfg(test)]
#[path = "circuits_tests.rs"]
mod circuits_tests;
