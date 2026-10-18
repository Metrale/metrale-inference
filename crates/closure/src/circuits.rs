// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The circuit files every target of a hardware hashes as configs: each
//! `kernels/circuits/**/*.toml` (the op graphs, precision tables and instances) and every
//! `kernels/<class>/common/FUSIONS.toml` the hardware's rule set is resolved from. They decide
//! which kernel runs for which op, so a record's closure must move when they do.
//!
//! Owner: metrale-closure (shared by the kernels build script and the bench gate, so both
//! hash the same list).
//! Invariants:
//! - The list is sorted and holds only files that exist; `kernels/circuits/plans/` holds
//!   rendered `.txt` plans, which are derived and not listed.
//! - 2026-09-30: [`fusions_chain`] follows the rule inheritance the planner resolves
//!   (`metrale_circuit::hardware::class::class_rules`): a class's own FUSIONS.toml and, through
//!   its `inherits = "<class>"`, each ancestor's; a class with no FUSIONS.toml takes its
//!   HARDWARE.toml `[hardware] inherits` parent's rules. So editing gb10's rules moves the
//!   closure of hopper and b300 (FUSIONS `inherits`) and of b200 (HARDWARE `inherits`, no rules
//!   of its own). A cross-crate test in metrale-circuit holds the two resolutions equal on the
//!   real tree.

use std::path::{Path, PathBuf};

use crate::layout_manifest::{LayoutError, hardware_raw};

/// 2026-09-28: The circuit configs of `hardware` under `root`.
pub fn circuit_configs(root: &Path, hardware: &str) -> Result<Vec<PathBuf>, LayoutError> {
    let mut out = Vec::new();
    tomls_under(&root.join("kernels").join("circuits"), &mut out);
    out.extend(fusions_chain(root, hardware)?);
    out.sort();
    Ok(out)
}

/// 2026-09-30: The FUSIONS.toml files `hardware` plans with, base first; empty when neither the
/// class nor an ancestor declares rules. A cycle, an unreadable FUSIONS.toml or an `inherits`
/// that is not a class name is an error, not a shorter list: a list that stopped early would
/// leave an ancestor's rules out of the hash.
pub fn fusions_chain(root: &Path, hardware: &str) -> Result<Vec<PathBuf>, LayoutError> {
    let kernels = root.join("kernels");
    let mut seen: Vec<String> = Vec::new();
    let mut files: Vec<PathBuf> = Vec::new();
    let mut next = Some(hardware.to_string());
    while let Some(class) = next.take() {
        let fusions = kernels.join(&class).join("common").join("FUSIONS.toml");
        if seen.contains(&class) {
            return Err(LayoutError::Manifest {
                path: fusions,
                message: format!("FUSIONS inheritance cycle through `{class}`"),
            });
        }
        seen.push(class.clone());
        if !fusions.is_file() {
            next = hardware_raw(&kernels, &class)?.inherits;
            continue;
        }
        let text = std::fs::read_to_string(&fusions).map_err(|e| LayoutError::Manifest {
            path: fusions.clone(),
            message: e.to_string(),
        })?;
        let table: toml::Table = toml::from_str(&text).map_err(|e| LayoutError::Manifest {
            path: fusions.clone(),
            message: format!("bad TOML: {e}"),
        })?;
        next = match table.get("inherits") {
            None => None,
            Some(v) => Some(
                v.as_str()
                    .ok_or_else(|| LayoutError::Manifest {
                        path: fusions.clone(),
                        message: "`inherits` is not a class name".into(),
                    })?
                    .to_string(),
            ),
        };
        files.push(fusions);
    }
    files.reverse();
    Ok(files)
}

/// 2026-09-28: The directories [`circuit_configs`] reads, for `cargo:rerun-if-changed`, so a
/// circuit file added or removed reruns the build script too.
///
/// 2026-10-17: Only the largest subtrees of `kernels/circuits` that hold nothing but `.toml`
/// files (today `blocks/`, `checkpoints/`, `precision/`), sorted. Cargo watches a directory
/// recursively, so watching `kernels/circuits` itself reran the kernels build (and recompiled
/// every kernel) whenever the derived plans under `plans/` or a design `.md` was regenerated,
/// though neither is hashed. Every existing circuit `.toml` keeps its own per-file
/// `rerun-if-changed` from the closure attestation, so an edit still reruns; these directories
/// add "a new `.toml` appeared". The embedding is unchanged: the same files are hashed.
///
/// 2026-10-17: Why not a finer trigger: cargo has no non-recursive directory watch, and a
/// watch on a directory that also holds `.txt`/`.md` (the root, `plans/`, `venn/`) is the
/// trigger this narrows away. So a NEW `.toml` placed directly in one of those directories
/// waits for the next rerun of the build script (any kernel, manifest or env change). That
/// fails closed, not open: the gate recomputes each closure from the working tree, the new file
/// is in it and not in the binary's attestation, so `gate::closure::excuses` refuses the
/// record. A subtree that gains a `.md` or `.txt` drops out of this list on the next rerun.
pub fn circuit_dirs(root: &Path) -> Vec<PathBuf> {
    let circuits = root.join("kernels").join("circuits");
    let mut out = Vec::new();
    if toml_only_subtrees(&circuits, &mut out) {
        out.push(circuits);
    }
    out.sort();
    out
}

/// 2026-10-17: Whether `dir` holds only `.toml` files, at any depth. When it does not, the
/// largest toml-only subtrees under it are appended to `out`; when it does, nothing is, and the
/// caller decides whether `dir` itself is one. An unreadable directory counts as not toml-only,
/// which narrows the watch; the circuit files in it keep their per-file triggers.
fn toml_only_subtrees(dir: &Path, out: &mut Vec<PathBuf>) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    let mut whole = true;
    let mut kept = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if toml_only_subtrees(&path, out) {
                kept.push(path);
            } else {
                whole = false;
            }
        } else if path.extension().is_none_or(|e| e != "toml") {
            whole = false;
        }
    }
    if !whole {
        out.extend(kept);
    }
    whole
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
