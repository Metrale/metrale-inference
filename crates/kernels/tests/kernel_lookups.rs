// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Checks that every kernel the workspace's Rust code looks up by
//! name, `module::func` with both written as literals, is declared by that
//! module in at least one target of the `kernels/` tree.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! A lookup names a module (a file stem, or its `[modules]` rename) and an
//! entry point. Nothing ties the two strings to the tree at compile time: a
//! renamed, removed or un-`use`d source is found only when a lookup fails at
//! boot on a GPU. This test resolves every target with
//! `metrale_closure::layout`, reads each module's entry points with the
//! scanner build.rs uses (`build_shadow.rs`), and fails on a lookup no target
//! can answer. It checks existence in some target, not in every target that
//! reaches the call: which targets reach a call is a runtime property.
//!
//! The lookups read are `.kernel(`, `try_kernel(`, `try_target_kernel(` and
//! `gated(`, whose last two arguments are the module and the entry point. A
//! module given as a `SCREAMING_CASE` constant is resolved through the
//! `const NAME: &str = "..."` items of the same crate. Test-only files and
//! inline `#[cfg(test)]` modules are skipped: their lookups go to a mock.
//! 2026-10-07: The scan is `support/lookup_scan.rs`, shared with
//! `strix_hip_laguna_int4.rs`.

// 2026-09-26: Only `entry_points` is used here; the shadow comparison is
// `kernel_shadow_detector.rs`'s.
#[allow(dead_code)]
#[path = "../build_shadow.rs"]
mod build_shadow;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use metrale_closure::layout::{discover, walk};

#[path = "support/lookup_scan.rs"]
mod lookup_scan;
use lookup_scan::lookups;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/kernels is two levels below the workspace root")
        .to_path_buf()
}

/// 2026-09-26: Every `(module, entry point)` some target declares.
fn declared_kernels(root: &Path) -> BTreeSet<(String, String)> {
    let mut out = BTreeSet::new();
    // 2026-09-26: Targets share most sources; scan each file once.
    let mut scanned: BTreeMap<PathBuf, BTreeSet<String>> = BTreeMap::new();
    for target in walk(root).expect("the tree resolves") {
        let layout = discover(root, &target).unwrap_or_else(|e| panic!("{target}: {e}"));
        let mut module_of: BTreeMap<String, String> = BTreeMap::new();
        for config in layout.configs() {
            let Ok(text) = std::fs::read_to_string(&config) else {
                continue;
            };
            let toml: toml::Value =
                toml::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", config.display()));
            for (stem, name) in toml
                .get("modules")
                .and_then(|m| m.as_table())
                .into_iter()
                .flatten()
            {
                if let Some(name) = name.as_str() {
                    module_of.insert(stem.clone(), name.to_string());
                }
            }
        }
        for (stem, entry) in layout.modules() {
            let module = module_of.get(&stem).cloned().unwrap_or(stem);
            let funcs = scanned
                .entry(entry.source.clone())
                .or_insert_with(|| build_shadow::entry_points(&entry.source));
            for func in funcs.iter() {
                out.insert((module.clone(), func.clone()));
            }
        }
    }
    out
}

#[test]
fn every_literal_kernel_lookup_names_a_kernel_some_target_declares() {
    let root = workspace_root();
    let declared = declared_kernels(&root);
    let lookups = lookups(&root);
    assert!(
        lookups.len() > 500,
        "only {} lookups found — did the lookup API or the crate layout change?",
        lookups.len()
    );
    let mut missing: Vec<String> = lookups
        .iter()
        .filter(|l| {
            !l.modules
                .iter()
                .any(|m| declared.contains(&(m.clone(), l.func.clone())))
        })
        .map(|l| {
            let m: Vec<&str> = l.modules.iter().map(String::as_str).collect();
            format!("{}: {}::{}", l.site, m.join("|"), l.func)
        })
        .collect();
    missing.sort();
    assert!(
        missing.is_empty(),
        "{} kernel lookup(s) name a module::func that no target under kernels/ declares \
         (renamed or removed source, a lost `[sources] use`, or a `[modules]` rename):\n  {}",
        missing.len(),
        missing.join("\n  ")
    );
}
