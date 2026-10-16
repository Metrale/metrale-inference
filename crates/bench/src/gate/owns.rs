// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: A model's `owns` list: the paths outside `kernels/` that only that model runs,
//! declared at the top of its `BENCH.toml` (`owns = ["crates/config/src/parsers/foo"]`). A
//! change under an owned path reaches that model alone; every other path outside `kernels/`
//! reaches every model (owner ruling Q1, 2026-10-15).
//!
//! Owner: bench gate.
//! Invariants:
//! - Pure: [`from_bench_toml`] parses text, [`owners_of`] matches a path.
//! - An entry is a repository-relative path, matched per component (`coverage::under`), so
//!   `crates/a` owns `crates/a/b.rs` and not `crates/ab.rs`. An empty, absolute, `.`/`..`,
//!   glob, trailing-slash or `kernels/` entry is refused: `kernels/` paths are mapped by the
//!   kernel map, never by a list.
//! - A file whose list is refused owns nothing, so its paths reach every model (fail-closed).

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

/// 2026-10-16: Model directory → the paths it owns, from every BENCH.toml of one tree.
pub type Owners = BTreeMap<String, BTreeSet<String>>;

#[derive(Deserialize)]
struct OwnsOnly {
    #[serde(default)]
    owns: Vec<String>,
}

fn refusal(entry: &str) -> Option<&'static str> {
    let parts: Vec<&str> = entry.split('/').collect();
    if entry.is_empty() || entry.starts_with('/') || entry.ends_with('/') {
        Some("must be a non-empty relative path without a trailing slash")
    } else if parts
        .iter()
        .any(|p| p.is_empty() || *p == "." || *p == "..")
    {
        Some("must not hold an empty, `.` or `..` component")
    } else if entry.contains(['*', '?', '[', ']', '{', '}', '\\']) {
        Some("is matched per path component and takes no glob or escape")
    } else if parts[0] == "kernels" {
        Some("is under kernels/, which the kernel map assigns to models")
    } else {
        None
    }
}

/// 2026-10-16: The `owns` list of one BENCH.toml's text, refused whole if any entry is invalid.
pub fn from_bench_toml(text: &str) -> Result<Vec<String>, String> {
    let parsed: OwnsOnly = toml::from_str(text).map_err(|e| e.to_string())?;
    for entry in &parsed.owns {
        if let Some(why) = refusal(entry) {
            return Err(format!("owns entry {entry:?} {why}"));
        }
    }
    Ok(parsed.owns)
}

/// 2026-10-16: The models that own `path`; empty when none does.
pub fn owners_of<'a>(path: &str, owners: &'a Owners) -> BTreeSet<&'a str> {
    owners
        .iter()
        .filter(|(_, entries)| entries.iter().any(|e| super::coverage::under(path, e)))
        .map(|(model, _)| model.as_str())
        .collect()
}

#[cfg(test)]
#[path = "owns_tests.rs"]
mod owns_tests;
