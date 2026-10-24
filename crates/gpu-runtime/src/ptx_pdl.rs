// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The kernel entries of a PTX module that wait on their grid dependencies
//! (`griddepcontrol.wait` in the entry's own body), read from its text. Only such an entry may
//! be launched with programmatic stream serialization (`registry_pdl`): the attribute lets the
//! kernel start before its predecessor finishes, and the wait is what holds back every read of
//! the predecessor's output. An entry without it would race.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - An entry is listed exactly when the text between its `.entry` line and the next `.entry`
//!   or `.func` line (or the end of the module) holds a `griddepcontrol.wait` instruction. A wait
//!   inside a called `.func` does not count (the entry is not listed: the safe side).
//! - Pure: no driver calls.

use std::collections::HashSet;

/// 2026-10-21: The entries of `ptx` whose body holds `griddepcontrol.wait`.
pub(crate) fn waiting_entries(ptx: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    let mut current: Option<String> = None;
    for line in ptx.lines() {
        let code = line.split("//").next().unwrap_or("").trim();
        if let Some(name) = entry_name(code) {
            current = Some(name);
        } else if is_func_header(code) {
            current = None;
        } else if code.starts_with("griddepcontrol.wait")
            && let Some(name) = &current
        {
            out.insert(name.clone());
        }
    }
    out
}

/// 2026-10-21: The entry a line opens: `[.visible|.weak] .entry name(`.
fn entry_name(code: &str) -> Option<String> {
    let mut toks = code.split_whitespace();
    let mut t = toks.next()?;
    while matches!(t, ".visible" | ".weak" | ".extern") {
        t = toks.next()?;
    }
    if t != ".entry" {
        return None;
    }
    let name = toks.next()?.split('(').next()?;
    (!name.is_empty()).then(|| name.to_string())
}

/// 2026-10-21: Whether a line opens a `.func` (its body is not the previous entry's).
fn is_func_header(code: &str) -> bool {
    code.split_whitespace()
        .find(|t| !matches!(*t, ".visible" | ".weak" | ".extern"))
        .is_some_and(|t| t == ".func")
}

#[cfg(test)]
#[path = "ptx_pdl_tests.rs"]
mod tests;
