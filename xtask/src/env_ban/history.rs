// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The git I/O behind the shrink-only check: the merge-base of a base revision and
//! `HEAD`, the allowlist there, and every commit since that changed it. A git
//! failure (a shallow clone without the merge-base, an unknown revision) is an error, never an
//! empty history.

use super::ALLOWLIST;
use super::additions::{Change, TRAILER};
use std::path::Path;
use std::process::Command;

/// 2026-10-17: What the additions judge needs from the repository.
pub struct History {
    pub merge_base: String,
    pub base_list: Option<String>,
    pub changes: Vec<Change>,
}

fn git(root: &Path, args: &[&str]) -> Result<String, String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|e| format!("git {}: cannot run: {e}", args.join(" ")))?;
    if !out.status.success() {
        return Err(format!(
            "git {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    String::from_utf8(out.stdout)
        .map_err(|e| format!("git {}: non-UTF-8 output: {e}", args.join(" ")))
}

/// 2026-10-17: The allowlist at `rev`, or `None` when the file is absent there.
fn list_at(root: &Path, rev: &str) -> Result<Option<String>, String> {
    if git(root, &["ls-tree", "--name-only", rev, "--", ALLOWLIST])?
        .trim()
        .is_empty()
    {
        return Ok(None);
    }
    git(root, &["show", &format!("{rev}:{ALLOWLIST}")]).map(Some)
}

/// 2026-10-17: The history of the allowlist between the merge-base of `base` and `HEAD`.
pub fn read(root: &Path, base: &str) -> Result<History, String> {
    let merge_base = git(root, &["merge-base", base, "HEAD"])
        .map_err(|e| {
            format!("{e}\nxtask env-ban: no merge-base of `{base}` and HEAD (a shallow clone needs fetch-depth: 0)")
        })?
        .trim()
        .to_owned();
    let range = format!("{merge_base}..HEAD");
    let shas = git(
        root,
        &[
            "rev-list",
            "--reverse",
            "--topo-order",
            &range,
            "--",
            ALLOWLIST,
        ],
    )?;
    let trailer = format!("--format=%(trailers:key={TRAILER},valueonly,separator=%x20)");
    let mut changes = Vec::new();
    for sha in shas.lines().map(str::trim).filter(|s| !s.is_empty()) {
        changes.push(Change {
            sha: sha.to_owned(),
            before: parents_list(root, sha)?,
            after: list_at(root, sha)?,
            reason: git(root, &["log", "-1", &trailer, sha])?.trim().to_owned(),
        });
    }
    Ok(History {
        base_list: list_at(root, &merge_base)?,
        merge_base,
        changes,
    })
}

/// 2026-10-17: The union of the allowlist across every parent of `sha`, or `None` when no
/// parent has it. A merge therefore introduces only what none of its sides had, so the
/// synthetic merge a pull request is checked out as never stands in for the commit that
/// really added an entry.
fn parents_list(root: &Path, sha: &str) -> Result<Option<String>, String> {
    let line = git(root, &["rev-list", "--parents", "-n", "1", sha])?;
    let mut lists = Vec::new();
    for parent in line.split_whitespace().skip(1) {
        lists.extend(list_at(root, parent)?);
    }
    Ok((!lists.is_empty()).then(|| lists.join("\n")))
}

#[cfg(test)]
#[path = "history_tests.rs"]
mod tests;
