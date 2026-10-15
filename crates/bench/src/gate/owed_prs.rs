// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Which certification PRs a merge to main opens: one per model it left owed
//! (`check_models::Owed`), on branch `cert/<model>/<sha7>` carrying the manifest
//! `.benchmarks/owed/<model>-<sha>.toml`. The post-merge workflow runs the `owed_certs` binary
//! and only executes the actions this returns.
//!
//! Owner: bench gate.
//! Invariants:
//! - Pure: the caller passes what is owed, the merge sha and the branches and PR heads that
//!   already exist.
//! - Idempotent: a model whose branch and PR both exist yields nothing; one whose branch exists
//!   without a PR (a run that died between the two calls) yields the PR alone.
//! - One PR per model directory: its gates, hardware and checkpoints are listed in one manifest.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use super::check_models::Owed;

/// 2026-10-15: One certification PR to open, and whether its branch must be created first.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OwedPr {
    /// 2026-10-15: Model directory (`kernels/<hw>/<model>/`).
    pub model: String,
    pub branch: String,
    pub create_branch: bool,
    pub manifest_path: String,
    pub manifest: String,
    pub title: String,
    pub body: String,
}

/// 2026-10-15: The merge sha's first seven characters, as the branch names it.
fn sha7(sha: &str) -> &str {
    sha.get(..7).unwrap_or(sha)
}

fn quoted(s: &str) -> String {
    toml::Value::String(s.to_string()).to_string()
}

fn manifest(model: &str, merge_sha: &str, owed: &[&Owed]) -> String {
    let mut out = format!(
        "# Certification owed after {merge_sha}: a passing record of each entry below,\n\
         # measured at or after that commit, settles it.\n\
         schema = 1\nmerge_sha = {}\nmodel = {}\n",
        quoted(merge_sha),
        quoted(model)
    );
    for o in owed {
        out.push_str(&format!(
            "\n[[owed]]\ngate = {}\nhw = {}\ncheckpoint = {}\n",
            quoted(&o.gate),
            quoted(&o.subject.hw),
            quoted(&o.subject.checkpoint)
        ));
    }
    out
}

fn body(model: &str, merge_sha: &str, owed: &[&Owed]) -> String {
    let mut out = format!(
        "{merge_sha} changed code `{model}` runs and merged certified on another model, so \
         these gates are open for `{model}`. Commit a passing record of each, measured at or \
         after {merge_sha}, to this branch:\n\n| gate | hw | checkpoint |\n|---|---|---|\n"
    );
    for o in owed {
        out.push_str(&format!(
            "| `{}` | {} | `{}` |\n",
            o.gate, o.subject.hw, o.subject.checkpoint
        ));
    }
    out
}

/// 2026-10-15: The certification PRs to open for `owed` after the merge `merge_sha`.
pub fn owed_prs(
    owed: &[Owed],
    merge_sha: &str,
    branches: &BTreeSet<String>,
    pr_heads: &BTreeSet<String>,
) -> Vec<OwedPr> {
    let mut by_model: BTreeMap<&str, Vec<&Owed>> = BTreeMap::new();
    for o in owed {
        by_model
            .entry(o.subject.model.as_str())
            .or_default()
            .push(o);
    }
    let mut out = Vec::new();
    for (model, mut list) in by_model {
        list.sort();
        list.dedup();
        let branch = format!("cert/{model}/{}", sha7(merge_sha));
        if pr_heads.contains(&branch) {
            continue;
        }
        out.push(OwedPr {
            model: model.to_string(),
            create_branch: !branches.contains(&branch),
            manifest_path: format!(".benchmarks/owed/{model}-{merge_sha}.toml"),
            manifest: manifest(model, merge_sha, &list),
            title: format!(
                "cert: {model} owes {} gate(s) after {}",
                list.len(),
                sha7(merge_sha)
            ),
            body: body(model, merge_sha, &list),
            branch,
        });
    }
    out
}

#[cfg(test)]
#[path = "owed_prs_tests.rs"]
mod owed_prs_tests;
