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
use super::owed_manifest::{Manifest, Pair, SCHEMA, path, render};

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

fn manifest(model: &str, merge_sha: &str, owed: &[&Owed]) -> Result<String, String> {
    render(&Manifest {
        schema: SCHEMA,
        merge_sha: merge_sha.to_string(),
        model: model.to_string(),
        owed: owed
            .iter()
            .map(|o| Pair {
                gate: o.gate.clone(),
                hw: o.subject.hw.clone(),
                checkpoint: o.subject.checkpoint.clone(),
            })
            .collect(),
    })
}

fn body(
    model: &str,
    merge_sha: &str,
    owed: &[&Owed],
    battles: Option<&BTreeSet<String>>,
) -> String {
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
    for id in battles.into_iter().flatten() {
        out.push_str(&format!(
            "\nThese gates hold cells of the battle `{id}`: once the records land, \
             `met benchmark battle {id} --model {model}` shows whether `{model}` still wins it.\n"
        ));
    }
    out.push_str(&format!(
        "\nOnce every row passes, delete `{}` in this branch: the gate \
         check fails while a settled manifest is committed.\n",
        path(model, merge_sha)
    ));
    out
}

/// 2026-10-15: The certification PRs to open for `owed` after the merge `merge_sha`; an error
/// only when a manifest cannot be rendered. 2026-10-16: `battles` names, per model directory,
/// the battles whose cells it owes (`battle::owed_battles`); each is named in the PR body.
pub fn owed_prs(
    owed: &[Owed],
    merge_sha: &str,
    branches: &BTreeSet<String>,
    pr_heads: &BTreeSet<String>,
    battles: &BTreeMap<String, BTreeSet<String>>,
) -> Result<Vec<OwedPr>, String> {
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
            manifest_path: path(model, merge_sha),
            manifest: manifest(model, merge_sha, &list)?,
            title: format!(
                "cert: {model} owes {} gate(s) after {}",
                list.len(),
                sha7(merge_sha)
            ),
            body: body(model, merge_sha, &list, battles.get(model)),
            branch,
        });
    }
    Ok(out)
}

#[cfg(test)]
#[path = "owed_prs_tests.rs"]
mod owed_prs_tests;
