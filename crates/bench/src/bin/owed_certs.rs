// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Print the certification PRs a change owes, as JSON
//! (`owed_certs --root . --base <before> --head <merge sha> --branches f --pr-heads f`).
//! The post-merge workflow executes the `prs` it prints; the PR dry-run job only prints them.
//!
//! | stream | meaning |
//! |---|---|
//! | stdout | `{head, kernel_map, statuses, owed, standings, manifests, prs}` |
//! | stderr | the cause (exit != 0) |
//! | exit 0 | an answer was computed, possibly no PRs |
//! | exit 1 | no answer: bad invocation, unreadable list file, or no `head` in this clone |
//!
//! `--branches` and `--pr-heads` are files of branch names, one per line: the existing
//! `cert/*` branches and the head refs of every `cert/*` PR, open or closed.
//!
//! Owner: bench gate.
//! Invariants:
//! - Decides nothing: `check_models` names what is owed and `owed_prs` the PRs to open.
//! - The working tree is `head` checked out; its records and kernel map are read from it.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, bail};
use metrale_bench::gate::GateStatus;
use metrale_bench::gate::check_models::check_models;
use metrale_bench::gate::kernel_map::{FsRepo, load};
use metrale_bench::gate::model_scope::KernelMap;
use metrale_bench::gate::owed_prs::owed_prs;

fn arg(name: &str) -> Result<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .with_context(|| format!("{name} is required"))
}

fn lines(path: &str) -> Result<BTreeSet<String>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect())
}

fn main() -> Result<()> {
    let root = std::path::PathBuf::from(arg("--root")?);
    let base = arg("--base")?;
    let head = arg("--head")?;
    let branches = lines(&arg("--branches")?)?;
    let pr_heads = lines(&arg("--pr-heads")?)?;
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(&root)
        .args(["rev-parse", "--verify", &format!("{head}^{{commit}}")])
        .output()
        .context("git rev-parse")?;
    if !out.status.success() {
        bail!("{head} is not a commit in {}", root.display());
    }
    let head = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let map = load(&FsRepo { root: root.clone() });
    let checked = check_models(&root, &head, &base, &map);
    let statuses: BTreeMap<&str, String> = checked
        .statuses
        .iter()
        .map(|(id, s)| {
            let s = match s {
                GateStatus::Pass => "pass".to_string(),
                GateStatus::Fail(p) => format!("fail: {}", p.join("; ")),
                GateStatus::Missing(why) => format!("missing: {why}"),
            };
            (id.as_str(), s)
        })
        .collect();
    let prs = owed_prs(&checked.owed, &head, &branches, &pr_heads).map_err(anyhow::Error::msg)?;
    let kernel_map = match &map {
        KernelMap::Derived { .. } => "derived".to_string(),
        KernelMap::Unavailable(why) => format!("unavailable (every model reached): {why}"),
    };
    let json = serde_json::json!({
        "head": head,
        "kernel_map": kernel_map,
        "statuses": statuses,
        "owed": checked.owed,
        "standings": checked.standings,
        "manifests": checked.manifests,
        "prs": prs,
    });
    println!("{}", serde_json::to_string_pretty(&json)?);
    Ok(())
}
