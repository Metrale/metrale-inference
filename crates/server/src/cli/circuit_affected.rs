// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `met circuit affected`: which registered models a change reaches through the
//! kernels they run, from the KERNEL_FAMILIES.toml `files` and the INSTANCES.toml plans at the
//! head revision.
//!
//! Owner: server CLI.
//! The `///` comments on clap items below are the `--help` text and carry no date.
//! Invariants:
//! - Nothing here decides: `metrale_circuit::venn::affected_repo::affected_in` does, over the
//!   head revision read through `gate::kernel_map::GitRepo`; the working tree is never read.
//! - The change is `base...head`: what `head` added since it and `base` last shared a commit.

use std::path::PathBuf;

use anyhow::{Result, anyhow};
use metrale_bench::gate::kernel_map::{GitRepo, changed_paths};
use metrale_circuit::venn::affected::Affected;
use metrale_circuit::venn::affected_repo::affected_in;

use super::circuit_venn::find_root;

/// The models a change reaches through the kernel families it edits. A path under kernels/
/// that no family lists is unmapped and reaches every model; a path outside kernels/ is not
/// scoped by this map (the gate treats it as reaching every model).
#[derive(clap::Args, Debug)]
pub struct CircuitAffectedArgs {
    /// The revision the change is measured from (a branch's merge target).
    #[arg(long)]
    pub base: String,
    /// The revision whose kernels and plans are read, and whose change is reported.
    #[arg(long)]
    pub head: String,
    /// Print JSON: {models: [{model, hw, checkpoint, reason: [{file, family, instance,
    /// assumed}]}], unmapped, unused, non_kernel, assumed}.
    #[arg(long)]
    pub json: bool,
    /// Repository root; by default the nearest directory above the working directory that has
    /// kernels/circuits/INSTANCES.toml.
    #[arg(long)]
    pub root: Option<PathBuf>,
}

/// 2026-10-15: The plain-text listing.
pub(crate) fn text(a: &Affected) -> String {
    let mut out = String::new();
    let list = |out: &mut String, label: &str, paths: &[String]| {
        if !paths.is_empty() {
            out.push_str(&format!("{label} ({}):\n", paths.len()));
            for p in paths {
                out.push_str(&format!("  {p}\n"));
            }
        }
    };
    if a.models.is_empty() {
        out.push_str("no model is reached through a kernel family\n");
    }
    for m in &a.models {
        out.push_str(&format!("{}/{} ({})\n", m.hw, m.model, m.checkpoint));
        for h in &m.reason {
            let assumed = if h.assumed { " [usage unknown]" } else { "" };
            out.push_str(&format!(
                "  {} -> {} -> {}{assumed}\n",
                h.file, h.family, h.instance
            ));
        }
    }
    list(&mut out, "unmapped, reaches every model", &a.unmapped);
    list(&mut out, "listed by a family no instance runs", &a.unused);
    list(
        &mut out,
        "outside kernels/, reaches every model",
        &a.non_kernel,
    );
    for (recipe, why) in &a.assumed {
        out.push_str(&format!(
            "usage unknown, runs every family: {recipe} — {why}\n"
        ));
    }
    out
}

pub(crate) fn run(a: CircuitAffectedArgs) -> Result<()> {
    let root = match a.root {
        Some(r) => r,
        None => find_root(&std::env::current_dir()?)?,
    };
    let changed = changed_paths(&root, &a.base, &a.head).map_err(|e| anyhow!(e))?;
    let repo = GitRepo { root, rev: a.head };
    let affected = affected_in(&repo, &changed).map_err(|e| anyhow!("{e}"))?;
    if a.json {
        println!("{}", serde_json::to_string_pretty(&affected)?);
    } else {
        print!("{}", text(&affected));
    }
    Ok(())
}
