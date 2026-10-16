// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `met circuit affected`: which registered models a change reaches through the
//! kernels they run, from the KERNEL_FAMILIES.toml `files` and the INSTANCES.toml plans at the
//! head revision.
//!
//! Owner: server CLI.
//! Invariants:
//! - Nothing here decides: `metrale_circuit::venn::affected_repo::affected_in` does, over the
//!   head revision read through `gate::kernel_map::GitRepo`; the working tree is never read.
//! - The change is `base...head`: what `head` added since it and `base` last shared a commit.

use anyhow::{Result, anyhow};
use metrale_bench::gate::kernel_map::{GitRepo, changed_paths};
use metrale_circuit::venn::affected::Affected;
use metrale_circuit::venn::affected_repo::affected_in;

use super::circuit_venn::find_root;

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

pub(crate) fn run(a: super::circuit_args::CircuitAffectedArgs) -> Result<()> {
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
