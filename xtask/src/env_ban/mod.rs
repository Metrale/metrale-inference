// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `cargo xtask env-ban [--base <rev>]`: the process environment is read in one
//! place, the `ProcessEnv` host. `std::env::{var, var_os, vars, vars_os}`, `levers::{var,
//! var_os}` and the `metrale_gpu_sys::env` helpers, reached under any import name, may appear
//! only in that file, in build scripts, and in the files the checked-in allowlist names
//! (today's offenders). With `--base`, an entry added since the merge-base with `<rev>` fails
//! unless the commit that added it carries an `Env-Ban-Allow: <reason>` trailer; without it,
//! only the tree is judged and the run says so.
//!
//! Layout: `lex` tokenizes, `resolve` finds reads, `judge` gives the tree verdict, `additions`
//! the shrink-only verdict, `walk` and `history` are the only I/O, `probes` is the selftest.

mod additions;
mod history;
mod judge;
mod lex;
mod probes;
mod resolve;
mod walk;

use std::path::Path;

/// 2026-10-17: The allowlist, relative to the scan root; CODEOWNERS owns it.
pub const ALLOWLIST: &str = "xtask/env-ban-allow.txt";

/// 2026-10-17: The one sanctioned reader of the process environment.
pub const HOST_ENV: &str = "crates/io-hosts/std/src/env.rs";

/// 2026-10-17: The allowlist text under `root` and the tree verdict. A missing allowlist is an
/// error.
fn check(root: &Path) -> Result<(String, Vec<judge::Source>, Vec<judge::Violation>), String> {
    let list_path = root.join(ALLOWLIST);
    let allowlist = std::fs::read_to_string(&list_path)
        .map_err(|e| format!("{ALLOWLIST}: cannot read {}: {e}", list_path.display()))?;
    let sources = walk::scan(root)?;
    let violations = judge::judge(&sources, &allowlist);
    Ok((allowlist, sources, violations))
}

/// 2026-10-17: How many additions since the merge-base with `base` lack a reason; prints each
/// addition and how it is accounted for.
fn shrink_only(root: &Path, base: &str, allowlist: &str) -> Result<usize, String> {
    let h = history::read(root, base)?;
    let v = additions::judge(h.base_list.as_deref(), &h.changes, allowlist);
    if let Some((sha, n)) = &v.inventory {
        println!(
            "xtask env-ban: {ALLOWLIST} is absent at the merge-base {}; {sha} creates it, and its \
             {n} entries are the inventory",
            h.merge_base
        );
    }
    for (added, reason) in &v.explained {
        println!(
            "xtask env-ban: `{}` added by {} ({}: {reason})",
            added.entry,
            added.sha.as_deref().unwrap_or("-"),
            additions::TRAILER
        );
    }
    for added in &v.unexplained {
        eprintln!("{added}");
    }
    Ok(v.unexplained.len())
}

/// 2026-10-17: `cargo xtask env-ban` on the repository at `root`; `base` turns on the
/// shrink-only check.
pub fn gate(root: &Path, base: Option<&str>) -> Result<(), String> {
    let (allowlist, sources, violations) = check(root)?;
    for v in &violations {
        eprintln!("{v}");
    }
    let unexplained = match base {
        Some(base) => shrink_only(root, base, &allowlist)?,
        None => {
            println!("xtask env-ban: no --base given; the shrink-only check did not run");
            0
        }
    };
    let total = violations.len().saturating_add(unexplained);
    if total > 0 {
        return Err(format!("xtask env-ban: {total} violation(s)"));
    }
    let readers = sources
        .iter()
        .filter(|s| s.kind == judge::Kind::Ordinary && !s.reads.is_empty())
        .count();
    println!(
        "xtask env-ban: {} sources scanned; {readers} allowlisted files still read the environment",
        sources.len()
    );
    Ok(())
}

/// 2026-10-17: The env-ban probes under `fixtures`; returns how many did not end as intended.
pub fn selftest(fixtures: &Path) -> Result<usize, String> {
    probes::run(fixtures)
}
