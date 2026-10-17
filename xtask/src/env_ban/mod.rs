// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `cargo xtask env-ban`: the process environment is read in one place, the
//! `ProcessEnv` host. `std::env::{var, var_os, vars, vars_os}` and `levers::{var, var_os}` may
//! appear only in that file, in build scripts, and in the files the checked-in allowlist names
//! (today's offenders, which may only shrink).
//!
//! Layout: `lex` finds reads in text, `judge` gives the verdict, `walk` reads the tree (the
//! only I/O), `probes` is the selftest.

mod judge;
mod lex;
mod probes;
mod walk;

use std::path::Path;

/// 2026-10-17: The allowlist, relative to the scan root; CODEOWNERS owns it.
pub const ALLOWLIST: &str = "xtask/env-ban-allow.txt";

/// 2026-10-17: The one sanctioned reader of the process environment.
pub const HOST_ENV: &str = "crates/io-hosts/std/src/env.rs";

/// 2026-10-17: The verdict on the tree rooted at `root`. A missing allowlist is an error.
fn check(root: &Path) -> Result<(Vec<judge::Source>, Vec<judge::Violation>), String> {
    let list_path = root.join(ALLOWLIST);
    let allowlist = std::fs::read_to_string(&list_path)
        .map_err(|e| format!("{ALLOWLIST}: cannot read {}: {e}", list_path.display()))?;
    let sources = walk::scan(root)?;
    let violations = judge::judge(&sources, &allowlist);
    Ok((sources, violations))
}

/// 2026-10-17: `cargo xtask env-ban` on the repository at `root`.
pub fn gate(root: &Path) -> Result<(), String> {
    let (sources, violations) = check(root)?;
    if violations.is_empty() {
        let readers = sources
            .iter()
            .filter(|s| s.kind == judge::Kind::Ordinary && !s.reads.is_empty())
            .count();
        println!(
            "xtask env-ban: {} sources scanned; {readers} allowlisted files still read the \
             environment",
            sources.len()
        );
        return Ok(());
    }
    for v in &violations {
        eprintln!("{v}");
    }
    Err(format!("xtask env-ban: {} violation(s)", violations.len()))
}

/// 2026-10-17: The env-ban probes under `fixtures`; returns how many did not end as intended.
pub fn selftest(fixtures: &Path) -> Result<usize, String> {
    probes::run(fixtures)
}
