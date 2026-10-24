// SPDX-License-Identifier: MIT OR Apache-2.0
#![forbid(unsafe_code)]
#![deny(warnings)]
#![deny(clippy::all)]

//! 2026-10-17: The repository gates.
//! - `cargo xtask deps`: every manifest follows the workspace dependency convention (the root
//!   declares each dependency once with `default-features = false`; members inherit it with
//!   `workspace = true` and name only their features). Exits non-zero on any violation.
//! - `cargo xtask env-ban [--base <rev>]`: only the `ProcessEnv` host, build scripts and the
//!   files listed in `xtask/env-ban-allow.txt` read the process environment; with `--base`, an
//!   entry added since the merge-base needs an `Env-Ban-Allow:` trailer (see `env_ban`).
//! - `cargo xtask selftest`: every gate refuses each known-bad fixture under `xtask/fixtures/`
//!   for its named reason, and passes the clean code; the env-ban gate yields exactly its
//!   expected violations on each `env-ban-*` fixture.
//! - `cargo xtask kzx-corpus <dir>`: the private kzx benchmark corpus (bodies and `.kbp`
//!   containers built from the PTX modules under `<dir>/ptx/`); see `kzx_corpus`.
//! - `cargo xtask <roles|depgraph|wasm|file-size|vendor-ban|kv-codec-ban|lints>`: one purity gate (see
//!   `gates.rs`); `cargo xtask purity` runs them all. Each reconciles its offenders with a
//!   checked-in allowlist under `xtask/` that may only shrink.

mod allow;
mod depgraph;
mod env_ban;
mod filesize;
mod gates;
mod kv_codec_ban;
mod kzx_corpus;
mod lints;
mod manifest;
mod metadata;
mod purity_io;
mod roles;
mod rules;
mod rust_text;
mod selftest;
mod selftest_purity;
mod shell;
mod vendor;

use std::process::ExitCode;

const USAGE: &str = "usage: cargo xtask <deps|env-ban [--base <rev>]|selftest|purity|roles|depgraph|wasm|file-size|vendor-ban|kv-codec-ban|lints|kzx-corpus <dir>>";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let outcome = match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["deps"] => deps(),
        ["env-ban"] => shell::repo_root().and_then(|root| env_ban::gate(&root, None)),
        ["env-ban", "--base", base] => {
            shell::repo_root().and_then(|root| env_ban::gate(&root, Some(base)))
        }
        ["selftest"] => selftest(),
        ["kzx-corpus", dir] => {
            shell::repo_root().and_then(|root| kzx_corpus::run(&root, std::path::Path::new(dir)))
        }
        ["purity"] => purity(&gates::ALL),
        [name] => match gates::Gate::parse(name) {
            Some(gate) => purity(&[gate]),
            None => return usage(),
        },
        _ => return usage(),
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn usage() -> ExitCode {
    eprintln!("{USAGE}");
    ExitCode::from(2)
}

fn deps() -> Result<(), String> {
    let ws = shell::load_workspace(&shell::repo_root()?)?;
    let violations = rules::check(&ws)?;
    if violations.is_empty() {
        println!(
            "xtask deps: the root and {} member manifests follow the dependency convention",
            ws.members.len()
        );
        return Ok(());
    }
    for v in &violations {
        eprintln!("{v}");
    }
    Err(format!("xtask deps: {} violation(s)", violations.len()))
}

fn selftest() -> Result<(), String> {
    let deps_failures = selftest::run(&shell::fixtures_dir())?;
    let purity_failures = selftest_purity::run(&shell::fixtures_dir(), true)?;
    match deps_failures.saturating_add(purity_failures) {
        0 => {
            println!("xtask selftest: every fixture was judged for its named reason");
            Ok(())
        }
        n => Err(format!(
            "xtask selftest: {n} probe(s) did not end as intended"
        )),
    }
}

/// 2026-10-17: Run `which` on the repository; every gate runs even after one fails, so one
/// run lists everything to fix.
fn purity(which: &[gates::Gate]) -> Result<(), String> {
    let root = shell::repo_root()?;
    let mut failed = Vec::new();
    for gate in which {
        let outcome = gates::run(*gate, &root, None)?;
        for p in &outcome.problems {
            eprintln!("{p}");
        }
        let verdict = if outcome.problems.is_empty() {
            "pass"
        } else {
            failed.push(gate.name());
            "FAIL"
        };
        println!(
            "xtask {}: {verdict}; {} lists {} entr{} (the shrinking work queue)",
            gate.name(),
            gate.allow_file(),
            outcome.listed,
            if outcome.listed == 1 { "y" } else { "ies" }
        );
    }
    if failed.is_empty() {
        Ok(())
    } else {
        Err(format!("xtask: failed gate(s): {}", failed.join(", ")))
    }
}
