// SPDX-License-Identifier: MIT OR Apache-2.0
#![forbid(unsafe_code)]
#![deny(warnings)]
#![deny(clippy::all)]

//! 2026-10-17: The repository gates.
//! - `cargo xtask deps`: every manifest follows the workspace dependency convention (the root
//!   declares each dependency once with `default-features = false`; members inherit it with
//!   `workspace = true` and name only their features). Exits non-zero on any violation.
//! - `cargo xtask selftest`: the deps gate refuses each known-bad fixture under
//!   `xtask/fixtures/` for its named reason, and passes the clean one.

mod manifest;
mod rules;
mod selftest;
mod shell;

use std::process::ExitCode;

const USAGE: &str = "usage: cargo xtask <deps|selftest>";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let outcome = match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["deps"] => deps(),
        ["selftest"] => selftest(),
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
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
    match selftest::run(&shell::fixtures_dir())? {
        0 => {
            println!("xtask selftest: every fixture was judged for its named reason");
            Ok(())
        }
        n => Err(format!(
            "xtask selftest: {n} probe(s) did not end as intended"
        )),
    }
}
