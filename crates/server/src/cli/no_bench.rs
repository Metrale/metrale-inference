// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `met benchmark` and `met doctor` in a binary built without the `bench` cargo
//! feature.
//!
//! The commands still parse, hidden from `--help`, and are refused with a message that names the
//! feature and the rebuild command. Leaving them out of clap entirely would answer
//! `unrecognized subcommand 'benchmark'`, which reads as a typo or a removed command rather than
//! as a build choice; a declared, reported mode is what PCND asks for when a capability is
//! absent. The refusal happens before any logging subscriber, dashboard or GPU exists.
//!
//! Owner: server CLI.
//! Invariants:
//! - A refused command exits with status 2 (a usage error, as clap's own) and never runs.
//! - Every argument after the subcommand, `--help` included, is accepted and ignored, so the
//!   refusal is the only answer whatever follows.

use super::Command;

/// 2026-10-17: The arguments of `met benchmark` in a build without the harness: everything
/// after the subcommand, kept only so any invocation reaches the refusal.
#[derive(clap::Args, Debug)]
#[command(disable_help_flag = true)]
pub struct BenchmarkArgs {
    #[arg(trailing_var_arg = true, allow_hyphen_values = true, num_args = 0..)]
    ignored: Vec<String>,
}

/// 2026-10-17: The subcommand name when `command` needs the `bench` feature, else `None`.
pub(crate) fn bench_only(command: &Command) -> Option<&'static str> {
    match command {
        Command::Benchmark(_) => Some("benchmark"),
        Command::Doctor => Some("doctor"),
        Command::Serve(_)
        | Command::DumpServeOptions
        | Command::SyncRecipes
        | Command::Circuit(_)
        | Command::Blueprint(_) => None,
    }
}

/// 2026-10-17: The build command that restores the bench-only commands and the Benchmarks tab.
pub(crate) const REBUILD: &str = "cargo build -p met --release --features bench";

/// 2026-10-17: The refusal for a bench-only subcommand.
pub(crate) fn message(subcommand: &str) -> String {
    format!(
        "`met {subcommand}` is not in this build: met was built without the `bench` feature. \
         Rebuild with `{REBUILD}`."
    )
}

/// 2026-10-17: Refuse `command` and exit with status 2 when it needs the `bench` feature;
/// return otherwise.
pub(crate) fn refuse_if_bench_only(command: &Command) {
    if let Some(subcommand) = bench_only(command) {
        eprintln!("error: {}", message(subcommand));
        std::process::exit(2);
    }
}

#[cfg(test)]
#[path = "no_bench_tests.rs"]
mod tests;
