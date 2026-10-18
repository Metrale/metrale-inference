// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Tests for the refusal of bench-only subcommands in a build without `bench`.
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

use clap::{CommandFactory, Parser};

use super::*;
use crate::cli::Cli;

fn parse(argv: &[&str]) -> Command {
    Cli::try_parse_from(argv)
        .unwrap_or_else(|e| panic!("{argv:?} must parse: {e}"))
        .command
}

#[test]
fn every_benchmark_invocation_reaches_the_refusal() {
    for argv in [
        &["met", "benchmark"][..],
        &["met", "bench", "certify", "--json", "--gate", "x"],
        &["met", "benchmark", "--help"],
        &[
            "met",
            "benchmark",
            "run",
            "quick-speed",
            "--",
            "--port",
            "1",
        ],
    ] {
        assert_eq!(bench_only(&parse(argv)), Some("benchmark"), "{argv:?}");
    }
    assert_eq!(bench_only(&parse(&["met", "doctor"])), Some("doctor"));
}

#[test]
fn serving_commands_are_not_refused() {
    for argv in [
        &["met", "serve", "some/model"][..],
        &["met", "dump-serve-options"],
        &["met", "sync-recipes"],
    ] {
        assert_eq!(bench_only(&parse(argv)), None, "{argv:?}");
    }
}

#[test]
fn the_refusal_names_the_feature_and_the_rebuild() {
    let m = message("benchmark");
    assert!(m.contains("`met benchmark`"), "{m}");
    assert!(m.contains("`bench` feature"), "{m}");
    assert!(m.contains("--features bench"), "{m}");
}

#[test]
fn help_lists_neither_bench_only_command() {
    let help = Cli::command().render_help().to_string();
    assert!(help.contains("serve"), "{help}");
    for absent in ["benchmark", "doctor"] {
        assert!(
            !help.lines().any(|l| l.trim_start().starts_with(absent)),
            "`{absent}` must be hidden: {help}"
        );
    }
}
