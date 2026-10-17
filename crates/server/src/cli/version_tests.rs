// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `met --version` reports the packaged version.
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn version_flag_reports_the_packaged_version() {
    // 2026-09-26: `--version` ends parsing early, so clap returns it as an error whose
    // kind is DisplayVersion and whose rendering is the output.
    let err = Cli::try_parse_from(["met", "--version"]).expect_err("exits early");
    assert_eq!(err.kind(), clap::error::ErrorKind::DisplayVersion);
    assert!(
        err.to_string().contains(METRALE_VERSION),
        "`--version` printed {:?}, which does not carry {METRALE_VERSION}",
        err.to_string()
    );
}

#[test]
fn the_reported_version_is_the_cargo_version() {
    // 2026-09-26: The constant is the package version itself, not a literal copy.
    assert_eq!(METRALE_VERSION, env!("CARGO_PKG_VERSION"));
    assert!(!METRALE_VERSION.is_empty());
}
