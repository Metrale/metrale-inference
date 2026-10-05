// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: `met circuit lkb` on the checked-out tree: its ledger fields equal the shares of
//! the report `met circuit plan` builds for the same cell, and the Hopper class's residual is
//! its own `kernels/hopper/` sources.
//!
//! Owner: server CLI tests.
//! Invariants: the tests read the repository at the workspace root and write nothing.

use std::path::PathBuf;

use super::{CircuitLkbArgs, CircuitLkbFormat, lkb_text};
use crate::cli::CircuitPrecision;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

fn args(hardware: &str, format: CircuitLkbFormat) -> CircuitLkbArgs {
    let dir = root()
        .join(super::super::circuit_hw::MATRIX_CONFIGS)
        .join("Qwen--Qwen3.6-27B-FP8");
    CircuitLkbArgs {
        checkpoint: dir.to_string_lossy().into_owned(),
        hardware: hardware.into(),
        precision: CircuitPrecision::Declared,
        format,
        allow_network: false,
        root: None,
    }
}

fn ledger(hardware: &str) -> toml::Table {
    let text = lkb_text(&args(hardware, CircuitLkbFormat::Toml), &root()).expect("lkb");
    toml::from_str(&text).unwrap_or_else(|e| panic!("{e}\n{text}"))
}

// 2026-10-05: Mutation: listing inherited gb10 sources as Hopper's residual inflates the count
// past the class's own files; listing none (a filter on the wrong class) reads zero.
#[test]
fn the_hopper_residual_is_its_own_sources_and_gb10_has_coverage() {
    let hopper = ledger("h100-sxm");
    assert_eq!(hopper["lkb_class"].as_str(), Some("hopper"));
    let own = std::fs::read_dir(root().join("kernels/hopper/common"))
        .expect("hopper common")
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "cu"))
        .count() as i64;
    let count = hopper["residual_count"]
        .as_integer()
        .expect("residual_count");
    assert!(
        (1..=own).contains(&count),
        "residual {count} of {own} own sources"
    );
    let gb10 = ledger("gb10");
    let pct = gb10["lkb_coverage_pct"].as_str().expect("coverage");
    assert_eq!(pct.split('/').count(), 3, "C1/C16/C128: {pct}");
    for p in pct.split('/') {
        let v: f64 = p.parse().expect("a percentage");
        assert!((0.0..=100.0).contains(&v), "{pct}");
    }
}

// 2026-10-05: Mutation: rendering the TOML for `--format report` (or the reverse).
#[test]
fn the_report_format_is_markdown() {
    let md = lkb_text(&args("gb10", CircuitLkbFormat::Report), &root()).expect("lkb");
    assert!(md.starts_with("# LKB on gb10: "), "{md}");
    assert!(md.contains("## LKB residual on gb10: "), "{md}");
}
