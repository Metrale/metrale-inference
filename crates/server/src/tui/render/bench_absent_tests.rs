// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The Benchmarks section in a build without the `bench` feature.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::tests::{app, render};
use crate::tui::app::Section;

#[test]
fn the_benchmarks_tab_names_the_missing_feature_and_the_rebuild() {
    let mut a = app();
    a.section = Section::Benchmarks;
    let out = render(&a, 200, 50);
    assert!(out.contains("without the `bench` feature"), "{out}");
    assert!(out.contains(crate::cli::no_bench::REBUILD), "{out}");
    assert!(
        out.contains("⇥ cycle"),
        "the footer names the way out:\n{out}"
    );
}
