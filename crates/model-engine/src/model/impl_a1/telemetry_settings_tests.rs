// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The `TransformerModel::new` telemetry reads (`impl_a1.rs`) on the resolved
//! settings against the readers they replaced, each copied from `impl_a1.rs` before the move.
//!
//! Owner: model-engine (model).
//! Invariants: the probes that read differently are exactly the listed behaviour changes.

use metrale_model_layers::lever_equivalence::{Old, Outcome, check};

#[test]
fn profile_first_reads_as_before() {
    let changes = check(
        "METRALE_PROFILE_FIRST",
        Old::Presence,
        |p| p.is_some(),
        |s| s.telemetry.profile_first,
    );
    assert_eq!(changes, []);
}

#[test]
fn diag_gemma4_reads_as_before_except_the_switch_spellings() {
    let changes = check(
        "METRALE_DIAG_GEMMA4",
        Old::TruthyExact,
        |p| p.is_some_and(|v| v == "1" || v == "true"),
        |s| s.telemetry.diag_gemma4,
    );
    assert_eq!(
        changes,
        [("METRALE_DIAG_GEMMA4", Some("garbage"), Outcome::Rejected)]
    );
}

#[test]
fn debug_no_graph_reads_as_before_except_the_switch_spellings() {
    let changes = check(
        "METRALE_DEBUG_NO_GRAPH",
        Old::OptIn,
        |p| p == Some("1"),
        |s| s.telemetry.debug_no_graph,
    );
    assert_eq!(
        changes,
        [
            ("METRALE_DEBUG_NO_GRAPH", Some("true"), Outcome::Changed),
            ("METRALE_DEBUG_NO_GRAPH", Some("garbage"), Outcome::Rejected),
        ]
    );
}
