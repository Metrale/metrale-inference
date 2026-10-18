// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The Laguna RoPE-table lever read from the settings against the
//! `sliding_rope_table_enabled` reader it replaced, frozen here. The unified-layout and
//! grouped-CUTLASS levers are the shared ones (`shared_levers_tests.rs`).
//!
//! Owner: model-arch weight loader (Laguna).
//! Invariants: the probes that read differently are exactly the listed behaviour changes.

use metrale_model_layers::lever_equivalence::{Old, Outcome, check, settings_from};

const ENV: &str = "METRALE_LAGUNA_ROPE_TABLE";

#[test]
fn rope_table_moves_from_not_zero_to_a_switch() {
    let changes = check(
        ENV,
        Old::OptOut,
        |p| p != Some("0"),
        |s| s.kernels.laguna_rope_table,
    );
    assert_eq!(
        changes,
        [
            (ENV, Some("off"), Outcome::Changed),
            (ENV, Some("garbage"), Outcome::Rejected),
        ]
    );
    assert!(
        settings_from(&[(ENV, "")]).is_err(),
        "\"\" was on, now refused"
    );
}
