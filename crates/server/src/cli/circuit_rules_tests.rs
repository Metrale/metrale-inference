// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The committed critical-pair report is what the rules produce today, and the
//! allowlist gate passes over it. Regenerate after an intended rule change with
//! `met circuit rules --critical-pairs` and review the diff (and the allowlist).
//!
//! Owner: server CLI tests.
//! Invariants: reads the repository at the workspace root; writes nothing.

use std::path::PathBuf;

#[test]
fn critical_pairs_report_is_current_and_the_gate_passes() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root");
    super::run(super::CircuitRulesArgs {
        critical_pairs: true,
        check: true,
        root: Some(root),
    })
    .unwrap_or_else(|e| panic!("{e:#}\nregenerate with `met circuit rules --critical-pairs`"));
}
