// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `kernels.no_attn_q_inplace` against the reader it replaced (`q_inplace_enabled`:
//! the in-place Q read is off only at `METRALE_NO_ATTN_Q_INPLACE=1`).
//!
//! Owner: model-layers (qwen3_attention).
//! Invariants: the probes that read differently are exactly the listed behaviour changes.

use crate::lever_equivalence::{Old, Outcome, check};

#[test]
fn no_attn_q_inplace_reads_as_before_except_the_switch_spellings() {
    let changes = check(
        "METRALE_NO_ATTN_Q_INPLACE",
        Old::OptIn,
        |p| p == Some("1"),
        |s| s.kernels.no_attn_q_inplace,
    );
    assert_eq!(
        changes,
        [
            ("METRALE_NO_ATTN_Q_INPLACE", Some("true"), Outcome::Changed),
            (
                "METRALE_NO_ATTN_Q_INPLACE",
                Some("garbage"),
                Outcome::Rejected
            ),
        ]
    );
}
