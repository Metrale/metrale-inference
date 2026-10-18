// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: `QsaLevers::from_settings` against the three `== "1"` readers of
//! `qsa_select.rs` it replaced, frozen here as the predicate on the raw value.
//!
//! Owner: model-layers (QSA).
//! Invariants: the probes that read differently are exactly the listed behaviour change: each
//! switch now also reads `true` (and the other switch spellings) and refuses a value the switch
//! rule cannot read.

use super::*;
use crate::lever_equivalence::{Old, Outcome, check, defaults};

type Field = (&'static str, fn(&QsaLevers) -> bool);

/// 2026-10-18: Each lever probed as its variable reads: the kill switch as itself
/// (`!prefill_select`), the two opt-ins as they are.
#[test]
fn the_switches_read_as_before_except_the_switch_spellings() {
    let cases: [Field; 3] = [
        ("METRALE_QSA_NO_PREFILL_SELECT", |l| !l.prefill_select),
        ("METRALE_QSA_S2_DIAG", |l| l.s2_diag),
        ("METRALE_QSA_SCORE_SCALAR", |l| l.score_scalar),
    ];
    for (env, field) in cases {
        let changes = check(
            env,
            Old::OptIn,
            |p| p == Some("1"),
            |s| field(&QsaLevers::from_settings(s)),
        );
        assert_eq!(
            changes,
            [
                (env, Some("true"), Outcome::Changed),
                (env, Some("garbage"), Outcome::Rejected)
            ],
            "{env}"
        );
    }
}

/// 2026-10-18: Every lever at the value its reader read unset as.
#[test]
fn the_declared_defaults_are_the_old_unset_values() {
    assert_eq!(QsaLevers::from_settings(&defaults()), QsaLevers::declared());
}
