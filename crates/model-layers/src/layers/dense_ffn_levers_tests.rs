// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: `DenseFfnLevers::from_settings` against the environment readers it replaced,
//! frozen here as the predicates the closures apply to the raw value.
//!
//! Owner: model-layers (dense FFN).
//! Invariants: the probes that read differently are exactly the listed behaviour changes: the
//! batch16 opt-in now also reads the other switch spellings and refuses a value the switch rule
//! cannot read; the presence and tile levers read as before.

use super::*;
use crate::lever_equivalence::{Old, Outcome, check, defaults, settings_from};

/// 2026-10-18: The levers of `env` at `probe` and nothing else; `None` when the settings refuse
/// it.
fn new_at(env: &str, probe: &str) -> Option<DenseFfnLevers> {
    settings_from(&[(env, probe)])
        .ok()
        .map(|s| DenseFfnLevers::from_settings(&s))
}

/// 2026-10-18: `ffn_batch16_enabled` was on at exactly `1`.
#[test]
fn batch16_reads_as_before_except_the_switch_spellings() {
    const ENV: &str = "METRALE_FFN_BATCH16";
    let changes = check(
        ENV,
        Old::OptIn,
        |p| p == Some("1"),
        |s| DenseFfnLevers::from_settings(s).batch16,
    );
    assert_eq!(
        changes,
        [
            (ENV, Some("true"), Outcome::Changed),
            (ENV, Some("garbage"), Outcome::Rejected)
        ]
    );
}

/// 2026-10-18: `ffn_w8a16_only` was `var_os(..).is_some()`: set at any value, `0` and the
/// empty string included.
#[test]
fn w8a16_only_reads_by_presence_as_before() {
    const ENV: &str = "METRALE_FFN_W8A16_ONLY";
    let changes = check(
        ENV,
        Old::Presence,
        |p| p.is_some(),
        |s| DenseFfnLevers::from_settings(s).w8a16_only,
    );
    assert!(changes.is_empty(), "{changes:?}");
    for probe in ["", " ", "0", "no"] {
        assert_eq!(
            new_at(ENV, probe).map(|l| l.w8a16_only),
            Some(true),
            "{ENV}={probe:?}"
        );
    }
}

/// 2026-10-18: `resolve_m16_tc_levers` took the wide tile at exactly `64`; any other value,
/// an explicit `32`, an empty value and a typo included, kept 32.
#[test]
fn the_n_tile_is_wide_at_exactly_64() {
    const ENV: &str = "METRALE_FFN_M16_TC_NTILE";
    let old = |p: Option<&str>| p == Some("64");
    let changes = check(ENV, Old::Verbatim, old, |s| {
        DenseFfnLevers::from_settings(s).m16_tc_n_tile == W8A16_GEMM_M16_N_TILE_WIDE
    });
    assert!(changes.is_empty(), "{changes:?}");
    assert_eq!(
        new_at(ENV, "64").map(|l| l.m16_tc_n_tile),
        Some(W8A16_GEMM_M16_N_TILE_WIDE)
    );
    for probe in ["32", "", "128", "yes", "6 4", " 64", "064"] {
        assert_eq!(
            new_at(ENV, probe).map(|l| l.m16_tc_n_tile),
            Some(W8A16_GEMM_M16_N_TILE),
            "{ENV}={probe:?} must keep 32"
        );
    }
}

/// 2026-10-18: Every lever at the value its reader read unset as.
#[test]
fn the_declared_defaults_are_the_old_unset_values() {
    assert_eq!(
        DenseFfnLevers::from_settings(&defaults()),
        DenseFfnLevers::declared()
    );
}
