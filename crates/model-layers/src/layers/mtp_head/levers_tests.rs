// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: `MtpLevers::from_settings` against the environment readers it replaced, frozen
//! here as the predicates those readers applied to the raw value.
//!
//! Owner: model-layers (MTP head).
//! Invariants: the probes that read differently are exactly the listed behaviour changes: the
//! profile switch also reads the other switch spellings and refuses a value the switch rule
//! cannot read; the presence and text levers read as before.

use super::*;
use crate::lever_equivalence::{Old, Outcome, check, defaults, settings_from};

/// 2026-10-18: The levers of `env` at `probe` and nothing else; `None` when the settings refuse
/// it.
fn new_at(env: &str, probe: &str) -> Option<MtpLevers> {
    settings_from(&[(env, probe)])
        .ok()
        .map(|s| MtpLevers::from_settings(&s))
}

/// 2026-10-18: `METRALE_MTP_PREFILL_PROFILE` was on at exactly `1`.
#[test]
fn the_prefill_profile_reads_as_before_except_the_switch_spellings() {
    const ENV: &str = "METRALE_MTP_PREFILL_PROFILE";
    let changes = check(
        ENV,
        Old::OptIn,
        |p| p == Some("1"),
        |s| MtpLevers::from_settings(s).prefill_profile,
    );
    assert_eq!(
        changes,
        [
            (ENV, Some("true"), Outcome::Changed),
            (ENV, Some("garbage"), Outcome::Rejected)
        ]
    );
}

/// 2026-10-18: The `var_os(..).is_some()` opt-in and the `is_none()` kill switch: set at any
/// value, `0` and the empty string included.
#[test]
fn the_presence_levers_read_as_before() {
    let cases: [(&str, fn(&MtpLevers) -> bool, bool); 2] = [
        ("METRALE_MTP_KV_GEMV", |l| l.kv_gemv_pinned, true),
        ("METRALE_NO_MTP_LMHEAD_TGEMM", |l| l.lmhead_tgemm, false),
    ];
    for (env, field, set) in cases {
        let old = move |p: Option<&str>| p.is_some() == set;
        let changes = check(env, Old::Presence, old, |s| {
            field(&MtpLevers::from_settings(s))
        });
        assert!(changes.is_empty(), "{changes:?}");
        for probe in ["", " ", "no"] {
            assert_eq!(
                new_at(env, probe).map(|l| field(&l)),
                Some(set),
                "{env}={probe:?}"
            );
        }
    }
}

/// 2026-10-18: `METRALE_NO_DRAFTER_SMALL_M_TIER` turned the tier off at exactly `1`.
#[test]
fn the_small_m_tier_predicate_is_kept() {
    const ENV: &str = "METRALE_NO_DRAFTER_SMALL_M_TIER";
    let old = |p: Option<&str>| p == Some("1");
    let changes = check(ENV, Old::Verbatim, old, |s| {
        MtpLevers::from_settings(s).small_m_tier_off
    });
    assert!(changes.is_empty(), "{changes:?}");
    for probe in ["", " 1", "01", "TRUE", "on"] {
        assert_eq!(
            new_at(ENV, probe).map(|l| l.small_m_tier_off),
            Some(old(Some(probe))),
            "{probe:?}"
        );
    }
}

/// 2026-10-18: `METRALE_PROPOSE_META_STRIDE` reaches the stride parse verbatim: padding,
/// alignment, the clamps and every unparseable value read as before.
#[test]
fn the_propose_meta_stride_parse_is_kept() {
    const ENV: &str = "METRALE_PROPOSE_META_STRIDE";
    let old = super::super::batch_caps::parse_stride_override;
    let changes = check(ENV, Old::Verbatim, old, |s| {
        MtpLevers::from_settings(s).propose_meta_stride
    });
    assert!(changes.is_empty(), "{changes:?}");
    let huge = usize::MAX.to_string();
    for probe in [
        "",
        " 4361 ",
        "4361",
        "0",
        "-1",
        "not-bytes",
        "8",
        huge.as_str(),
    ] {
        assert_eq!(
            new_at(ENV, probe).map(|l| l.propose_meta_stride),
            Some(old(Some(probe))),
            "{probe:?}"
        );
    }
    assert_eq!(
        new_at(ENV, " 4361 ").and_then(|l| l.propose_meta_stride),
        Some(4368)
    );
}

/// 2026-10-18: Every lever at the value its reader read unset as.
#[test]
fn the_declared_defaults_are_the_old_unset_values() {
    assert_eq!(MtpLevers::from_settings(&defaults()), MtpLevers::declared());
}
