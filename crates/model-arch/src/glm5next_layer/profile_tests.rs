// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `Profile::from_settings` against the environment readers it replaced, frozen
//! here (`old_*`, unchanged but for taking the raw value as a parameter).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: the level keeps its text predicate (`1` or `2` exactly); the route trace, now a
//! switch, also reads the other switch spellings and refuses a value the switch rule cannot
//! read.

use super::*;
use metrale_model_layers::lever_equivalence::{Old, Outcome, check, defaults, settings_from};

/// 2026-10-17: The replaced `level`.
fn old_level(v: Option<&str>) -> u8 {
    match v {
        Some("1") => 1,
        Some("2") => 2,
        _ => 0,
    }
}

/// 2026-10-17: What each level opens: (`on`, `full`, `start`, `start_hot`) as booleans.
fn spans(p: Profile) -> (bool, bool, bool, bool) {
    (
        p.on(),
        p.full(),
        p.start().is_some(),
        p.start_hot().is_some(),
    )
}

#[test]
fn the_level_keeps_its_text_predicate() {
    let env = "METRALE_GLM_PROFILE";
    let changes = check(env, Old::Verbatim, old_level, |s| {
        Profile::from_settings(s).level
    });
    assert!(changes.is_empty(), "{changes:?}");
    for (probe, want) in [
        ("1", (true, true, true, true)),
        ("2", (true, false, false, true)),
        ("0", (false, false, false, false)),
        (" 1", (false, false, false, false)),
        ("3", (false, false, false, false)),
        ("", (false, false, false, false)),
    ] {
        let s = settings_from(&[(env, probe)]).expect("a text lever resolves");
        let p = Profile::from_settings(&s);
        assert_eq!(p.level, old_level(Some(probe)), "{probe:?}");
        assert_eq!(spans(p), want, "{probe:?}");
    }
}

#[test]
fn the_route_trace_reads_as_before_except_the_switch_spellings() {
    let env = "METRALE_GLM_ROUTE_TRACE";
    let changes = check(
        env,
        Old::OptIn,
        |p| p == Some("1"),
        |s| Profile::from_settings(s).trace_on(),
    );
    assert_eq!(
        changes,
        [
            (env, Some("true"), Outcome::Changed),
            (env, Some("garbage"), Outcome::Rejected)
        ]
    );
}

#[test]
fn the_declared_defaults_are_off() {
    let p = Profile::from_settings(&defaults());
    assert_eq!(
        p,
        Profile {
            level: 0,
            trace: false
        }
    );
    assert_eq!(spans(p), (false, false, false, false));
}
