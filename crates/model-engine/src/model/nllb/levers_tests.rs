// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `NllbLevers` against frozen copies of the environment readers it replaced,
//! through `MapEnv`-resolved settings (never the process environment).
//!
//! Owner: model-engine NLLB.
//! Invariants: every probe that reads differently from the old reader is asserted by name.

use metrale_model_layers::lever_equivalence::{Old, Outcome, check, settings_from};

use super::NllbLevers;

fn levers(s: &metrale_settings::InferenceServerSettings) -> NllbLevers {
    NllbLevers::from_settings(s)
}

/// 2026-10-17: Was `== "1"`; `true` now arms it and garbage is a startup error.
#[test]
fn beam_profile_changes_only_where_the_switch_rule_says() {
    let changes = check(
        "METRALE_NLLB_BEAM_PROFILE",
        Old::OptIn,
        |v: Option<&str>| v == Some("1"),
        |s| levers(s).beam_profile,
    );
    assert_eq!(
        changes,
        vec![
            ("METRALE_NLLB_BEAM_PROFILE", Some("true"), Outcome::Changed),
            (
                "METRALE_NLLB_BEAM_PROFILE",
                Some("garbage"),
                Outcome::Rejected
            ),
        ]
    );
}

/// 2026-10-17: Was `1` or `true` in any case; garbage is now a startup error and `on` / `yes`
/// now arm it.
#[test]
fn host_topk_changes_only_where_the_switch_rule_says() {
    let changes = check(
        "METRALE_NLLB_HOST_TOPK",
        Old::Truthy,
        |v: Option<&str>| v.is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true")),
        |s| levers(s).host_topk,
    );
    assert_eq!(
        changes,
        vec![("METRALE_NLLB_HOST_TOPK", Some("garbage"), Outcome::Rejected)]
    );
    for v in ["on", "yes", "TRUE"] {
        let s = settings_from(&[("METRALE_NLLB_HOST_TOPK", v)]).unwrap();
        assert!(levers(&s).host_topk, "{v:?}");
    }
}

/// 2026-10-17: Every integer keeps its row count (a negative one the default, as before); a
/// non-integer, `""` included (was 64), is now a startup error, and surrounding spaces parse.
#[test]
fn device_topk_min_rows_matches_the_old_reader_where_it_parses() {
    let old = |v: Option<&str>| v.and_then(|v| v.parse::<usize>().ok()).unwrap_or(64);
    let changes = check("METRALE_NLLB_DEVICE_TOPK_MIN_ROWS", Old::Number, old, |s| {
        levers(s).device_topk_min_rows
    });
    let rejected = ["true", "off", "garbage"].map(|p| {
        (
            "METRALE_NLLB_DEVICE_TOPK_MIN_ROWS",
            Some(p),
            Outcome::Rejected,
        )
    });
    assert_eq!(changes, rejected);
    for v in ["0", "16", "-4"] {
        let s = settings_from(&[("METRALE_NLLB_DEVICE_TOPK_MIN_ROWS", v)]).unwrap();
        assert_eq!(levers(&s).device_topk_min_rows, old(Some(v)), "{v:?}");
    }
    let s = settings_from(&[("METRALE_NLLB_DEVICE_TOPK_MIN_ROWS", " 16 ")]).unwrap();
    assert_eq!(levers(&s).device_topk_min_rows, 16);
    assert!(settings_from(&[("METRALE_NLLB_DEVICE_TOPK_MIN_ROWS", "")]).is_err());
}
