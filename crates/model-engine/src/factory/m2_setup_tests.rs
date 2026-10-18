// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `MoeTransposeLevers` against frozen copies of the environment readers it
//! replaced, through `MapEnv`-resolved settings (never the process environment).
//!
//! Owner: model-engine factory.
//! Invariants: every probe that reads differently from the old reader is asserted by name.

use metrale_model_layers::lever_equivalence::{Old, Outcome, check, settings_from};

use super::MoeTransposeLevers;

fn levers(s: &metrale_settings::InferenceServerSettings) -> MoeTransposeLevers {
    MoeTransposeLevers::from_settings(s)
}

/// 2026-10-17: Frozen `env::var(..).map(|v| v == "1" || v.eq_ignore_ascii_case("true"))`.
fn old_truthy(v: Option<&str>) -> bool {
    v.is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

#[test]
fn unified_layout_is_unchanged() {
    let changes = check(
        "METRALE_UNIFIED_MOE_LAYOUT",
        Old::Verbatim,
        old_truthy,
        |s| levers(s).unified_layout,
    );
    assert!(changes.is_empty(), "{changes:?}");
    for v in ["", "TRUE", " 1", "on", "yes", "0"] {
        let s = settings_from(&[("METRALE_UNIFIED_MOE_LAYOUT", v)]).unwrap();
        assert_eq!(levers(&s).unified_layout, old_truthy(Some(v)), "{v:?}");
    }
}

/// 2026-10-17: `METRALE_HYBRID_MOE_LAYOUT` became a Switch row: garbage, `""` included, is now
/// a startup error (was off), and `on` / `yes` now arm it (were off).
#[test]
fn hybrid_layout_changes_only_where_the_switch_rule_says() {
    let changes = check("METRALE_HYBRID_MOE_LAYOUT", Old::Truthy, old_truthy, |s| {
        levers(s).hybrid_layout
    });
    assert_eq!(
        changes,
        vec![(
            "METRALE_HYBRID_MOE_LAYOUT",
            Some("garbage"),
            Outcome::Rejected
        )]
    );
    for v in ["on", "yes", " TRUE "] {
        let s = settings_from(&[("METRALE_HYBRID_MOE_LAYOUT", v)]).unwrap();
        assert!(levers(&s).hybrid_layout, "{v:?}");
    }
    assert!(settings_from(&[("METRALE_HYBRID_MOE_LAYOUT", "")]).is_err());
}

/// 2026-10-17: Every integer keeps its byte count; a non-integer, `""` included (was the 2 GiB
/// default), is now a startup error, and surrounding spaces now parse.
#[test]
fn transpose_safety_matches_the_old_reader_where_it_parses() {
    let old = |v: Option<&str>| {
        v.and_then(|v| v.parse::<usize>().ok())
            .map(|mb| mb * 1024 * 1024)
            .unwrap_or(2 * 1024 * 1024 * 1024)
    };
    let changes = check("METRALE_MOE_TRANSPOSE_SAFETY_MB", Old::Number, old, |s| {
        levers(s).safety
    });
    let rejected = ["true", "off", "garbage"].map(|p| {
        (
            "METRALE_MOE_TRANSPOSE_SAFETY_MB",
            Some(p),
            Outcome::Rejected,
        )
    });
    assert_eq!(changes, rejected);
    for v in ["512", "-1", "0"] {
        let s = settings_from(&[("METRALE_MOE_TRANSPOSE_SAFETY_MB", v)]).unwrap();
        assert_eq!(levers(&s).safety, old(Some(v)), "{v:?}");
    }
    let s = settings_from(&[("METRALE_MOE_TRANSPOSE_SAFETY_MB", " 512 ")]).unwrap();
    assert_eq!(levers(&s).safety, 512 * 1024 * 1024);
    assert!(settings_from(&[("METRALE_MOE_TRANSPOSE_SAFETY_MB", "")]).is_err());
}
