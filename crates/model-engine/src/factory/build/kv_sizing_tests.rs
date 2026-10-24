// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `KvPoolLevers` against frozen copies of the environment readers it replaced,
//! through `MapEnv`-resolved settings (never the process environment).
//!
//! Owner: model-engine factory.
//! Invariants: every probe that reads differently from the old reader is asserted by name.

use metrale_model_layers::lever_equivalence::{Old, Outcome, check, settings_from};

use super::KvPoolLevers;

fn levers(s: &metrale_settings::InferenceServerSettings) -> KvPoolLevers {
    KvPoolLevers::from_settings(s)
}

#[test]
fn unclamped_is_presence() {
    let changes = check(
        "METRALE_KV_POOL_UNCLAMPED",
        Old::Presence,
        |v: Option<&str>| v.is_some(),
        |s| levers(s).unclamped,
    );
    assert!(changes.is_empty(), "{changes:?}");
    let s = settings_from(&[("METRALE_KV_POOL_UNCLAMPED", "")]).unwrap();
    assert!(levers(&s).unclamped);
}

#[test]
fn overcommit_matches_the_old_reader() {
    // 2026-10-17: Frozen `!matches!(env::var(..).as_deref(), Ok("0") | Ok("false"))`.
    let old = |v: Option<&str>| !matches!(v, Some("0" | "false"));
    let changes = check("METRALE_KV_OVERCOMMIT", Old::Verbatim, old, |s| {
        levers(s).overcommit
    });
    assert!(changes.is_empty(), "{changes:?}");
    for v in ["", "false", "FALSE", " 0", "off", "1"] {
        let s = settings_from(&[("METRALE_KV_OVERCOMMIT", v)]).unwrap();
        assert_eq!(levers(&s).overcommit, old(Some(v)), "{v:?}");
    }
}

/// 2026-10-17: The reserve is applied only when positive, so the old and new values are
/// compared after that filter. Garbage, `""`, `inf` and `NaN` (were "no reserve") are now
/// startup errors, and surrounding spaces now parse.
#[test]
fn external_reserve_matches_the_old_reader_where_it_parses() {
    let applied = |v: Option<f64>| v.filter(|&gb| gb > 0.0);
    let old = |v: Option<&str>| applied(v.and_then(|v| v.parse::<f64>().ok()));
    let changes = check("METRALE_KV_EXTERNAL_RESERVE_GB", Old::Number, old, |s| {
        applied(levers(s).external_reserve_gb)
    });
    let rejected = ["true", "off", "garbage"]
        .map(|p| ("METRALE_KV_EXTERNAL_RESERVE_GB", Some(p), Outcome::Rejected));
    assert_eq!(changes, rejected);
    for v in ["2.5", "-1", "0", "8"] {
        let s = settings_from(&[("METRALE_KV_EXTERNAL_RESERVE_GB", v)]).unwrap();
        assert_eq!(
            applied(levers(&s).external_reserve_gb),
            old(Some(v)),
            "{v:?}"
        );
    }
    let s = settings_from(&[("METRALE_KV_EXTERNAL_RESERVE_GB", " 2.5 ")]).unwrap();
    assert_eq!(levers(&s).external_reserve_gb, Some(2.5));
    for v in ["", "inf", "NaN"] {
        assert!(
            settings_from(&[("METRALE_KV_EXTERNAL_RESERVE_GB", v)]).is_err(),
            "{v:?}"
        );
    }
}
