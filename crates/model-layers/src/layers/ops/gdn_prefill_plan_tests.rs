// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `GdnPrefillPlan::from_settings` against the environment readers it replaced,
//! frozen here (`old_*`, unchanged but for taking the raw values as parameters).
//!
//! Owner: model-layers ops (GDN).
//! Invariants: the probes that read differently are exactly the listed behaviour changes: a
//! `0`/`1` switch now also reads the other switch spellings and refuses a value the switch rule
//! cannot read.

use super::*;
use crate::lever_equivalence::{Old, Outcome, PROBES, check, defaults, settings_from};

/// 2026-10-17: `ssm_gdn_tc_route::gdn_scalar_spine_from`.
fn old_spine(pipe: Option<&str>, vtile: Option<&str>) -> GdnScalarSpine {
    match (pipe, vtile) {
        (Some("1"), _) => GdnScalarSpine::Pipe,
        (_, Some("1")) => GdnScalarSpine::Vtile,
        (Some("0"), _) => GdnScalarSpine::Vfused,
        _ => GdnScalarSpine::Pipe,
    }
}

/// 2026-10-17: The plan of `pairs` and nothing else; `None` when the settings refuse them.
fn plan_of(pairs: &[(&str, &str)]) -> Option<GdnPrefillPlan> {
    settings_from(pairs)
        .ok()
        .map(|s| GdnPrefillPlan::from_settings(&s))
}

/// 2026-10-17: The `0`/`1` value the old reader needed for what a switch spelling means.
fn canonical(probe: Option<&str>) -> Option<&str> {
    match probe {
        Some("true") => Some("1"),
        Some("off") => Some("0"),
        other => other,
    }
}

/// 2026-10-17: Every pair of the two switches' probes: refused where either is unreadable,
/// else the old choice of the values the old reader needed for the same meaning, which is the
/// old choice itself wherever both probes are `0`, `1` or unset. At the defaults, with
/// `VTILE=0` alone and with `PIPE=1` alone, the pipe spine, as before.
#[test]
fn the_spine_is_the_old_choice_of_the_same_meaning() {
    const PIPE: &str = "METRALE_GDN_PIPE";
    const VTILE: &str = "METRALE_GDN_VTILE";
    let mut changed = Vec::new();
    for pipe in PROBES {
        for vtile in PROBES {
            let mut pairs = Vec::new();
            pairs.extend(pipe.map(|v| (PIPE, v)));
            pairs.extend(vtile.map(|v| (VTILE, v)));
            let new = plan_of(&pairs).map(|p| p.spine);
            if pipe == Some("garbage") || vtile == Some("garbage") {
                assert_eq!(new, None, "{pipe:?} {vtile:?}");
                continue;
            }
            assert_eq!(
                new,
                Some(old_spine(canonical(pipe), canonical(vtile))),
                "{pipe:?} {vtile:?}"
            );
            if new != Some(old_spine(pipe, vtile)) {
                changed.push((pipe, vtile));
            }
        }
    }
    for (pipe, vtile) in &changed {
        assert!(
            [pipe, vtile]
                .iter()
                .any(|p| matches!(p, Some("true" | "off"))),
            "{pipe:?} {vtile:?} changed without a new spelling"
        );
    }
    assert_eq!(
        changed,
        [
            (Some("true"), Some("1")),
            (Some("0"), Some("true")),
            (Some("off"), Some("true")),
            (Some("off"), Some("0")),
            (Some("off"), Some("off")),
            (Some("off"), None),
            (None, Some("true")),
        ]
    );
}

type Switch = (
    &'static str,
    Old,
    fn(Option<&str>) -> bool,
    fn(&GdnPrefillPlan) -> bool,
);

/// 2026-10-17: The fused spine's `!= "0"` (`ssm_gdn_a3.rs`), the DV-block-split and TMA
/// spines' `== "1"` (`ssm_gdn_a3.rs`) and the 8-warp twin's kill switch, present at any value
/// (`ssm_gdn_hopper_prefill::gdn_fwd_o_mma8`).
#[test]
fn the_spine_switches_read_as_before_except_the_switch_spellings() {
    let cases: [(Switch, &[(Option<&str>, Outcome)]); 4] = [
        (
            (
                "METRALE_GDN_VTILE",
                Old::OptOut,
                |p| p != Some("0"),
                |l| l.fused,
            ),
            &[
                (Some("off"), Outcome::Changed),
                (Some("garbage"), Outcome::Rejected),
            ],
        ),
        (
            (
                "METRALE_GDN_TC_VBLOCK",
                Old::OptIn,
                |p| p == Some("1"),
                |l| l.tc_vblock,
            ),
            &[
                (Some("true"), Outcome::Changed),
                (Some("garbage"), Outcome::Rejected),
            ],
        ),
        (
            ("METRALE_GDN_TMA", Old::OptIn, |p| p == Some("1"), |l| l.tma),
            &[
                (Some("true"), Outcome::Changed),
                (Some("garbage"), Outcome::Rejected),
            ],
        ),
        (
            (
                "METRALE_NO_GDN_FWD_O_MMA8",
                Old::Presence,
                |p| p.is_none(),
                |l| l.fwd_o_mma8,
            ),
            &[],
        ),
    ];
    for ((env, kind, old, field), want) in cases {
        let changes = check(env, kind, old, |s| field(&GdnPrefillPlan::from_settings(s)));
        let want: Vec<_> = want.iter().map(|&(p, o)| (env, p, o)).collect();
        assert_eq!(changes, want, "{env}");
    }
    assert_eq!(
        plan_of(&[("METRALE_NO_GDN_FWD_O_MMA8", "")]).map(|p| p.fwd_o_mma8),
        Some(false),
        "an empty value is present"
    );
}

/// 2026-10-17: `ssm_gdn_hopper_prefill::gdn_hopper_remnants_killed`: exactly `1`.
#[test]
fn the_remnants_kill_switch_keeps_its_text_predicate() {
    let env = "METRALE_NO_GDN_PREFILL_TC_REMNANTS";
    let old = |p: Option<&str>| p == Some("1");
    let changes = check(env, Old::Verbatim, old, |s| {
        GdnPrefillPlan::from_settings(s).remnants_killed
    });
    assert!(changes.is_empty(), "{changes:?}");
    for probe in ["", " 1", "01", "yes", "TRUE"] {
        assert_eq!(
            plan_of(&[(env, probe)]).map(|p| p.remnants_killed),
            Some(old(Some(probe))),
            "{probe:?}"
        );
    }
}

/// 2026-10-17: Every lever at the value its reader read unset as.
#[test]
fn the_declared_defaults_are_the_old_unset_values() {
    let plan = GdnPrefillPlan::from_settings(&defaults());
    assert_eq!(plan, GdnPrefillPlan::declared());
    assert_eq!(plan.spine, old_spine(None, None));
}
