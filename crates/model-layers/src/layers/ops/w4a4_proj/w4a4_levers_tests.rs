// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `W4a4Levers::from_settings` against the environment readers it replaced, frozen
//! here (`old_*`, unchanged but for taking the raw value as a parameter and returning the panic
//! message as an error).
//!
//! Owner: model-layers ops.
//! Invariants: the probes that read differently are exactly the listed behaviour changes: the
//! tile factor and the persistent switch read and refuse as before (a refusal is now an error,
//! not a panic), and the audit moves from "set and non-empty" to a switch.

use super::W4a4Levers;
use crate::lever_equivalence::{Old, Outcome, check, defaults, settings_from};

/// 2026-10-17: `mx_plan::mx_nt`.
fn old_nt(v: Option<&str>) -> Result<u32, String> {
    match v {
        None | Some("") | Some("4") => Ok(4),
        Some("1") => Ok(1),
        Some(v) => Err(format!("METRALE_W4A4_MX_NT={v}: expected 1 or 4")),
    }
}

/// 2026-10-17: `mx_plan::mx_ps`.
fn old_ps(v: Option<&str>) -> Result<bool, String> {
    match v {
        None | Some("") | Some("1") => Ok(true),
        Some("0") => Ok(false),
        Some(v) => Err(format!("METRALE_W4A4_MX_PS={v}: expected 0 or 1")),
    }
}

/// 2026-10-17: `w4a4_proj::audit_enabled`.
fn old_audit(v: Option<&str>) -> bool {
    v.is_some_and(|v| !v.is_empty())
}

/// 2026-10-17: `field` of the levers of `s`, or the refusal's text.
fn new_field<T>(
    s: &metrale_settings::InferenceServerSettings,
    field: fn(&W4a4Levers) -> T,
) -> Result<T, String> {
    W4a4Levers::from_settings(s)
        .map(|l| field(&l))
        .map_err(|e| format!("{e:#}"))
}

/// 2026-10-17: The probes beyond `PROBES` that the two text readers name or trim.
const TEXT_PROBES: [&str; 7] = ["", "4", " 4", "4 ", "2", "01", "TRUE"];

#[test]
fn the_tile_factor_reads_and_refuses_as_before() {
    let env = "METRALE_W4A4_MX_NT";
    let changes = check(env, Old::Verbatim, old_nt, |s| new_field(s, |l| l.nt));
    assert!(changes.is_empty(), "{changes:?}");
    for probe in TEXT_PROBES {
        let s = settings_from(&[(env, probe)]).expect("a text row resolves");
        assert_eq!(new_field(&s, |l| l.nt), old_nt(Some(probe)), "{probe:?}");
    }
}

#[test]
fn the_persistent_switch_reads_and_refuses_as_before() {
    let env = "METRALE_W4A4_MX_PS";
    let changes = check(env, Old::Verbatim, old_ps, |s| new_field(s, |l| l.ps));
    assert!(changes.is_empty(), "{changes:?}");
    for probe in TEXT_PROBES {
        let s = settings_from(&[(env, probe)]).expect("a text row resolves");
        assert_eq!(new_field(&s, |l| l.ps), old_ps(Some(probe)), "{probe:?}");
    }
}

/// 2026-10-17: A refused tile factor fails the whole resolution, so no lever of a refused
/// setting reaches a launch.
#[test]
fn a_refused_tile_factor_refuses_the_ops_levers() {
    let s = settings_from(&[("METRALE_W4A4_MX_NT", "2")]).expect("a text row resolves");
    let err = crate::layers::ops::OpsLevers::from_settings(&s).expect_err("NT=2 is refused");
    assert_eq!(format!("{err:#}"), "METRALE_W4A4_MX_NT=2: expected 1 or 4");
    let err = crate::layers::ops::ModelLevers::from_settings(&s, true, false)
        .expect_err("the model levers refuse it too");
    assert_eq!(format!("{err:#}"), "METRALE_W4A4_MX_NT=2: expected 1 or 4");
}

#[test]
fn the_audit_moves_from_non_empty_to_a_switch() {
    let env = "METRALE_W4A4_PROJ_AUDIT";
    let changes = check(env, Old::Presence, old_audit, |s| {
        new_field(s, |l| l.audit).expect("the audit alone resolves")
    });
    assert_eq!(
        changes,
        [
            (env, Some("0"), Outcome::Changed),
            (env, Some("off"), Outcome::Changed),
            (env, Some("garbage"), Outcome::Rejected),
        ]
    );
    // 2026-10-17: Set but empty read as off; the switch rule refuses it.
    assert!(!old_audit(Some("")));
    assert!(settings_from(&[(env, "")]).is_err());
}

#[test]
fn the_declared_defaults_are_the_old_unset_values() {
    let d = W4a4Levers::from_settings(&defaults()).expect("the defaults resolve");
    assert_eq!(d, W4a4Levers::declared());
    assert_eq!(
        (Ok(d.nt), Ok(d.ps), d.audit),
        (old_nt(None), old_ps(None), old_audit(None))
    );
}
