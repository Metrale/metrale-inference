// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: [`legacy_env_drafter_settings_from`] against the `DrafterSettings` projection that
//! replaces it, over every value in `VALUES` (plus numeric edges): equal except for the listed
//! deviations, which are the behaviour change the 2c-device-wire commit makes when it deletes the
//! legacy reader.
//!
//! Owner: speculative.
//! Invariants: never reads or writes the process environment.

use metrale_settings::InferenceServerSettings;
use metrale_settings::groups::device::ProjectionError;
use metrale_settings::groups::device_drafter::{DflashRungTuning, DrafterSettings, MtpRungTuning};
use metrale_settings::groups::equivalence::{VALUES, Want, check_values, var_in, var_one};

use super::legacy_env_drafter_settings_from;

fn drafter(s: &InferenceServerSettings) -> Result<DrafterSettings, ProjectionError> {
    DrafterSettings::try_from(s)
}

/// 2026-10-17: The legacy reader's view of one field with only `env` set to the value.
fn legacy<T>(
    env: &'static str,
    field: impl Fn(&DrafterSettings) -> T,
) -> impl Fn(Option<&str>) -> T {
    move |v| field(&legacy_env_drafter_settings_from(var_one(env, v)))
}

/// 2026-10-17: [`VALUES`] plus `extra`.
fn with(extra: &[Option<&'static str>]) -> Vec<Option<&'static str>> {
    [VALUES.as_slice(), extra].concat()
}

/// 2026-10-17: An Int lever whose old reader parsed an unsigned type and defaulted on failure:
/// the non-numbers and the negative value now fail.
fn int_rejects<T>() -> Vec<(Option<&'static str>, Want<T>)> {
    ["true", "off", "garbage", "-1"]
        .map(|v| (Some(v), Want::Err))
        .into()
}

#[test]
fn an_empty_environment_reads_as_the_declared_settings() {
    assert_eq!(
        legacy_env_drafter_settings_from(var_in(&[])),
        DrafterSettings::declared().expect("declared defaults project")
    );
}

#[test]
fn presence_levers_are_on_for_any_value() {
    type Field = fn(&DrafterSettings) -> bool;
    let levers: [(&str, Field); 4] = [
        ("METRALE_MTP_STATIC_RUNG", |d| d.mtp_static_rung),
        ("METRALE_MTP_K_LADDER", |d| d.mtp_ladder_set),
        ("METRALE_DFLASH_STATIC_GAMMA", |d| d.dflash_static_gamma),
        ("METRALE_DFLASH_GAMMA_RESOLVER", |d| d.dflash_gamma_resolver),
    ];
    for (env, field) in levers {
        check_values(
            env,
            &with(&[Some("")]),
            legacy(env, field),
            |s| drafter(s).map(|d| field(&d)),
            &[],
        );
    }
}

#[test]
fn mtp_rung_overrides_ignore_non_positive_values_and_reject_non_numbers() {
    type Field = fn(&MtpRungTuning) -> Option<f64>;
    let levers: [(&str, Field); 6] = [
        ("METRALE_MTP_RUNG_ENTER", |t| t.enter),
        ("METRALE_MTP_RUNG_LEAVE", |t| t.leave),
        ("METRALE_MTP_RUNG_ALPHA", |t| t.alpha),
        ("METRALE_MTP_RUNG_ALPHA_SLOW", |t| t.alpha_slow),
        ("METRALE_MTP_RUNG_PROBE_TICKS", |t| t.probe_ticks),
        ("METRALE_MTP_RUNG_P1_TRIGGER", |t| t.p1_trigger),
    ];
    for (env, field) in levers {
        check_values(
            env,
            &with(&[Some("1.32"), Some("-1"), Some("inf"), Some("2048")]),
            legacy(env, |d| field(&d.mtp_rung)),
            |s| drafter(s).map(|d| field(&d.mtp_rung)),
            &[
                (Some("true"), Want::Err),
                (Some("off"), Want::Err),
                (Some("garbage"), Want::Err),
                (Some("inf"), Want::Err),
            ],
        );
    }
}

#[test]
fn dflash_rung_widths_reject_what_the_legacy_parse_defaulted() {
    type Field = fn(&DflashRungTuning) -> Option<usize>;
    let levers: [(&str, Field); 4] = [
        ("METRALE_DFLASH_RUNG_MULTI", |t| t.multi),
        ("METRALE_DFLASH_RUNG_NARROW", |t| t.narrow),
        ("METRALE_DFLASH_RUNG_WIDE", |t| t.wide),
        ("METRALE_DFLASH_RUNG_DWELL", |t| t.dwell),
    ];
    for (env, field) in levers {
        check_values(
            env,
            &with(&[Some("4"), Some("-1")]),
            legacy(env, |d| field(&d.dflash_rung)),
            |s| drafter(s).map(|d| field(&d.dflash_rung)),
            &int_rejects(),
        );
    }
}

#[test]
fn dflash_rung_thresholds_keep_their_lenient_parse() {
    type Field = fn(&DflashRungTuning) -> Option<u32>;
    let levers: [(&str, Field); 2] = [
        ("METRALE_DFLASH_RUNG_ENTER", |t| t.enter),
        ("METRALE_DFLASH_RUNG_LEAVE", |t| t.leave),
    ];
    for (env, field) in levers {
        check_values(
            env,
            &with(&[Some("60"), Some("-1"), Some("4294967296")]),
            legacy(env, |d| field(&d.dflash_rung)),
            |s| drafter(s).map(|d| field(&d.dflash_rung)),
            &[],
        );
    }
}

#[test]
fn mtp_gate_intervals_reject_what_the_legacy_parse_defaulted() {
    type Field = fn(&DrafterSettings) -> usize;
    let levers: [(&str, Field); 2] = [
        ("METRALE_MTP_GATE_REPROBE", |d| d.mtp_gate_reprobe),
        ("METRALE_MTP_GATE_REFRESH", |d| d.mtp_gate_refresh),
    ];
    for (env, field) in levers {
        check_values(
            env,
            &with(&[Some("64"), Some("-1")]),
            legacy(env, field),
            |s| drafter(s).map(|d| field(&d)),
            &int_rejects(),
        );
    }
}

#[test]
fn spec_entry_pin_keeps_the_legacy_parse() {
    const ENV: &str = "METRALE_SPEC_ENTRY_PIN";
    let field = |d: &DrafterSettings| d.spec_entry_pin_tokens;
    check_values(
        ENV,
        &with(&[Some("12"), Some("-3"), Some("4294967296")]),
        legacy(ENV, field),
        |s| drafter(s).map(|d| field(&d)),
        &[],
    );
}
