// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `DrafterSettings` against the metrale-speculative readers it replaced.
//!
//! Owner: settings.
//! Invariants: never reads or writes the process environment.

use super::{DflashRungTuning, DrafterSettings, MtpRungTuning};
use crate::groups::device::ProjectionError;
use crate::groups::equivalence::{VALUES, Want, check, check_values};

fn drafter(s: &crate::InferenceServerSettings) -> Result<DrafterSettings, ProjectionError> {
    DrafterSettings::try_from(s)
}

fn with(extra: &[Option<&'static str>]) -> Vec<Option<&'static str>> {
    [VALUES.as_slice(), extra].concat()
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
        check(env, |v| v.is_some(), |s| drafter(s).map(|d| field(&d)), &[]);
    }
}

#[test]
fn mtp_rung_overrides_ignore_non_positive_values_and_reject_non_numbers() {
    // 2026-10-17: Old `adaptive_rung::tunable`, as an override: `None` keeps the constant.
    let old = |v: Option<&str>| {
        v.and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
    };
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
            old,
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
fn dflash_rung_widths_reject_what_the_old_parse_defaulted() {
    // 2026-10-17: Old `dflash_rung::env_usize`, as an override.
    let old = |v: Option<&str>| v.and_then(|v| v.parse::<usize>().ok());
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
            old,
            |s| drafter(s).map(|d| field(&d.dflash_rung)),
            &[
                (Some("true"), Want::Err),
                (Some("off"), Want::Err),
                (Some("garbage"), Want::Err),
                (Some("-1"), Want::Err),
            ],
        );
    }
}

#[test]
fn dflash_rung_thresholds_keep_their_lenient_parse() {
    let old = |v: Option<&str>| v.and_then(|v| v.parse::<u32>().ok());
    type Field = fn(&DflashRungTuning) -> Option<u32>;
    let levers: [(&str, Field); 2] = [
        ("METRALE_DFLASH_RUNG_ENTER", |t| t.enter),
        ("METRALE_DFLASH_RUNG_LEAVE", |t| t.leave),
    ];
    for (env, field) in levers {
        check_values(
            env,
            &with(&[Some("60"), Some("-1")]),
            old,
            |s| drafter(s).map(|d| field(&d.dflash_rung)),
            &[],
        );
    }
}

#[test]
fn mtp_gate_intervals_reject_what_the_old_parse_defaulted() {
    type Field = fn(&DrafterSettings) -> usize;
    let levers: [(&str, usize, Field); 2] = [
        ("METRALE_MTP_GATE_REPROBE", 256, |d| d.mtp_gate_reprobe),
        ("METRALE_MTP_GATE_REFRESH", 1024, |d| d.mtp_gate_refresh),
    ];
    for (env, default, field) in levers {
        check_values(
            env,
            &with(&[Some("64"), Some("-1")]),
            |v| v.and_then(|v| v.parse::<usize>().ok()).unwrap_or(default),
            |s| drafter(s).map(|d| field(&d)),
            &[
                (Some("true"), Want::Err),
                (Some("off"), Want::Err),
                (Some("garbage"), Want::Err),
                (Some("-1"), Want::Err),
            ],
        );
    }
}

#[test]
fn spec_entry_pin_keeps_the_old_parse() {
    // 2026-10-17: Old `mtp_gate::parse_entry_pin_tokens`.
    check_values(
        "METRALE_SPEC_ENTRY_PIN",
        &with(&[Some("12"), Some("-1")]),
        |v| v.and_then(|v| v.parse::<u32>().ok()).unwrap_or(8),
        |s| drafter(s).map(|d| d.spec_entry_pin_tokens),
        &[],
    );
}
