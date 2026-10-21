// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `KdaLevers::from_settings` against the environment readers it replaced, frozen
//! here (`old_*`, unchanged but for returning `None` where they returned an error).
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants: the probes that read differently are exactly the listed behaviour changes: a
//! strict `0`/`1` switch now also reads the other switch spellings (`true`, `off`), an `== "1"`
//! switch now also arms at `true` and refuses a value the switch rule cannot read, and a floor is
//! trimmed before it is parsed.

use super::*;
use metrale_model_layers::lever_equivalence::{Old, Outcome, check, defaults, settings_from};

/// 2026-10-17: `chunk16::parse_min_rows` and `chunk64::parse_min_rows64`, the floor a parameter.
fn old_floor(v: Option<&str>, min: usize) -> Option<Option<usize>> {
    match v {
        None | Some("0") => Some(None),
        Some(s) => match s.parse::<usize>() {
            Ok(r) if r >= min => Some(Some(r)),
            _ => None,
        },
    }
}

/// 2026-10-17: `parse_conv_pack`, `parse_chunk64_conv` and `parse_gate_in_prepare`.
fn old_strict(v: Option<&str>) -> Option<bool> {
    match v {
        None | Some("0") => Some(false),
        Some("1") => Some(true),
        Some(_) => None,
    }
}

/// 2026-10-17: The `kda_seq_rows` reader.
fn old_seq_rows(v: Option<&str>) -> KdaSeqRows {
    match v {
        Some("1") => KdaSeqRows::All,
        Some("decode") => KdaSeqRows::DecodeOnly,
        _ => KdaSeqRows::Off,
    }
}

/// 2026-10-17: `env` at `probe` and nothing else, read through `field`; `None` when the settings
/// or the levers refuse it.
fn new_at<T>(env: &str, probe: Option<&str>, field: impl Fn(&KdaLevers) -> T) -> Option<T> {
    let pairs: Vec<(&str, &str)> = probe.map(|v| (env, v)).into_iter().collect();
    settings_from(&pairs)
        .ok()
        .and_then(|s| KdaLevers::from_settings(&s).ok())
        .map(|l| field(&l))
}

type Field<T> = (&'static str, fn(&KdaLevers) -> T);

#[test]
fn the_strict_switches_read_as_before_and_take_the_switch_spellings() {
    let cases: [Field<bool>; 3] = [
        ("METRALE_GLM_KDA_CHUNK64_CONV", |l| l.chunk64_conv),
        ("METRALE_GLM_KDA_CONV_PACK", |l| l.conv_pack),
        ("METRALE_GLM_KDA_GATE_IN_PREPARE", |l| l.gate_in_prepare),
    ];
    for (env, field) in cases {
        for probe in [
            None,
            Some("0"),
            Some("1"),
            Some("garbage"),
            Some(""),
            Some("2"),
        ] {
            assert_eq!(
                new_at(env, probe, field),
                old_strict(probe),
                "{env}={probe:?}"
            );
        }
        for (probe, now) in [("true", true), ("on", true), (" 1", true), ("off", false)] {
            assert_eq!(
                (old_strict(Some(probe)), new_at(env, Some(probe), field)),
                (None, Some(now)),
                "{env}={probe:?}"
            );
        }
    }
}

#[test]
fn the_opt_in_switches_read_as_before_except_the_switch_spellings() {
    let cases: [Field<bool>; 6] = [
        ("METRALE_GLM_KDA_GATES_FUSED", |l| l.gates_fused),
        ("METRALE_GLM_KDA_NO_SMEM", |l| l.no_smem),
        ("METRALE_GLM_KDA_ROW_REG", |l| l.row_reg),
        ("METRALE_GLM_KDA_ROWS_REG", |l| l.rows_reg),
        ("METRALE_GLM_KDA_SEQ_TOKENS", |l| l.seq_tokens),
        ("METRALE_GLM_KDA_VERIFY_TOKENS", |l| l.verify_tokens),
    ];
    for (env, field) in cases {
        let changes = check(
            env,
            Old::OptIn,
            |probe| probe == Some("1"),
            |s| field(&KdaLevers::from_settings(s).expect("one switch resolves")),
        );
        assert_eq!(
            changes,
            [
                (env, Some("true"), Outcome::Changed),
                (env, Some("garbage"), Outcome::Rejected)
            ],
            "{env}"
        );
    }
}

#[test]
fn the_floors_read_as_before_but_trim() {
    let cases: [(Field<Option<usize>>, usize); 2] = [
        (("METRALE_GLM_KDA_CHUNK_SCAN", |l| l.chunk_scan), CHUNK16),
        (("METRALE_GLM_KDA_CHUNK64", |l| l.chunk64), CHUNK64),
    ];
    for ((env, field), min) in cases {
        let (below, at) = ((min - 1).to_string(), min.to_string());
        for probe in [
            None,
            Some("0"),
            Some("1"),
            Some(below.as_str()),
            Some(at.as_str()),
            Some("4096"),
            Some("-1"),
            Some("garbage"),
            Some(""),
            Some("true"),
        ] {
            assert_eq!(
                new_at(env, probe, field),
                old_floor(probe, min),
                "{env}={probe:?}"
            );
        }
        let padded = format!(" {min}");
        assert_eq!(
            (
                old_floor(Some(&padded), min),
                new_at(env, Some(&padded), field)
            ),
            (None, Some(Some(min))),
            "{env}"
        );
    }
}

#[test]
fn the_seq_rows_text_reads_as_before() {
    let env = "METRALE_GLM_KDA_SEQ_ROWS";
    let changes = check(env, Old::Verbatim, old_seq_rows, |s| {
        KdaLevers::from_settings(s)
            .expect("any text resolves")
            .seq_rows
    });
    assert!(changes.is_empty(), "{changes:?}");
    for probe in ["decode", "DECODE", " 1", "", "2"] {
        assert_eq!(
            new_at(env, Some(probe), |l| l.seq_rows),
            Some(old_seq_rows(Some(probe))),
            "{probe:?}"
        );
    }
}

/// 2026-10-17: Every lever off at the declared defaults, as each old reader read unset.
#[test]
fn the_declared_defaults_are_the_old_unset_values() {
    let l = KdaLevers::from_settings(&defaults()).expect("defaults resolve");
    assert_eq!(
        l,
        KdaLevers {
            chunk_scan: None,
            chunk64: None,
            chunk64_conv: false,
            conv_pack: false,
            conv_snap_to: true,
            gate_in_prepare: false,
            gates_fused: false,
            gates_rows: true,
            no_smem: false,
            row_reg: false,
            rows_reg: false,
            seq_rows: KdaSeqRows::Off,
            seq_tokens: false,
            verify_tokens: false,
        }
    );
}
