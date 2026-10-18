// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `SsmLevers::from_settings` against the environment readers it replaced, frozen
//! here (`old_*` and the closures, unchanged but for taking the raw value as a parameter).
//!
//! Owner: model-layers (qwen3_ssm).
//! Invariants: the probes that read differently are exactly the listed behaviour changes: a `1`
//! switch now also reads the other switch spellings and refuses a value the switch rule cannot
//! read; the presence and text levers read as before.

use super::*;
use crate::lever_equivalence::{Old, Outcome, check, defaults, settings_from};

/// 2026-10-17: `trait_decode_multi_seq::ssm_ffn_prefill_min_n`.
fn old_ffn_prefill_min_n(v: Option<&str>) -> usize {
    v.and_then(|v| v.parse().ok())
        .filter(|&v| v >= 2)
        .unwrap_or(5)
}

/// 2026-10-17: `ssm_batched::ssm_tc_proj_min_n`.
fn old_tc_proj_min_n(v: Option<&str>) -> Option<usize> {
    match v {
        None => Some(9),
        Some("0") => None,
        Some("1") => Some(9),
        Some(v) => v.parse::<usize>().ok().filter(|&x| x >= 2),
    }
}

/// 2026-10-17: The levers of `env` at `probe` and nothing else; `None` when the settings refuse
/// it.
fn new_at(env: &str, probe: &str) -> Option<SsmLevers> {
    settings_from(&[(env, probe)])
        .ok()
        .map(|s| SsmLevers::from_settings(&s))
}

type Field = (&'static str, fn(&SsmLevers) -> bool);

/// 2026-10-17: The `var_os(..).is_none()` / `var(..).is_err()` kill switches and the two
/// `is_some()` / `is_ok()` opt-ins: set at any value, `0` and the empty string included.
#[test]
fn the_presence_levers_read_as_before() {
    let kills: [Field; 8] = [
        ("METRALE_NO_GDN_WY2_RESIDENT", |l| l.wy2_resident),
        ("METRALE_NO_GDN_WY3_RESIDENT", |l| l.wy3_resident),
        ("METRALE_NO_BATCHED_BA_GATES", |l| l.batched_ba_gates),
        ("METRALE_NO_BATCHED_GDN_NORM", |l| l.batched_gdn_norm),
        ("METRALE_NO_QKVZ_NVFP4_DECODE", |l| l.qkvz_nvfp4_decode),
        ("METRALE_NO_VERIFY_OUTPROJ_TGEMM", |l| {
            l.verify_outproj_tgemm
        }),
        ("METRALE_NO_VERIFY_GDN_BATCH", |l| l.verify_gdn_batch),
        ("METRALE_NO_SSM_M128", |l| l.m128_min_m.is_some()),
    ];
    let opt_ins: [Field; 2] = [
        ("METRALE_SSM_OUT_W8A16_ONLY", |l| l.out_w8a16_only),
        ("METRALE_GDN_SMEM_STAGE", |l| l.smem_stage),
    ];
    let all = kills
        .iter()
        .map(|&f| (f, false))
        .chain(opt_ins.iter().map(|&f| (f, true)));
    for ((env, field), set) in all {
        let old = move |p: Option<&str>| p.is_some() == set;
        let changes = check(env, Old::Presence, old, |s| {
            field(&SsmLevers::from_settings(s))
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

/// 2026-10-17: The `== "1"` readers that became switches: the two profilers, and the
/// `== "1"` / `!= "1"` kill switches of the FLA prefill, the batched dense FFN and the
/// half-register recurrence (each probed as the kill switch itself).
#[test]
fn the_opt_in_switches_read_as_before_except_the_switch_spellings() {
    let cases: [Field; 5] = [
        ("METRALE_QWEN4EXP_PREFILL_PROF", |l| l.hc_prefill_prof),
        ("METRALE_QWEN4EXP_DECODE_PROF", |l| l.hc_decode_prof),
        ("METRALE_NO_GDN_FLA", |l| !l.fla),
        ("METRALE_NO_SSM_FFN_PREFILL", |l| !l.ffn_prefill),
        ("METRALE_NO_GDN_HALF_REG", |l| !l.half_reg),
    ];
    for (env, field) in cases {
        let changes = check(
            env,
            Old::OptIn,
            |p| p == Some("1"),
            |s| field(&SsmLevers::from_settings(s)),
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

/// 2026-10-17: The text predicates: exactly `1` turns on the BF16 weights, the fused verify
/// and the forced W8A8 out_proj; exactly `0` turns off the fused conv.
#[test]
fn the_text_predicates_are_kept() {
    let cases: [(&str, fn(Option<&str>) -> bool, Field); 4] = [
        (
            "METRALE_GDN_BF16_WEIGHTS",
            |p| matches!(p, Some("1")),
            ("", |l| l.bf16_weights),
        ),
        (
            "METRALE_GDN_FUSED_VERIFY",
            |p| matches!(p, Some("1")),
            ("", |l| l.fused_verify),
        ),
        (
            "METRALE_FP8_W8A8",
            |p| matches!(p, Some("1")),
            ("", |l| l.force_w8a8),
        ),
        (
            "METRALE_GDN_FUSED_CONV17",
            |p| !matches!(p, Some("0")),
            ("", |l| l.fused_conv17),
        ),
    ];
    for (env, old, (_, field)) in cases {
        let changes = check(env, Old::Verbatim, old, |s| {
            field(&SsmLevers::from_settings(s))
        });
        assert!(changes.is_empty(), "{changes:?}");
        for probe in ["", " 1", " 0", "01", "00", "TRUE", "yes"] {
            assert_eq!(
                new_at(env, probe).map(|l| field(&l)),
                Some(old(Some(probe))),
                "{env}={probe:?}"
            );
        }
    }
}

/// 2026-10-17: The two row thresholds keep their parse, including the values outside the
/// probes: below 2, signed, padded and empty.
#[test]
fn the_row_thresholds_keep_their_parse() {
    const FFN: &str = "METRALE_SSM_FFN_PREFILL_MIN_N";
    const TC: &str = "METRALE_SSM_TC_PROJ";
    let changes = check(FFN, Old::Verbatim, old_ffn_prefill_min_n, |s| {
        SsmLevers::from_settings(s).ffn_prefill_min_n
    });
    assert!(changes.is_empty(), "{changes:?}");
    let changes = check(TC, Old::Verbatim, old_tc_proj_min_n, |s| {
        SsmLevers::from_settings(s).tc_proj_min_n
    });
    assert!(changes.is_empty(), "{changes:?}");
    for probe in ["", " 12", "+7", "-3", "2", "12", "18446744073709551616"] {
        assert_eq!(
            new_at(FFN, probe).map(|l| l.ffn_prefill_min_n),
            Some(old_ffn_prefill_min_n(Some(probe))),
            "{probe:?}"
        );
        assert_eq!(
            new_at(TC, probe).map(|l| l.tc_proj_min_n),
            Some(old_tc_proj_min_n(Some(probe))),
            "{probe:?}"
        );
    }
}

/// 2026-10-17: Every lever at the value its reader read unset as.
#[test]
fn the_declared_defaults_are_the_old_unset_values() {
    let levers = SsmLevers::from_settings(&defaults());
    assert_eq!(levers, SsmLevers::declared());
    assert_eq!(levers.ffn_prefill_min_n, old_ffn_prefill_min_n(None));
    assert_eq!(levers.tc_proj_min_n, old_tc_proj_min_n(None));
    assert_eq!(levers.m128_min_m, Some(65));
}
