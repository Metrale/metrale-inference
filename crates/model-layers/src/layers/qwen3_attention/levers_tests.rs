// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `AttnLevers::from_settings` against the environment readers it replaced, frozen
//! here as the predicates the closures apply to the raw value.
//!
//! Owner: model-layers (attention).
//! Invariants: the probes that read differently are exactly the listed behaviour changes: a
//! switch now also reads the other switch spellings and refuses a value the switch rule cannot
//! read; the presence and text levers read as before.

use super::*;
use crate::lever_equivalence::{Old, Outcome, check, defaults, settings_from};

/// 2026-10-17: The levers of `env` at `probe` and nothing else; `None` when the settings refuse
/// it.
fn new_at(env: &str, probe: &str) -> Option<AttnLevers> {
    settings_from(&[(env, probe)])
        .ok()
        .map(|s| AttnLevers::from_settings(&s))
}

type Field = (&'static str, fn(&AttnLevers) -> bool);

/// 2026-10-17: The `var_os(..).is_none()` kill switch and the two `is_some()` / `is_ok()`
/// opt-ins: set at any value, `0` and the empty string included.
#[test]
fn the_presence_levers_read_as_before() {
    let cases: [(Field, bool); 3] = [
        (("METRALE_NO_FP8_QKV_BATCH", |l| l.fp8_qkv_batch), false),
        (("METRALE_ATTN_QKV_W8A16_ONLY", |l| l.qkv_w8a16_only), true),
        (("METRALE_ATTN_DBG", |l| l.attn_dbg), true),
    ];
    for ((env, field), set) in cases {
        let old = move |p: Option<&str>| p.is_some() == set;
        let changes = check(env, Old::Presence, old, |s| {
            field(&AttnLevers::from_settings(s))
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

/// 2026-10-17: The `== "1"` readers that became switches, the kill switches probed as the kill
/// switch itself (`!= "1"` keeps the path on).
#[test]
fn the_opt_in_switches_read_as_before_except_the_switch_spellings() {
    let cases: [Field; 6] = [
        ("METRALE_MLA_HSD", |l| l.mla_hsd),
        ("METRALE_PREFILL_HOST_TIMING", |l| l.prefill_host_timing),
        ("METRALE_NO_QK_NORM_STRIDED", |l| !l.qk_norm_strided),
        ("METRALE_ATTN_PREFILL_T_PIPE", |l| l.prefill_t_pipe),
        ("METRALE_FUSED_KV", |l| l.fused_kv),
        ("METRALE_FLASHINFER_PREFILL", |l| l.flashinfer_prefill),
    ];
    for (env, field) in cases {
        let changes = check(
            env,
            Old::OptIn,
            |p| p == Some("1"),
            |s| field(&AttnLevers::from_settings(s)),
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

/// 2026-10-17: `METRALE_DIAG_V4_ALL_LAYERS` was on at exactly `1` or `true`; the switch keeps
/// both and refuses the garbage it read as off.
#[test]
fn the_v4_diagnostics_read_as_before_except_the_switch_spellings() {
    const ENV: &str = "METRALE_DIAG_V4_ALL_LAYERS";
    let changes = check(
        ENV,
        Old::TruthyExact,
        |p| matches!(p, Some("1" | "true")),
        |s| AttnLevers::from_settings(s).diag_v4_all_layers,
    );
    assert_eq!(changes, [(ENV, Some("garbage"), Outcome::Rejected)]);
    for probe in ["on", "yes", "TRUE"] {
        assert_eq!(
            new_at(ENV, probe).map(|l| l.diag_v4_all_layers),
            Some(true),
            "{probe:?}"
        );
    }
}

/// 2026-10-18: `METRALE_FP8_KV_EMA_RECAL` was on at `1` or `true` in any case; the switch keeps
/// both and refuses the garbage it read as off.
#[test]
fn the_ema_recal_reads_as_before_except_the_switch_spellings() {
    const ENV: &str = "METRALE_FP8_KV_EMA_RECAL";
    let changes = check(
        ENV,
        Old::Truthy,
        |p| p.is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true")),
        |s| AttnLevers::from_settings(s).fp8_kv_ema_recal,
    );
    assert_eq!(changes, [(ENV, Some("garbage"), Outcome::Rejected)]);
    for probe in ["TRUE", "on", "yes"] {
        assert_eq!(
            new_at(ENV, probe).map(|l| l.fp8_kv_ema_recal),
            Some(true),
            "{probe:?}"
        );
    }
}

/// 2026-10-17: The text predicates: exactly `1` turns the fused Q-RoPE on, and exactly `4` picks
/// the four-column width.
#[test]
fn the_text_predicates_are_kept() {
    let cases: [(&str, fn(Option<&str>) -> bool, fn(&AttnLevers) -> bool); 2] = [
        (
            "METRALE_ATTN_PREFILL_FUSED_QROPE",
            |p| p == Some("1"),
            |l| l.prefill_fused_qrope,
        ),
        (
            "METRALE_ATTN_NCOL_WIDTH",
            |p| p == Some("4"),
            |l| l.ncol_width == NcolWidth::Four,
        ),
    ];
    for (env, old, field) in cases {
        let changes = check(env, Old::Verbatim, old, |s| {
            field(&AttnLevers::from_settings(s))
        });
        assert!(changes.is_empty(), "{changes:?}");
        for probe in ["", " 1", "01", "4", " 4", "04", "TRUE", "2"] {
            assert_eq!(
                new_at(env, probe).map(|l| field(&l)),
                Some(old(Some(probe))),
                "{env}={probe:?}"
            );
        }
    }
}

/// 2026-10-17: Every lever at the value its reader read unset as.
#[test]
fn the_declared_defaults_are_the_old_unset_values() {
    assert_eq!(
        AttnLevers::from_settings(&defaults()),
        AttnLevers::declared()
    );
}
