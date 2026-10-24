// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Each `EngineLevers` and `SsmTierGates` field against a frozen copy of the
//! environment reader it replaced, through `MapEnv`-resolved settings (never the process
//! environment).
//!
//! Owner: model-engine.
//! Invariants: every probe that reads differently from the old reader is asserted by name.

use metrale_model_layers::lever_equivalence::{Old, Outcome, check, settings_from};

use super::EngineLevers;
use crate::model::prefill_zero::KvZeroBlocks;
use crate::model::ssm_spill_gate::SsmTierGates;

fn levers(pairs: &[(&str, &str)]) -> EngineLevers {
    EngineLevers::from_settings(&settings_from(pairs).expect("resolves")).expect("valid")
}

/// 2026-10-17: A Text lever reads every probe as the old reader did, `""` included.
fn same_on_text<T: PartialEq + core::fmt::Debug>(
    env: &'static str,
    old: impl Fn(Option<&str>) -> T,
    new: impl Fn(&EngineLevers) -> T,
    extra: &[&'static str],
) {
    let changes = check(env, Old::Verbatim, &old, |s| {
        new(&EngineLevers::from_settings(s).unwrap())
    });
    assert!(changes.is_empty(), "{changes:?}");
    for v in extra {
        assert_eq!(new(&levers(&[(env, v)])), old(Some(v)), "{env}={v:?}");
    }
}

#[test]
fn lm_head_m16_tc_n_tile_matches_the_old_reader() {
    // 2026-10-17: Frozen `m16_tc_n_tile_from_value(env::var(..).ok())`.
    let old = |v: Option<&str>| match v.map(str::trim) {
        Some("64") => 64u32,
        _ => 32,
    };
    same_on_text(
        "METRALE_LM_HEAD_M16_TC_NTILE",
        old,
        |l| l.lm_head_m16_tc_n_tile,
        &["64", " 64 ", "", "32"],
    );
}

#[test]
fn lmhead_batch_gemv_matches_the_old_reader() {
    let old = |v: Option<&str>| v != Some("0");
    same_on_text(
        "METRALE_LMHEAD_BATCH_GEMV",
        old,
        |l| l.lmhead_batch_gemv,
        &["", " 0"],
    );
}

#[test]
fn borrow_streak_limit_matches_the_old_reader() {
    // 2026-10-17: Frozen `var_os(..).is_none_or(|v| v.is_empty())`.
    let old = |v: Option<&str>| v.is_none_or(str::is_empty);
    same_on_text(
        "METRALE_NO_BORROW_STREAK_LIMIT",
        old,
        |l| l.borrow_streak_limit,
        &["", "yes", "unset"],
    );
}

#[test]
fn glm_vision_check_matches_the_old_reader() {
    let old = |v: Option<&str>| {
        matches!(
            v.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
            Some("1" | "true" | "yes" | "on")
        )
    };
    same_on_text(
        "METRALE_GLM_VISION_CHECK",
        old,
        |l| l.glm_vision_check,
        &["", " YES ", "On", "2"],
    );
}

#[test]
fn kv_zero_blocks_matches_the_old_reader() {
    // 2026-10-17: Frozen parse; an unrecognised value was an error at first use and is now a
    // model-build error, so both sides read it as `None`.
    let old = |v: Option<&str>| match v {
        None | Some("0") | Some("block") => Some(KvZeroBlocks::Block),
        Some("range") => Some(KvZeroBlocks::Range),
        Some(_) => None,
    };
    let new = |s: &metrale_settings::InferenceServerSettings| {
        EngineLevers::from_settings(s)
            .ok()
            .map(|l| l.kv_zero_blocks)
    };
    assert!(check("METRALE_KV_ZERO_BLOCKS", Old::Verbatim, old, new).is_empty());
    for v in ["range", "block", "", "Range", "unset"] {
        let s = settings_from(&[("METRALE_KV_ZERO_BLOCKS", v)]).unwrap();
        assert_eq!(new(&s), old(Some(v)), "{v:?}");
    }
}

#[test]
fn redzone_every_matches_the_old_reader_where_both_parse() {
    // 2026-10-17: Frozen `redzone_every()`.
    let old = |redzone: Option<&str>, every: Option<&str>| -> usize {
        if redzone.is_none() {
            return 0;
        }
        every.and_then(|v| v.parse::<usize>().ok()).unwrap_or(1)
    };
    for redzone in [None, Some("0"), Some("4096")] {
        for every in [None, Some("0"), Some("1"), Some("5"), Some("-3")] {
            let pairs: Vec<(&str, &str)> = [
                ("METRALE_REDZONE", redzone),
                ("METRALE_REDZONE_EVERY", every),
            ]
            .into_iter()
            .filter_map(|(k, v)| v.map(|v| (k, v)))
            .collect();
            assert_eq!(
                levers(&pairs).redzone_every,
                old(redzone, every),
                "{redzone:?} {every:?}"
            );
        }
    }
}

/// 2026-10-17: Values the old reader took silently that are now startup errors: a
/// non-integer `METRALE_REDZONE_EVERY` (was 1) and a non-integer `METRALE_REDZONE`, `""`
/// included (was "set", so scanning every step).
#[test]
fn redzone_garbage_is_rejected() {
    for pairs in [
        [
            ("METRALE_REDZONE", "1"),
            ("METRALE_REDZONE_EVERY", "garbage"),
        ],
        [("METRALE_REDZONE", ""), ("METRALE_REDZONE_EVERY", "1")],
        [("METRALE_REDZONE", "on"), ("METRALE_REDZONE_EVERY", "1")],
    ] {
        assert!(settings_from(&pairs).is_err(), "{pairs:?}");
    }
}

fn gates(s: &metrale_settings::InferenceServerSettings) -> SsmTierGates {
    SsmTierGates::resolve(
        s.kernels.ssm_spill_min_tokens.as_deref(),
        s.kernels.ssm_fault_min_tokens.as_deref(),
        s.telemetry.ssm_tier_timing,
    )
}

#[test]
fn ssm_tier_gates_match_the_old_readers() {
    // 2026-10-17: Frozen `fault_in_min_tokens()` and `spill_min_tokens()` (parse, then clamp
    // against the fault gate).
    let fault_old = |v: Option<&str>| v.and_then(|v| v.parse::<usize>().ok()).unwrap_or(256);
    let spill_old = |v: Option<&str>, fault: usize| {
        let raw = v.and_then(|v| v.parse::<usize>().ok()).unwrap_or(1024);
        if raw == 0 { 0 } else { raw.max(fault) }
    };
    let fault_gate = |s: &metrale_settings::InferenceServerSettings| gates(s).fault_min_tokens;
    assert!(
        check(
            "METRALE_SSM_FAULT_MIN_TOKENS",
            Old::Verbatim,
            fault_old,
            fault_gate
        )
        .is_empty()
    );
    let sp = |v: Option<&str>| spill_old(v, 256);
    assert!(
        check("METRALE_SSM_SPILL_MIN_TOKENS", Old::Verbatim, sp, |s| {
            gates(s).spill_min_tokens
        })
        .is_empty()
    );
    let probes = [
        None,
        Some(""),
        Some("0"),
        Some("64"),
        Some("2048"),
        Some(" 8"),
        Some("x"),
    ];
    for spill in probes {
        for fault in probes {
            let pairs: Vec<(&str, &str)> = [
                ("METRALE_SSM_SPILL_MIN_TOKENS", spill),
                ("METRALE_SSM_FAULT_MIN_TOKENS", fault),
            ]
            .into_iter()
            .filter_map(|(k, v)| v.map(|v| (k, v)))
            .collect();
            let g = gates(&settings_from(&pairs).unwrap());
            assert_eq!(g.fault_min_tokens, fault_old(fault), "{spill:?} {fault:?}");
            assert_eq!(
                g.spill_min_tokens,
                spill_old(spill, fault_old(fault)),
                "{spill:?} {fault:?}"
            );
        }
    }
}

/// 2026-10-17: `METRALE_SSM_TIER_TIMING` stays a presence switch: `0` and `""` arm it.
#[test]
fn ssm_tier_timing_is_presence() {
    let changes = check(
        "METRALE_SSM_TIER_TIMING",
        Old::Presence,
        |v: Option<&str>| v.is_some(),
        |s| gates(s).timing,
    );
    assert_eq!(changes, Vec::<(&str, Option<&str>, Outcome)>::new());
    assert!(gates(&settings_from(&[("METRALE_SSM_TIER_TIMING", "")]).unwrap()).timing);
}
