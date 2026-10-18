// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for what a gate record discloses about its server's
//! environment: `serve_env`, and the `perf_env` switch defaults checked
//! against each lever's source.
//!
//! Owner: bench gate (records).
//! Invariants: none beyond the types.

use super::super::GateRecord;
use super::super::tests::{SHA, hw, run_record};
use crate::result::Verdict;
use std::collections::BTreeMap;

/// 2026-09-26: `with_serve_env` records the applied lever set and resolves
/// `perf_env` through it first: a declared
/// `METRALE_PREFILL_CODISPATCH_WINDOW_MS=50` that this process lacks is
/// recorded as `50`, not `100`. Without it the field is empty and omitted.
#[test]
fn the_record_discloses_the_applied_serve_env_and_reads_perf_env_through_it() {
    // 2026-09-26: Control: without an applied set the value is this
    // process's, which the test does not export.
    let bare = GateRecord::from_run(
        &run_record(BTreeMap::new(), Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        Some("qwen3.8/qwen3.8-27b-nvfp4-unsloth".into()),
    )
    .unwrap();
    assert!(bare.serve_env.is_empty());
    assert_eq!(
        bare.perf_env
            .get("METRALE_PREFILL_CODISPATCH_WINDOW_MS")
            .map(String::as_str),
        Some("100"),
        "the test environment must not export the control this test flips"
    );
    assert!(
        serde_json::to_value(&bare)
            .unwrap()
            .get("serve_env")
            .is_none()
    );

    let applied: BTreeMap<String, String> = [
        ("METRALE_FP8_ROWWISE", "1"),
        ("METRALE_MTP_DCUT_RATIO", "1.0"),
        ("METRALE_MTP_K_LADDER", "1:3,2:1,4:2,8:2,16:1"),
        ("METRALE_PREFILL_CODISPATCH", "1"),
        ("METRALE_PREFILL_CODISPATCH_WINDOW_MS", "50"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    let record = bare.clone().with_serve_env(applied.clone());
    assert_eq!(record.serve_env, applied);
    assert_eq!(
        record
            .perf_env
            .get("METRALE_PREFILL_CODISPATCH_WINDOW_MS")
            .map(String::as_str),
        Some("50"),
        "the applied set is what the server read, so it is what the record says"
    );
    assert_eq!(
        record
            .perf_env
            .get("METRALE_PREFILL_CODISPATCH_SETTLE_MS")
            .map(String::as_str),
        Some("10"),
        "a control the set does not name still resolves to its default"
    );
    // 2026-09-26: An applied `METRALE_PREFILL_CODISPATCH` is disclosed in
    // `serve_env` only, never in `perf_env`.
    assert!(
        !record.perf_env.contains_key("METRALE_PREFILL_CODISPATCH"),
        "{:?}",
        record.perf_env
    );
    assert_eq!(record.serve_env["METRALE_PREFILL_CODISPATCH"], "1");
    let json = serde_json::to_value(&record).unwrap();
    assert_eq!(json["serve_env"]["METRALE_FP8_ROWWISE"], "1");
    assert_eq!(
        json["serve_env"]["METRALE_MTP_K_LADDER"],
        "1:3,2:1,4:2,8:2,16:1"
    );
    let back: GateRecord = serde_json::from_value(json).unwrap();
    assert_eq!(back.serve_env, applied);
    assert_eq!(back.perf_env, record.perf_env);
}

/// 2026-09-26: A set `METRALE_NO_W4A16_TC` or `METRALE_W4A16_TC_WIDE` is
/// recorded verbatim; `0` is a set value for these presence switches.
#[test]
fn a_set_tc_kill_switch_is_disclosed_verbatim() {
    for var in ["METRALE_NO_W4A16_TC", "METRALE_W4A16_TC_WIDE"] {
        for v in ["1", "0"] {
            let resolved = super::resolve_perf_env(|k| (k == var).then(|| v.to_string()));
            assert_eq!(resolved.get(var).map(String::as_str), Some(v));
        }
    }
}

/// 2026-10-17: The source of the `layers::ops` lever resolution, where the
/// tensor-core kill switches, the widening and their shared rule live.
fn ops_levers_src() -> String {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = root.join("crates/model-layers/src/layers/ops/ops_levers.rs");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// 2026-10-17: The rule `OpsLevers::from_settings` reads the tensor-core
/// switches through: unset or empty reads as unset, any other value as set.
fn assert_unset_or_empty_rule(src: &str) {
    let rule = src
        .find("fn unset_or_empty(")
        .map(|at| src[at..].chars().take(120).collect::<String>())
        .expect("the tensor-core switch rule is gone; PERF_CONTROLS discloses a dead variable");
    assert!(
        rule.contains("v.is_none_or(str::is_empty)"),
        "the tensor-core switch rule changed; the record's \"unset\" defaults assume unset or \
         empty means unset: {rule}"
    );
}

/// 2026-09-26: The `unset` defaults are right only while the lever treats
/// unset or empty `METRALE_NO_W4A16_TC` as on and only a non-empty
/// `METRALE_W4A16_TC_WIDE` as on. Reads `ops_levers.rs` so a rule change there
/// fails here.
#[test]
fn tc_kill_switch_default_matches_the_lever() {
    let src = ops_levers_src();
    assert_unset_or_empty_rule(&src);
    assert!(
        src.contains("let tc = unset_or_empty(k.no_w4a16_tc.as_deref());"),
        "the tc kill switch no longer reads through the rule; the record's \"unset\" default \
         assumes unset or empty means the tensor-core path ran"
    );
    assert!(
        src.contains("wide: tc && !unset_or_empty(k.w4a16_tc_wide.as_deref()),"),
        "the widening opt-in rule changed; the record's \"unset\" default assumes unset or \
         empty means the widening did NOT run"
    );
    let resolved = super::resolve_perf_env(|_| None);
    for var in ["METRALE_NO_W4A16_TC", "METRALE_W4A16_TC_WIDE"] {
        assert_eq!(resolved.get(var).map(String::as_str), Some("unset"));
    }
}

/// 2026-09-26: The `unset` default for `METRALE_NO_MTP_TC` is right only while
/// the lever treats unset or empty as on and any other value, `0` included,
/// as off. Reads `ops_levers.rs`; a set value is recorded verbatim.
#[test]
fn tc_mtp_switch_default_matches_the_lever() {
    let src = ops_levers_src();
    assert_unset_or_empty_rule(&src);
    assert!(
        src.contains("mtp_tc: unset_or_empty(s.speculative.no_mtp_tc.as_deref()),"),
        "the lever no longer reads METRALE_NO_MTP_TC through the rule; the record's \"unset\" \
         default assumes unset or empty means the tensor-core path ran"
    );
    assert_eq!(
        super::resolve_perf_env(|_| None)
            .get("METRALE_NO_MTP_TC")
            .map(String::as_str),
        Some("unset")
    );
    for v in ["1", "0"] {
        let resolved =
            super::resolve_perf_env(|k| (k == "METRALE_NO_MTP_TC").then(|| v.to_string()));
        assert_eq!(
            resolved.get("METRALE_NO_MTP_TC").map(String::as_str),
            Some(v)
        );
    }
}

/// 2026-09-26: The `unset` default for `METRALE_NO_BORROW_STREAK_LIMIT` is
/// right only while `streak_limit_from` treats unset or empty as on. Reads
/// `borrow_streak.rs`; a set value is recorded verbatim.
#[test]
fn borrow_streak_switch_default_matches_the_lever() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = root.join("crates/model-engine/src/model/trait_impl/borrow_streak.rs");
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let rule = src
        .find("pub fn streak_limit_from(")
        .map(|at| src[at..].chars().take(200).collect::<String>())
        .expect("the borrow-streak rule is gone; PERF_CONTROLS discloses a dead variable");
    assert!(
        rule.contains("kill.is_none_or(|v| v.is_empty())"),
        "the METRALE_NO_BORROW_STREAK_LIMIT rule changed; the record's \"unset\" default \
         assumes unset or empty means the guard ran: {rule}"
    );
    assert!(
        src.contains("streak_limit_from(std::env::var_os(\"METRALE_NO_BORROW_STREAK_LIMIT\")"),
        "the guard no longer reads METRALE_NO_BORROW_STREAK_LIMIT through streak_limit_from"
    );
    assert_eq!(
        super::resolve_perf_env(|_| None)
            .get("METRALE_NO_BORROW_STREAK_LIMIT")
            .map(String::as_str),
        Some("unset")
    );
    for v in ["1", "0"] {
        let resolved = super::resolve_perf_env(|k| {
            (k == "METRALE_NO_BORROW_STREAK_LIMIT").then(|| v.to_string())
        });
        assert_eq!(
            resolved
                .get("METRALE_NO_BORROW_STREAK_LIMIT")
                .map(String::as_str),
            Some(v)
        );
    }
}

/// 2026-10-17: Every perf control is a declared lever, and an unset one discloses that
/// lever's declared default: the two co-dispatch timings as numbers, the switches as
/// `unset`.
#[test]
fn every_perf_control_discloses_its_declared_default_when_unset() {
    let resolved = super::resolve_perf_env(|_| None);
    assert_eq!(resolved.len(), super::PERF_CONTROLS.len());
    for key in super::PERF_CONTROLS {
        let lever = metrale_config::levers::lookup(key)
            .unwrap_or_else(|| panic!("{key} is not a declared lever"));
        assert_eq!(
            resolved.get(key).map(String::as_str),
            Some(lever.default),
            "{key}"
        );
    }
    assert_eq!(
        resolved
            .get("METRALE_PREFILL_CODISPATCH_SETTLE_MS")
            .map(String::as_str),
        Some("10")
    );
    assert_eq!(
        resolved.get("METRALE_NO_MTP_TC").map(String::as_str),
        Some("unset")
    );
}
