// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `Qwen4ExpLevers::from_settings` against the environment reads of
//! `qwen4_exp.rs`, `ffn.rs`, `ple.rs` and `aux_sites.rs` it replaced, frozen here.
//!
//! Owner: model-arch weight loader (qwen4_exp).
//! Invariants: the probes that read differently are exactly the listed behaviour changes: a
//! strict-`1` switch now also arms at `true`, the BF16 GDN opt-out now also turns off at `off`,
//! a value the switch rule cannot read is refused at startup, and
//! `METRALE_HOLO_MOE_GROUPED_CUTLASS=true` now arms (`shared_levers_tests`).

use super::*;
use metrale_model_layers::lever_equivalence::{Old, Outcome, PROBES, check, settings_from};

/// 2026-10-17: The reads `from_settings` replaced, unchanged but for their paths.
fn from_values(value: impl Fn(&str) -> Option<String>) -> Qwen4ExpLevers {
    let is_one = |k: &str| value(k).as_deref() == Some("1");
    let attach_qsa = value("METRALE_QSA_DISABLE").as_deref() != Some("1");
    Qwen4ExpLevers {
        ple_max_tokens: value("METRALE_PLE_MAX_TOKENS")
            .and_then(|v| v.parse().ok())
            .unwrap_or(2048),
        ple_cache_slots: value("METRALE_PLE_CACHE_SLOTS")
            .and_then(|v| v.parse().ok())
            .unwrap_or(65536),
        no_ple: is_one("METRALE_QWEN4EXP_NO_PLE"),
        bf16_gdn: value("METRALE_QWEN4EXP_BF16_GDN").as_deref() != Some("0"),
        grouped_cutlass: is_one("METRALE_HOLO_MOE_GROUPED_CUTLASS"),
        qsa_disable: !attach_qsa,
    }
}

const OPT_IN: [&str; 1] = ["METRALE_QWEN4EXP_NO_PLE"];
const VERBATIM: [&str; 2] = ["METRALE_PLE_MAX_TOKENS", "METRALE_PLE_CACHE_SLOTS"];
const BF16_GDN: &str = "METRALE_QWEN4EXP_BF16_GDN";
const QSA: &str = "METRALE_QSA_DISABLE";
const GROUPED: &str = "METRALE_HOLO_MOE_GROUPED_CUTLASS";

fn old_at(env: &str, probe: Option<&str>) -> Qwen4ExpLevers {
    from_values(|var| (var == env).then(|| probe.map(str::to_string)).flatten())
}

fn changes(env: &'static str, old: Old) -> Vec<(&'static str, Option<&'static str>, Outcome)> {
    check(env, old, |p| old_at(env, p), Qwen4ExpLevers::from_settings)
}

#[test]
fn every_lever_reads_as_before_except_the_switch_spellings() {
    for env in OPT_IN {
        assert_eq!(
            changes(env, Old::OptIn),
            [
                (env, Some("true"), Outcome::Changed),
                (env, Some("garbage"), Outcome::Rejected)
            ],
            "{env}"
        );
    }
    assert_eq!(
        changes(BF16_GDN, Old::OptOut),
        [
            (BF16_GDN, Some("off"), Outcome::Changed),
            (BF16_GDN, Some("garbage"), Outcome::Rejected)
        ]
    );
    for env in VERBATIM {
        assert_eq!(changes(env, Old::Verbatim), [], "{env}");
    }
}

/// 2026-10-17: The old QSA read attached the indexer unless the value was exactly `1`; the
/// field is its negation.
#[test]
fn the_qsa_kill_switch_reads_as_before_except_true() {
    assert_eq!(
        check(
            QSA,
            Old::OptIn,
            |p| old_at(QSA, p).qsa_disable,
            |s| Qwen4ExpLevers::from_settings(s).qsa_disable
        ),
        [
            (QSA, Some("true"), Outcome::Changed),
            (QSA, Some("garbage"), Outcome::Rejected)
        ]
    );
}

/// 2026-10-17: The grouped-CUTLASS lever differs from the strict-`1` read only at `true`.
#[test]
fn grouped_cutlass_differs_only_at_true() {
    for probe in PROBES.into_iter().chain([Some(""), Some("TRUE")]) {
        let pairs: Vec<(&str, &str)> = probe.map(|v| (GROUPED, v)).into_iter().collect();
        let new = Qwen4ExpLevers::from_settings(&settings_from(&pairs).expect("text lever"));
        assert_eq!(
            new == old_at(GROUPED, probe),
            probe != Some("true"),
            "{probe:?}"
        );
    }
}

/// 2026-10-17: The counts at values where a parse could part from the old one.
#[test]
fn the_counts_read_as_before_at_edge_values() {
    for env in VERBATIM {
        for v in ["", "0", "4096", " 4096", "-1", "2.5", "1e3"] {
            let s = settings_from(&[(env, v)]).expect("a text lever takes any value");
            assert_eq!(
                Qwen4ExpLevers::from_settings(&s),
                old_at(env, Some(v)),
                "{env}={v:?}"
            );
        }
    }
}

/// 2026-10-17: The probe lists cover every variable the old reads read.
#[test]
fn the_probe_lists_cover_every_lever_the_old_reads_read() {
    let read = core::cell::RefCell::new(Vec::new());
    let _ = from_values(|var| {
        read.borrow_mut().push(var.to_string());
        None
    });
    let mut read = read.into_inner();
    read.sort();
    read.dedup();
    let mut listed: Vec<String> = OPT_IN
        .iter()
        .chain(&VERBATIM)
        .chain(&[BF16_GDN, QSA, GROUPED])
        .map(|v| (*v).to_string())
        .collect();
    listed.sort();
    assert_eq!(read, listed);
}
