// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Tests for the bring-up text: winners in both directions, the disclosed
//! asymmetries, and energy that was not measured.
//!
//! Owner: server CLI (`met ml-utils`).
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use serde_json::json;

use super::super::bring_up_conc::{Instrument, Rung, Section};
use super::super::bring_up_core::{CacheCheck, Engine, EngineKind, SCHEMA, Step, row};
use super::*;

fn ttft_row(bench: Bench, p50: f64, prompt: f64, verdict: CacheVerdict) -> Row {
    let m: BTreeMap<String, f64> = [
        ("samples", 3.0),
        ("median_ms", p50),
        ("p90_ms", p50 * 1.1),
        ("min_ms", p50 * 0.9),
        ("prompt_tokens", prompt),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    let mut r = row(
        &Step {
            bench,
            tokens: 1024,
        },
        3,
        &m,
        "completed".into(),
        None,
        None,
        json!({}),
    );
    r.cache = CacheCheck {
        counter: None,
        hit_tokens: None,
        verdict,
    };
    r
}

fn rung(conc: usize, tok_s: f64, j: Option<f64>) -> Rung {
    Rung {
        conc,
        comparable: true,
        tok_s: Some(tok_s),
        ttft_p50_ms: Some(300.0),
        tpot_p50_ms: Some(30.0),
        completion_tokens: Some(1024.0),
        window: Some((0.0, 1.0)),
        energy_j: j.map(|j| j * 1024.0),
        j_per_tok: j,
    }
}

fn record(label: &str, ttft: Vec<Row>, rungs: Vec<Rung>, hosts: &[&str]) -> Record {
    Record {
        schema: SCHEMA.to_string(),
        url: format!("http://{label}"),
        model: "GLM-5.3-Flash".into(),
        tokenizer: Some("/m/tokenizer.json".into()),
        tokenizer_sha256: Some("ab".repeat(32)),
        engine: Engine {
            kind: EngineKind::Unknown,
            label: label.into(),
            owned_by: None,
            version: None,
            max_model_len: None,
            serve_identity: None,
            cache_config: BTreeMap::new(),
        },
        started_at: 1,
        metrale_version: "0".into(),
        ttft,
        concurrency: Some(Section {
            instrument: Instrument {
                concs: vec![1, 16],
                isl: 128,
                osl: 1024,
                prompt_mode: "essay".into(),
                warmup: 1,
            },
            energy_hosts: hosts.iter().map(|h| h.to_string()).collect(),
            rungs,
            status: "completed".into(),
            run: json!({}),
        }),
    }
}

#[test]
fn lower_ttft_and_higher_tok_s_win_with_their_ratio() {
    assert_eq!(winner(Some(100.0), Some(200.0), "A", "B", false), "A 2.00x");
    assert_eq!(winner(Some(300.0), Some(150.0), "A", "B", false), "B 2.00x");
    assert_eq!(winner(Some(30.0), Some(20.0), "A", "B", true), "A 1.50x");
    assert_eq!(winner(Some(20.0), Some(30.0), "A", "B", true), "B 1.50x");
    assert_eq!(winner(Some(5.0), Some(5.0), "A", "B", true), "tie");
    assert_eq!(winner(None, Some(5.0), "A", "B", false), "-");
}

#[test]
fn compare_names_winners_in_both_sections_and_discloses_asymmetries() {
    let a = record(
        "vllm",
        vec![ttft_row(Bench::Cold, 900.0, 1030.0, CacheVerdict::Verified)],
        vec![rung(1, 30.0, Some(3.0)), rung(16, 140.0, Some(0.9))],
        &["localhost"],
    );
    let b = record(
        "metrale",
        vec![ttft_row(
            Bench::Cold,
            450.0,
            1031.0,
            CacheVerdict::NotExposed,
        )],
        vec![rung(1, 36.0, Some(3.3)), rung(16, 130.0, None)],
        &["localhost"],
    );
    let text = compare(&a, &b).unwrap();
    assert!(text.contains("metrale 2.00x"), "{text}");
    assert!(text.contains("server prompt tokens 1030 vs 1031"), "{text}");
    assert!(text.contains("prefix cache not exposed"), "{text}");
    let tok_c1 = text
        .lines()
        .find(|l| l.contains("tok/s") && l.trim_start().starts_with("1 "))
        .unwrap();
    assert!(tok_c1.contains("metrale 1.20x"), "{tok_c1}");
    let j_c1 = text
        .lines()
        .find(|l| l.contains("J/tok") && l.trim_start().starts_with("1 "))
        .unwrap();
    assert!(j_c1.contains("vllm 1.10x"), "{j_c1}");
    let j_c16 = text
        .lines()
        .find(|l| l.contains("J/tok") && l.trim_start().starts_with("16 "))
        .unwrap();
    assert!(
        j_c16.trim_end().ends_with('-'),
        "a missing J/tok has no winner: {j_c16}"
    );
}

#[test]
fn compare_refuses_two_records_with_one_label_and_flags_instrument_drift() {
    let a = record("x", vec![], vec![rung(1, 30.0, None)], &[]);
    assert!(compare(&a, &a.clone()).is_err());
    let mut b = record("y", vec![], vec![rung(1, 30.0, None)], &[]);
    b.concurrency.as_mut().unwrap().instrument.osl = 512;
    let text = compare(&a, &b).unwrap();
    assert!(text.contains("concurrency instrument:"), "{text}");
}

#[test]
fn a_table_says_not_measured_for_energy_never_zero() {
    let r = record(
        "metrale",
        vec![ttft_row(
            Bench::HighIslWarm,
            120.0,
            32780.0,
            CacheVerdict::Verified,
        )],
        vec![rung(1, 36.0, None)],
        &[],
    );
    let text = table(&r);
    assert!(text.contains("high-isl warm @1024"), "{text}");
    assert!(text.contains("energy not measured"), "{text}");
    assert!(
        text.lines()
            .any(|l| l.trim_start().starts_with("1 ") && l.contains("not measured")),
        "{text}"
    );
    assert!(!text.contains("0.000"), "{text}");
}
