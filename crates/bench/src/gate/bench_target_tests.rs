// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: Tests for the prose that rides beside a target in BENCH.toml (`target_caveats`,
//! `target_note`, `target_max_model_len`): what the loader refuses, and that no verdict reads any.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use super::bench_tests::{fixture, verdict_at};
use super::*;

fn with_caveat(name: &str, target_note: &str, caveats: &str) -> std::path::PathBuf {
    fixture(
        name,
        &format!(
            r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "ttft-cold-gate"
default = true
status = "measured"
note = "the engineering account"
targets = {{ median_ms = 100.0 }}
target_note = "{target_note}"
{caveats}
[benchmarks.metrics.median_ms]
max = 500.0
"#
        ),
    )
}

/// 2026-10-16: A caveat must name a target, carry text, and sit on an entry with a `target_note`
/// for its marker to point to (the engineering `note` does not count); a caveat meeting all three
/// loads.
#[test]
fn a_target_caveat_must_name_a_target_have_text_and_a_note() {
    let refused = |name: &str, note: &str, caveats: &str| {
        format!(
            "{:#}",
            load_all(&with_caveat(name, note, caveats)).unwrap_err()
        )
    };
    let err = refused(
        "caveat-untargeted",
        "n",
        r#"target_caveats = { p90_ms = "stall" }"#,
    );
    assert!(
        err.contains("[benchmarks.target_caveats] names p90_ms, which is not a target"),
        "{err}"
    );
    let err = refused(
        "caveat-empty",
        "n",
        r#"target_caveats = { median_ms = "  " }"#,
    );
    assert!(err.contains("target caveat median_ms is empty"), "{err}");
    let err = refused(
        "caveat-no-note",
        "",
        r#"target_caveats = { median_ms = "stall" }"#,
    );
    assert!(
        err.contains("has target caveats but no target_note for them to point to"),
        "{err}"
    );
    let loaded = load_all(&with_caveat(
        "caveat-ok",
        "n",
        r#"target_caveats = { median_ms = "stall" }"#,
    ))
    .unwrap();
    assert_eq!(
        loaded[0]
            .1
            .target_caveats
            .get("median_ms")
            .map(String::as_str),
        Some("stall")
    );
}

/// 2026-10-16: A caveat is prose: the baseline every gate verdict reads, the verdict itself on
/// both sides of the bound, and the entry a battle judges are the same with and without it.
#[test]
fn a_target_caveat_never_moves_a_verdict() {
    let with = with_caveat(
        "caveat-verdict-with",
        "n",
        r#"target_caveats = { median_ms = "stall" }"#,
    );
    let without = with_caveat("caveat-verdict-without", "n", "");
    let baseline = |root: &Path| {
        serde_json::to_string(&baseline_for(root, "ttft-cold-gate").unwrap()).unwrap()
    };
    assert_eq!(baseline(&with), baseline(&without));
    for value in [100.0, 499.0, 501.0] {
        assert_eq!(
            verdict_at(&with, value),
            verdict_at(&without, value),
            "{value}"
        );
    }
    let entries = super::super::battle_load::entries;
    let mut carried = entries(&with).unwrap();
    assert_eq!(
        carried[0].caveats.get("median_ms").map(String::as_str),
        Some("stall"),
        "the loader hands the caveat to the battle"
    );
    carried[0].caveats.clear();
    assert_eq!(
        format!("{carried:?}"),
        format!("{:?}", entries(&without).unwrap()),
        "the caveat is the only difference a battle sees"
    );
}

fn with_reference_context(name: &str, targets: &str, context: &str) -> std::path::PathBuf {
    fixture(
        name,
        &format!(
            r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "ttft-cold-gate"
default = true
status = "measured"
{targets}
{context}
[benchmarks.metrics.median_ms]
max = 500.0
"#
        ),
    )
}

/// 2026-10-16: The reference run's context describes the targets, so an entry with none may not
/// declare it, and a context is positive; a positive one beside targets loads.
#[test]
fn a_reference_context_needs_targets_and_is_positive() {
    let targets = "targets = { median_ms = 100.0 }";
    let refused = |name: &str, targets: &str, context: &str| {
        format!(
            "{:#}",
            load_all(&with_reference_context(name, targets, context)).unwrap_err()
        )
    };
    let err = refused("refctx-untargeted", "", "target_max_model_len = 262144");
    assert!(
        err.contains("declares target_max_model_len with no targets for it to describe"),
        "{err}"
    );
    for (name, len) in [("refctx-zero", "0"), ("refctx-negative", "-1")] {
        let err = refused(name, targets, &format!("target_max_model_len = {len}"));
        assert!(
            err.contains(&format!("target_max_model_len must be positive, got {len}")),
            "{err}"
        );
    }
    let loaded = load_all(&with_reference_context(
        "refctx-ok",
        targets,
        "target_max_model_len = 262144",
    ))
    .unwrap();
    assert_eq!(loaded[0].1.target_max_model_len, Some(262144));
}

/// 2026-10-16: The reference run's context is prose: the baseline every gate verdict reads, the
/// verdict on both sides of the bound, and the entries a battle judges are the same with and
/// without it.
#[test]
fn a_reference_context_never_moves_a_verdict() {
    let targets = "targets = { median_ms = 100.0 }";
    let with = with_reference_context(
        "refctx-verdict-with",
        targets,
        "target_max_model_len = 262144",
    );
    let without = with_reference_context("refctx-verdict-without", targets, "");
    let baseline = |root: &Path| {
        serde_json::to_string(&baseline_for(root, "ttft-cold-gate").unwrap()).unwrap()
    };
    assert_eq!(baseline(&with), baseline(&without));
    for value in [100.0, 499.0, 501.0] {
        assert_eq!(
            verdict_at(&with, value),
            verdict_at(&without, value),
            "{value}"
        );
    }
    let entries = super::super::battle_load::entries;
    assert_eq!(
        format!("{:?}", entries(&with).unwrap()),
        format!("{:?}", entries(&without).unwrap()),
        "a battle never sees the reference context"
    );
}

/// 2026-10-16: The reader-facing account describes the targets, so an entry with none may not
/// declare it; beside targets it loads, apart from the engineering `note`.
#[test]
fn a_target_note_needs_targets() {
    let line = r#"target_note = "the account""#;
    let err = format!(
        "{:#}",
        load_all(&with_reference_context("tnote-untargeted", "", line)).unwrap_err()
    );
    assert!(
        err.contains("declares target_note with no targets for it to describe"),
        "{err}"
    );
    let targets = "targets = { median_ms = 100.0 }";
    let loaded = load_all(&with_reference_context("tnote-ok", targets, line)).unwrap();
    assert_eq!(loaded[0].1.target_note, "the account");
    assert!(loaded[0].1.note.is_empty(), "the note is its own field");
}

/// 2026-10-16: The reader-facing account is prose: the baseline every gate verdict reads, the
/// verdict on both sides of the bound, and the entries a battle judges are the same with and
/// without it.
#[test]
fn a_target_note_never_moves_a_verdict() {
    let targets = "targets = { median_ms = 100.0 }";
    let with = with_reference_context(
        "tnote-verdict-with",
        targets,
        r#"target_note = "the account""#,
    );
    let without = with_reference_context("tnote-verdict-without", targets, "");
    let baseline = |root: &Path| {
        serde_json::to_string(&baseline_for(root, "ttft-cold-gate").unwrap()).unwrap()
    };
    assert_eq!(baseline(&with), baseline(&without));
    for value in [100.0, 499.0, 501.0] {
        assert_eq!(
            verdict_at(&with, value),
            verdict_at(&without, value),
            "{value}"
        );
    }
    let entries = super::super::battle_load::entries;
    assert_eq!(
        format!("{:?}", entries(&with).unwrap()),
        format!("{:?}", entries(&without).unwrap()),
        "a battle never sees the target note"
    );
}
