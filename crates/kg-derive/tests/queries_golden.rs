// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Golden answers of the campaign-record queries (queries part §1: Q1, Q2, Q7, Q8)
//! on the fixture tree; each test also has its negative (the answer moves when the fact does).

mod common;

use metrale_kg_derive::queries::{blocked_by_gate, claims_without_gate, impact};
use metrale_kg_derive::{KgId, Rel, derive};

fn id(s: &str) -> KgId {
    KgId::parse(s).unwrap()
}

#[test]
fn q1_blocked_rows_name_their_blocker_and_an_unnamed_blocker_is_unresolved() {
    let b = derive(&common::config(), &common::sources(), None).unwrap();
    let rows = blocked_by_gate(&b);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].0, id("ledger:p0-x"));
    assert_eq!(rows[0].2, [id("decision:2026-10-22/p0-split")]);

    let mut src = common::sources();
    let l = src.files.get_mut("c/LEDGER.tsv").unwrap();
    *l = l.replace("blocked on decision:2026-10-22/p0-split, ", "");
    let b = derive(&common::config(), &src, None).unwrap();
    assert_eq!(
        blocked_by_gate(&b)[0].2,
        [id("unresolved:gate:blocker-of-p0-x")]
    );
}

#[test]
fn q2_every_claim_bearing_verdict_without_a_gate_is_listed() {
    let b = derive(&common::config(), &common::sources(), None).unwrap();
    let got: Vec<(String, String, bool)> = claims_without_gate(&b)
        .into_iter()
        .map(|(i, v, s)| (i.to_string(), v, s))
        .collect();
    assert_eq!(
        got,
        [
            ("ledger:l2-pf".into(), "NULL".into(), true),
            ("ledger:seg".into(), "LOSS".into(), true)
        ]
    );
}

#[test]
fn q7_principles_named_in_a_row_are_touched_and_others_are_not() {
    let b = derive(&common::config(), &common::sources(), None).unwrap();
    let touched: Vec<String> = b
        .out(&id("ledger:p0-x"), Rel::Touches)
        .map(|e| e.to.to_string())
        .collect();
    assert_eq!(touched, ["principle:ROTI"]);
    assert_eq!(b.out(&id("ledger:seg"), Rel::Touches).count(), 0);
}

#[test]
fn q8_dependency_chain_and_downstream_closure() {
    let b = derive(&common::config(), &common::sources(), None).unwrap();
    let (up, cut) = impact(&b, &id("ledger:seg"), false, &[Rel::DependsOn], 100).unwrap();
    let up: Vec<(String, u32)> = up
        .into_iter()
        .map(|r| (r.id.to_string(), r.depth))
        .collect();
    assert_eq!(up, [("ledger:l2-pf".into(), 1), ("ledger:p0-x".into(), 2)]);
    assert!(!cut);
    let (down, _) = impact(&b, &id("ledger:p0-x"), true, &[Rel::DependsOn], 100).unwrap();
    let down: Vec<String> = down.into_iter().map(|r| r.id.to_string()).collect();
    assert_eq!(down, ["ledger:l2-pf", "ledger:seg"]);
    let (one, cut) = impact(&b, &id("ledger:p0-x"), true, &[Rel::DependsOn], 1).unwrap();
    assert_eq!(one.len(), 1);
    assert!(cut, "depth 1 must report the truncation");
    assert_eq!(
        impact(&b, &id("ledger:nope"), true, &[Rel::DependsOn], 1)
            .unwrap_err()
            .code(),
        metrale_error::ErrorCode::KgUnknownId
    );
}

#[test]
fn decisions_levers_profiles_and_commits_come_from_their_text() {
    let b = derive(&common::config(), &common::sources(), None).unwrap();
    for want in [
        "decision:2026-10-22/p0-first-glm-decode",
        "decision:2026-10-22/kernel-generation-cuda-templates",
        "decision:2026-10-23/allowlist-growth-approved",
        "decision:2026-10-23/a-new-server",
        "decision:2026-10-23/the-embedder-is-openrouter",
        "design:DESIGN-x.md#6",
        "lever:METRALE_GLM_AR_L2_PREFETCH",
        "profile:glm53-tp3-512k-promo1",
        "commit:ec5326efb",
        "idea:gtm-3-critical-pair-report",
        "paper:arxiv:2109.06049",
        "idea:mathbomb-process-1",
    ] {
        assert!(b.node(&id(want)).is_some(), "{want} missing");
    }
    let open = b
        .node(&id("decision:2026-10-23/the-embedder-is-openrouter"))
        .unwrap();
    assert!(
        open.p["text"]
            .as_str()
            .unwrap()
            .contains("model from config")
    );
    assert!(
        b.node(&id("design:DESIGN-x.md#not-a-heading")).is_none(),
        "fenced lines are not headings"
    );
}

#[test]
fn calibration_lists_each_predicted_saving_with_its_settled_outcome() {
    use metrale_kg_derive::queries::calibration;
    use metrale_kg_derive::status::{Freshness, status};
    let got = |src: &metrale_kg_derive::SourceSet| -> Vec<(String, String, f64)> {
        let b = derive(&common::config(), src, None).unwrap();
        calibration(&b, "glm/mk")
            .into_iter()
            .map(|p| (p.id.to_string(), p.verdict, p.predicted_us))
            .collect()
    };
    // p0-x predicts 0 us: not a saving, so not listed.
    assert_eq!(
        got(&common::sources()),
        [
            ("ledger:l2-pf".into(), "NULL".into(), 315.0),
            ("ledger:seg".into(), "LOSS".into(), 100.0)
        ]
    );
    let fresh = Freshness {
        // 2026-10-23: test value: the clock and freshness are not under test.
        stale_reason: None,
        derived_at_ms: 0,
        source_commit: None,
        newest_change: None,
    };
    let sec = |src: &metrale_kg_derive::SourceSet| {
        let b = derive(&common::config(), src, None).unwrap();
        let names = ["calibration".to_string()];
        status(&b, &common::config(), "glm/mk", &names, &fresh, None).unwrap()["calibration"]
            .clone()
    };
    let v = sec(&common::sources());
    assert_eq!((v["hit"].as_u64(), v["miss"].as_u64()), (Some(0), Some(2)), "{v}");
    assert_eq!(v["hit_rate"], 0.0);
    assert_eq!(v["rows"][0]["measured"]["1k"][0], -0.53, "{v}");

    // Negative: the answer moves with the fact. seg turns into a WIN, l2-pf predicts nothing.
    let mut src = common::sources();
    let l = src.files.get_mut("c/LEDGER.tsv").unwrap();
    *l = l
        .replace("\tLOSS\tmkp3", "\tWIN\tmkp3")
        .replace("<=315 us/step", "0 us/step");
    assert_eq!(got(&src), [("ledger:seg".into(), "WIN".into(), 100.0)]);
    let v = sec(&src);
    assert_eq!((v["hit"].as_u64(), v["miss"].as_u64()), (Some(1), Some(0)), "{v}");
    assert_eq!(v["hit_rate"], 1.0);

    // The predicted saving is the number that carries `us`, not the first digit in the field.
    let mut src = common::sources();
    let l = src.files.get_mut("c/LEDGER.tsv").unwrap();
    *l = l.replace("<=315 us/step", "K=2 warm, upside 267 us/step");
    assert_eq!(got(&src)[0], ("ledger:l2-pf".into(), "NULL".into(), 267.0));
}
