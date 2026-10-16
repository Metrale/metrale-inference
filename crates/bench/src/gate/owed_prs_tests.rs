// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `owed_prs`: a merge certified on one of two reached models opens exactly one PR,
//! for the other; a re-run opens nothing; nothing owed opens nothing; a branch left without its
//! PR gets the PR alone.
//!
//! Owner: bench gate tests.
//! Invariants: none beyond the types.

use std::collections::{BTreeMap, BTreeSet};

use super::super::check::GateStatus;
use super::super::check_models::{Owed, SubjectState, decide};
use super::super::model_scope::Subject;
use super::owed_prs;

const SHA: &str = "0123456789abcdef0123456789abcdef01234567";

fn s(model: &str) -> Subject {
    Subject {
        hw: "gb10".into(),
        model: model.into(),
        checkpoint: format!("org/{model}"),
    }
}

/// 2026-10-15: What a merge reaching glm and qwen (the declared subject, certified) owes.
fn owed_after_one_of_two_certified() -> Vec<Owed> {
    let states = BTreeMap::from([
        (s("qwen"), SubjectState::Pass),
        (s("glm"), SubjectState::Stale("invalidated".into())),
    ]);
    let both = BTreeSet::from([s("qwen"), s("glm")]);
    let (status, owed) = decide(&states, &both, &BTreeSet::from([s("qwen")]));
    assert!(matches!(status, GateStatus::Pass), "{status:?}");
    owed.into_iter()
        .map(|subject| Owed {
            gate: "ttft-cold-gate".into(),
            subject,
        })
        .collect()
}

fn none() -> BTreeSet<String> {
    BTreeSet::new()
}

#[test]
fn one_of_two_certified_opens_exactly_one_pr_for_the_other() {
    let prs = owed_prs(
        &owed_after_one_of_two_certified(),
        SHA,
        &none(),
        &none(),
        &BTreeMap::new(),
    )
    .expect("renders");
    assert_eq!(prs.len(), 1, "{prs:?}");
    let pr = &prs[0];
    assert_eq!(pr.model, "glm");
    assert_eq!(pr.branch, "cert/glm/0123456");
    assert!(pr.create_branch);
    assert_eq!(pr.manifest_path, format!(".benchmarks/owed/glm-{SHA}.toml"));
    assert!(
        pr.body.contains(&pr.manifest_path),
        "the body names the manifest to delete"
    );
    let m: toml::Value = toml::from_str(&pr.manifest).expect("the manifest is TOML");
    assert_eq!(m["merge_sha"].as_str(), Some(SHA));
    let owed = m["owed"].as_array().expect("owed list");
    assert_eq!(owed.len(), 1);
    assert_eq!(owed[0]["gate"].as_str(), Some("ttft-cold-gate"));
    assert_eq!(owed[0]["checkpoint"].as_str(), Some("org/glm"));
}

#[test]
fn a_rerun_opens_nothing_and_a_branch_without_its_pr_gets_the_pr_alone() {
    let owed = owed_after_one_of_two_certified();
    let branch = BTreeSet::from(["cert/glm/0123456".to_string()]);
    assert!(
        owed_prs(&owed, SHA, &branch, &branch, &BTreeMap::new())
            .expect("renders")
            .is_empty()
    );
    let prs = owed_prs(&owed, SHA, &branch, &none(), &BTreeMap::new()).expect("renders");
    assert_eq!(prs.len(), 1);
    assert!(!prs[0].create_branch, "the branch exists");
    // 2026-10-15: Control: another merge's branch for the same model does not suppress this one.
    let other = BTreeSet::from(["cert/glm/fedcba9".to_string()]);
    assert_eq!(
        owed_prs(&owed, SHA, &other, &other, &BTreeMap::new())
            .expect("renders")
            .len(),
        1
    );
}

#[test]
fn nothing_owed_opens_nothing() {
    assert!(
        owed_prs(&[], SHA, &none(), &none(), &BTreeMap::new())
            .expect("renders")
            .is_empty()
    );
}

#[test]
fn every_gate_a_model_owes_rides_one_pr() {
    let owed: Vec<Owed> = ["ttft-cold-gate", "agentic-webserver", "ttft-cold-gate"]
        .into_iter()
        .map(|gate| Owed {
            gate: gate.into(),
            subject: s("glm"),
        })
        .chain([Owed {
            gate: "agentic-webserver".into(),
            subject: s("gemma"),
        }])
        .collect();
    let prs = owed_prs(&owed, SHA, &none(), &none(), &BTreeMap::new()).expect("renders");
    assert_eq!(
        prs.iter().map(|p| p.model.as_str()).collect::<Vec<_>>(),
        ["gemma", "glm"]
    );
    let m: toml::Value = toml::from_str(&prs[1].manifest).expect("TOML");
    assert_eq!(m["owed"].as_array().map(Vec::len), Some(2), "deduplicated");
}

/// 2026-10-16: A model's PR names the battles it owes and the command that scores them; a model
/// owing no battle gets no such line (control).
#[test]
fn a_pr_names_the_battles_its_model_owes() {
    let owed: Vec<Owed> = ["glm", "gemma"]
        .into_iter()
        .map(|m| Owed {
            gate: "ttft-cold-gate".into(),
            subject: s(m),
        })
        .collect();
    let battles = BTreeMap::from([(
        "glm".to_string(),
        BTreeSet::from(["ttft-battle".to_string()]),
    )]);
    let prs = owed_prs(&owed, SHA, &none(), &none(), &battles).expect("renders");
    let body = |m: &str| &prs.iter().find(|p| p.model == m).expect("a PR").body;
    assert!(
        body("glm").contains("`met benchmark battle ttft-battle --model glm`"),
        "{}",
        body("glm")
    );
    assert!(!body("gemma").contains("battle"), "{}", body("gemma"));
}
