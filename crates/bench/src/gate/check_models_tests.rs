// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `decide`: every registered model's failure fails the gate, a reached declared
//! subject must certify, some declared subject must have a covering record, one reached model
//! must certify, and the reached models left uncertified are owed. Each rule has a control that
//! flips only the input the rule reads.
//!
//! Owner: bench gate tests.
//! Invariants: none beyond the types.

use std::collections::{BTreeMap, BTreeSet};

use super::super::check::GateStatus;
use super::super::model_scope::Subject;
use super::{SubjectState, decide};

fn s(model: &str) -> Subject {
    Subject {
        hw: "gb10".into(),
        model: model.into(),
        checkpoint: format!("org/{model}"),
    }
}

fn qwen() -> Subject {
    s("qwen")
}
fn glm() -> Subject {
    s("glm")
}
fn gemma() -> Subject {
    s("gemma")
}

fn stale() -> SubjectState {
    SubjectState::Stale("no record of it is committed".into())
}

fn states(of: &[(Subject, SubjectState)]) -> BTreeMap<Subject, SubjectState> {
    of.iter().cloned().collect()
}

fn set(of: &[Subject]) -> BTreeSet<Subject> {
    of.iter().cloned().collect()
}

fn is_missing(g: &GateStatus) -> bool {
    matches!(g, GateStatus::Missing(_))
}

#[test]
fn an_unreached_non_default_model_that_fails_its_own_limits_fails_the_gate() {
    let fail = SubjectState::Fail(vec!["ttft_ms 700 > 650".into()]);
    let (g, owed) = decide(
        &states(&[(qwen(), SubjectState::Pass), (glm(), fail)]),
        &set(&[]),
        &set(&[qwen()]),
    );
    let GateStatus::Fail(p) = g else {
        panic!("{g:?}")
    };
    assert_eq!(p, vec!["org/glm on gb10: ttft_ms 700 > 650".to_string()]);
    assert!(owed.is_empty());
    // 2026-10-15: Control: the same model stale instead of failing does not fail the gate.
    let (g, _) = decide(
        &states(&[(qwen(), SubjectState::Pass), (glm(), stale())]),
        &set(&[]),
        &set(&[qwen()]),
    );
    assert!(matches!(g, GateStatus::Pass), "{g:?}");
}

#[test]
fn a_reached_declared_subject_must_certify_even_when_another_reached_model_did() {
    let st = states(&[
        (qwen(), stale()),
        (glm(), SubjectState::Pass),
        (gemma(), SubjectState::Pass),
    ]);
    // 2026-10-15: gemma is a second declared subject whose record still stands.
    let (g, owed) = decide(&st, &set(&[qwen(), glm()]), &set(&[qwen(), gemma()]));
    assert!(is_missing(&g), "{g:?}");
    assert_eq!(owed, vec![qwen()]);
    // 2026-10-15: Control: the same change not reaching qwen passes.
    let (g, _) = decide(&st, &set(&[glm()]), &set(&[qwen(), gemma()]));
    assert!(matches!(g, GateStatus::Pass), "{g:?}");
}

#[test]
fn some_declared_subject_must_have_a_covering_record() {
    let st = states(&[(qwen(), stale()), (glm(), SubjectState::Pass)]);
    let (g, owed) = decide(&st, &set(&[]), &set(&[qwen()]));
    assert!(is_missing(&g), "{g:?}");
    assert!(owed.is_empty(), "an unreached model is never owed");
    // 2026-10-15: Control: a gate with no declared subject at all is also missing.
    assert!(is_missing(&decide(&st, &set(&[]), &set(&[])).0));
    // 2026-10-15: Control: the declared subject passing passes.
    let st = states(&[(qwen(), SubjectState::Pass), (glm(), stale())]);
    assert!(matches!(
        decide(&st, &set(&[]), &set(&[qwen()])).0,
        GateStatus::Pass
    ));
}

#[test]
fn one_reached_model_certifying_passes_and_the_rest_are_owed() {
    // 2026-10-15: A kernel glm and gemma run (not qwen, the declared subject).
    let reached = set(&[glm(), gemma()]);
    let defaults = set(&[qwen()]);
    let st = states(&[
        (qwen(), SubjectState::Pass),
        (glm(), SubjectState::Pass),
        (gemma(), stale()),
    ]);
    let (g, owed) = decide(&st, &reached, &defaults);
    assert!(matches!(g, GateStatus::Pass), "{g:?}");
    assert_eq!(owed, vec![gemma()]);
    // 2026-10-15: Control: neither reached model certified is missing, and both are owed.
    let st = states(&[
        (qwen(), SubjectState::Pass),
        (glm(), stale()),
        (gemma(), stale()),
    ]);
    let (g, owed) = decide(&st, &reached, &defaults);
    assert!(is_missing(&g), "{g:?}");
    assert_eq!(owed, vec![gemma(), glm()]);
}

#[test]
fn a_change_reaching_nothing_owes_nothing() {
    let st = states(&[(qwen(), SubjectState::Pass), (glm(), stale())]);
    let (g, owed) = decide(&st, &set(&[]), &set(&[qwen()]));
    assert!(matches!(g, GateStatus::Pass), "{g:?}");
    assert!(owed.is_empty());
}
