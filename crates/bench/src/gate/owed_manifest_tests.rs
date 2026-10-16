// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: `owed_manifest`: a manifest is settled only when every pair it lists passes or
//! is no longer registered; any pair still stale, failing or unrecorded keeps it owed; and only
//! schema 1 without unknown keys parses.
//!
//! Owner: bench gate tests.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use super::super::check_models::{ModelStanding, SubjectState};
use super::super::model_scope::Subject;
use super::{Manifest, Pair, SCHEMA, parse, render, settled};

fn pair(gate: &str) -> Pair {
    Pair {
        gate: gate.into(),
        hw: "gb10".into(),
        checkpoint: "org/glm".into(),
    }
}

fn manifest(gates: &[&str]) -> Manifest {
    Manifest {
        schema: SCHEMA,
        merge_sha: "0123456789abcdef0123456789abcdef01234567".into(),
        model: "glm".into(),
        owed: gates.iter().map(|g| pair(g)).collect(),
    }
}

fn standing(model: &str, state: SubjectState) -> ModelStanding {
    ModelStanding {
        subject: Subject {
            hw: "gb10".into(),
            model: model.into(),
            checkpoint: format!("org/{model}"),
        },
        reached: true,
        state,
    }
}

fn standings(of: &[(&str, Vec<ModelStanding>)]) -> BTreeMap<String, Vec<ModelStanding>> {
    of.iter().map(|(g, s)| (g.to_string(), s.clone())).collect()
}

#[test]
fn a_manifest_is_settled_only_when_every_pair_passes() {
    let m = manifest(&["ttft-cold-glm-64k", "agentic-webserver-glm"]);
    let both_pass = standings(&[
        (
            "ttft-cold-glm-64k",
            vec![standing("glm", SubjectState::Pass)],
        ),
        (
            "agentic-webserver-glm",
            vec![standing("glm", SubjectState::Pass)],
        ),
    ]);
    assert!(settled(&m, &both_pass));
    // 2026-10-16: Control: each state other than Pass on one pair keeps the manifest owed.
    for open in [
        SubjectState::Stale("its newest record is invalidated".into()),
        SubjectState::Fail(vec!["over its limit".into()]),
        SubjectState::Unrecorded,
    ] {
        let one_open = standings(&[
            (
                "ttft-cold-glm-64k",
                vec![standing("glm", SubjectState::Pass)],
            ),
            ("agentic-webserver-glm", vec![standing("glm", open.clone())]),
        ]);
        assert!(!settled(&m, &one_open), "{open:?}");
    }
}

#[test]
fn another_models_pass_does_not_settle_the_pair() {
    let m = manifest(&["ttft-cold-gate"]);
    let st = standings(&[(
        "ttft-cold-gate",
        vec![
            standing("qwen", SubjectState::Pass),
            standing("glm", SubjectState::Stale("invalidated".into())),
        ],
    )]);
    assert!(!settled(&m, &st));
}

#[test]
fn a_pair_no_longer_registered_is_settled() {
    let m = manifest(&["ttft-cold-gate"]);
    // 2026-10-16: The gate is gone, or it no longer registers glm: nothing is left to certify.
    assert!(settled(&m, &standings(&[])));
    let only_qwen = standings(&[(
        "ttft-cold-gate",
        vec![standing("qwen", SubjectState::Stale("invalidated".into()))],
    )]);
    assert!(settled(&m, &only_qwen));
}

#[test]
fn a_rendered_manifest_parses_back_and_other_schemas_are_refused() {
    let m = manifest(&["ttft-cold-gate", "agentic-webserver"]);
    let text = render(&m).expect("renders");
    assert_eq!(parse(&text), Ok(m));
    let v2 = text.replace("schema = 1", "schema = 2");
    assert_ne!(v2, text, "the schema line was rewritten");
    assert!(parse(&v2).is_err());
    let extra = format!("{text}\nnote = \"x\"\n");
    assert!(parse(&extra).is_err(), "an unknown key is refused");
}
