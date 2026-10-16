// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: Tests for `--gates <battle>`: which ids are battles, which required gates a
//! battle-only selection runs, and when a member's cell counts as current.

use std::collections::BTreeMap;

use metrale_bench::gate::battle::{Better, Cell};
use metrale_bench::gate::check_models::{ModelStanding, SubjectState};
use metrale_bench::gate::model_scope::Subject;
use metrale_bench::gate::{self, GateStatus};

use super::*;

fn ids(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

fn cell(gate: &str, hw: &str, checkpoint: &str) -> Cell {
    Cell {
        gate: gate.into(),
        hw: hw.into(),
        model: "glm".into(),
        checkpoint: checkpoint.into(),
        metric: "median_ms".into(),
        better: Better::Lower,
        target: 1.0,
        bound: None,
        caveat: None,
    }
}

fn standing(hw: &str, checkpoint: &str, state: SubjectState) -> ModelStanding {
    ModelStanding {
        subject: Subject {
            hw: hw.into(),
            model: "glm".into(),
            checkpoint: checkpoint.into(),
        },
        reached: false,
        state,
    }
}

#[test]
fn battle_ids_leave_the_gate_list_and_gate_ids_stay() {
    let mut gates = ids(&["decode-floor", "ttft-battle", "ttft-cold-gate"]);
    let battles = split(&mut gates, &ids(&["ttft-battle", "decode-battle"]));
    assert_eq!(battles, ["ttft-battle"]);
    assert_eq!(gates, ["decode-floor", "ttft-cold-gate"]);
}

#[test]
fn only_battles_named_runs_no_required_gate_and_nothing_named_runs_them_all() {
    let statuses: BTreeMap<String, GateStatus> = gate::REQUIRED_GATES
        .iter()
        .map(|g| (g.to_string(), GateStatus::Missing("none".into())))
        .collect();
    assert!(
        remaining(&statuses, &[], &ids(&["ttft-battle"]))
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        remaining(&statuses, &[], &[]).unwrap().len(),
        gate::REQUIRED_GATES.len(),
        "an empty --gates is still every open required gate"
    );
    let one = gate::REQUIRED_GATES[0];
    assert_eq!(
        remaining(&statuses, &ids(&[one]), &ids(&["ttft-battle"])).unwrap(),
        [one]
    );
}

#[test]
fn a_cell_is_current_only_when_its_own_subject_passes() {
    let c = cell("ttft-cold-glm-64k", "gb10", "org/GLM");
    let on = |list: Vec<ModelStanding>| -> BTreeMap<String, Vec<ModelStanding>> {
        [("ttft-cold-glm-64k".to_string(), list)]
            .into_iter()
            .collect()
    };
    assert!(current(
        &on(vec![standing("gb10", "org/GLM", SubjectState::Pass)]),
        &c
    ));
    for not in [
        standing("gb10", "org/GLM", SubjectState::Stale("moved".into())),
        standing("gb10", "org/GLM", SubjectState::Unrecorded),
        standing("h100", "org/GLM", SubjectState::Pass),
        standing("gb10", "org/Other", SubjectState::Pass),
    ] {
        assert!(!current(&on(vec![not.clone()]), &c), "{not:?}");
    }
    assert!(
        !current(&BTreeMap::new(), &c),
        "a gate with no standings is open"
    );
}

#[test]
fn members_follow_the_named_gates_each_once() {
    let plan = Plan {
        battles: Vec::new(),
        hardware: "gb10".into(),
        members: vec!["ttft-cold-glm-64k", "decode-floor"],
    };
    assert_eq!(
        plan.with_members(vec!["decode-floor"]),
        ["decode-floor", "ttft-cold-glm-64k"]
    );
}
