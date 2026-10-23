// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Plan shares on the toy model: a legacy run executes nothing, `*` executes every
//! measured site, a named site only that site, and a statement about a run or site the step does
//! not plan is refused. Without a capture there is no share at all.
//!
//! Owner: metrale-circuit tests.
//! Invariants: fixtures come from the parent `headroom_tests` module; the expected numbers are the
//! fixture's (lm_head 10 us and the ffn norm 3 us exclusive of a 100 us step).

use super::*;

/// 2026-10-23: lm_head 10 us and the ffn norm 3 us exclusive per 100 us step, under `e` for decode
/// n=1.
fn stated_trace(e: RunExecution) -> Trace {
    let mut t = trace(vec![
        kernel("lm_head", None, 2, 20.0, 20.0),
        kernel("norm", None, 4, 8.0, 6.0),
    ]);
    t.plan.execution = BTreeMap::from([("decode n=1".to_string(), e)]);
    t
}

fn plan(sites: &[&str]) -> RunExecution {
    RunExecution {
        by: ExecutedBy::Plan,
        sites: sites.iter().map(|s| s.to_string()).collect(),
        why: "test: --execution plan".into(),
    }
}

fn share(t: &Trace) -> super::super::headroom::PlanShare {
    let h = run("fp4dev", Mode::Decode, 1, &inputs(Some(t), None)).unwrap();
    assert_eq!(h.shares.len(), 1, "one run, one share");
    h.shares[0].clone()
}

#[test]
fn a_legacy_run_is_describable_but_executes_nothing() {
    let s = share(&stated_trace(legacy()));
    assert_eq!(s.by, ExecutedBy::Legacy);
    assert!((s.describable_us - 13.0).abs() < 1e-9, "{}", s.describable_us);
    assert_eq!((s.executed_us, s.executed_pct), (0.0, 0.0));
    assert!((s.describable_pct - s.describable_us).abs() < 1e-9, "a 100 us step");
}

#[test]
fn star_executes_every_measured_site_and_a_named_site_only_itself() {
    let all = share(&stated_trace(plan(&["*"])));
    assert_eq!(all.executed_us, all.describable_us);
    let one = share(&stated_trace(plan(&["head [lm_head]"])));
    assert_eq!(one.describable_us, all.describable_us);
    assert!((one.executed_us - 10.0).abs() < 1e-9, "{}", one.executed_us);
    assert!(one.executed_us < one.describable_us, "the norm site stayed legacy");
}

#[test]
fn a_statement_about_what_the_step_does_not_plan_is_refused() {
    let ghost = stated_trace(plan(&["head [no_such_site]"]));
    match run("fp4dev", Mode::Decode, 1, &inputs(Some(&ghost), None)) {
        Err(HeadroomError::Trace(m)) => assert!(m.contains("does not have"), "{m}"),
        other => panic!("expected a refusal, got {other:?}"),
    }
    let mut extra = stated_trace(legacy());
    extra
        .plan
        .execution
        .insert("verify n=2".to_string(), legacy());
    match run("fp4dev", Mode::Decode, 1, &inputs(Some(&extra), None)) {
        Err(HeadroomError::Trace(m)) => assert!(m.contains("does not plan"), "{m}"),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn without_a_capture_there_is_no_share() {
    let h = run("fp4dev", Mode::Decode, 1, &inputs(None, None)).unwrap();
    assert!(h.shares.is_empty());
}
