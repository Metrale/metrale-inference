// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The allowlist gate and the report's determinism.
//!
//! Owner: metrale-circuit.
//! Invariants: none beyond the types.

use super::*;
use crate::fuser::Match;
use crate::rules::Mode;

fn outcome(d: &str, cost: f64) -> Result<Outcome, String> {
    Ok(Outcome {
        digest: d.into(),
        cost,
        placeholders: 0,
        launches: 1,
    })
}

fn row(a: &str, b: &str, class: Joinability, (ca, cb): (f64, f64), current: Pick) -> PairRow {
    let m = |rule: &str| Match {
        rule: rule.into(),
        nodes: vec![0],
    };
    PairRow {
        site_a: "blk [x]".into(),
        site_b: "blk [x]".into(),
        pair: CriticalPair {
            a: m(a),
            b: m(b),
            a_first: outcome("da", ca),
            b_first: outcome("db", cb),
            current,
            class,
            occurrences: 1,
            current_cost: ca,
        },
    }
}

fn plan(model: &str, rows: Vec<PairRow>) -> PlanPairs {
    PlanPairs {
        model: model.into(),
        device: "gb10".into(),
        run: Run {
            mode: Mode::Decode,
            rows: 1,
        },
        matches: 2,
        floor_us: 1.0,
        rows,
    }
}

fn allow(text: &str) -> Vec<Allowed> {
    parse_allowlist(text).expect("allowlist")
}

const ZY: &str = "[[pair]]\nrules = [\"z\", \"y\"]\nwhy = \"fused wins\"\n";

#[test]
fn a_priority_decided_pair_needs_an_entry_and_an_entry_clears_it() {
    let plans = [plan(
        "m",
        vec![row(
            "z",
            "y",
            Joinability::PriorityDecided,
            (1.0, 2.0),
            Pick::A,
        )],
    )];
    let bad = gate(&plans, &[], 0);
    assert_eq!(bad.len(), 1, "{bad:?}");
    assert!(bad[0].contains("(\"y\", \"z\")"), "{bad:?}");
    // 2026-10-23: The entry's ids are matched in either order.
    assert!(gate(&plans, &allow(ZY), 1).is_empty());
}

#[test]
fn joinable_pairs_and_forced_pairs_on_the_legal_side_are_not_gated() {
    let mut forced = row("f", "g", Joinability::Forced, (1.0, 1.0), Pick::A);
    forced.pair.b_first = Err("refused".into());
    let plans = [plan(
        "m",
        vec![
            row("a", "b", Joinability::Identical, (1.0, 1.0), Pick::A),
            row("c", "d", Joinability::SameCost, (1.0, 1.0), Pick::B),
            forced.clone(),
        ],
    )];
    assert!(gate(&plans, &[], 0).is_empty());
    // 2026-10-23: The same forced pair with today's plan on the refused side is gated.
    forced.pair.current = Pick::B;
    assert_eq!(gate(&[plan("m", vec![forced])], &[], 0).len(), 1);
}

#[test]
fn a_stale_entry_a_duplicate_and_a_list_past_its_ceiling_fail() {
    let plans = [plan(
        "m",
        vec![row(
            "z",
            "y",
            Joinability::PriorityDecided,
            (1.0, 2.0),
            Pick::A,
        )],
    )];
    let stale = format!("{ZY}[[pair]]\nrules = [\"p\", \"q\"]\nwhy = \"old\"\n");
    let bad = gate(&plans, &allow(&stale), 2);
    assert!(
        bad.len() == 1 && bad[0].starts_with("stale entry"),
        "{bad:?}"
    );
    let dup = format!("{ZY}{ZY}");
    assert!(
        gate(&plans, &allow(&dup), 2)
            .iter()
            .any(|p| p.starts_with("duplicate"))
    );
    assert!(
        gate(&plans, &allow(ZY), 0)
            .iter()
            .any(|p| p.contains("past its ceiling"))
    );
}

#[test]
fn an_entry_without_a_reason_or_with_three_rules_is_refused() {
    assert!(parse_allowlist("[[pair]]\nrules = [\"a\", \"b\"]\n").is_err());
    assert!(parse_allowlist("[[pair]]\nrules = [\"a\", \"b\"]\nwhy = \" \"\n").is_err());
    assert!(parse_allowlist("[[pair]]\nrules = [\"a\", \"b\", \"c\"]\nwhy = \"x\"\n").is_err());
    assert!(parse_allowlist("# empty\n").unwrap().is_empty());
}

#[test]
fn the_report_is_independent_of_plan_order_and_counts_the_dearer_side() {
    let p1 = plan(
        "a-model",
        vec![row(
            "z",
            "y",
            Joinability::PriorityDecided,
            (2.0, 1.0),
            Pick::A,
        )],
    );
    let p2 = plan(
        "b-model",
        vec![row("c", "d", Joinability::SameCost, (1.0, 1.0), Pick::A)],
    );
    let one = render(&[p1.clone(), p2.clone()], &[], "cmd");
    assert_eq!(one, render(&[p2, p1], &[], "cmd"));
    assert!(
        one.contains("the cheaper side: 0; the dearer side: 1"),
        "{one}"
    );
    assert!(one.contains("| `y` × `z` | 1 | 1 | **missing** |"), "{one}");
}
