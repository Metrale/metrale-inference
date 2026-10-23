// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Critical pairs on the toy circuit, resolved through the real fuser
//! (`fuse_seeded`, `match_instances`). The cost is a per-rule price (each group costs its rule's
//! price, 1 unless stated), so a test chooses which resolution is dearer.
//!
//! Owner: metrale-circuit.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use super::*;
use crate::fuser::{AvailableKernels, FuseError, fuse_seeded, match_instances};
use crate::ir::Circuit;
use crate::rules::{Mode, Rule};
use crate::test_toy::{circuit, fused, plan, policy, rules};

const ACT_DOWN: &str = r#"{ op = "silu_mul" }, { op = "linear", role = "down" }"#;
const UP_ACT: &str = r#"{ op = "linear", role = "gate_up" }, { op = "silu_mul" }"#;

fn price(prices: &BTreeMap<&str, f64>, p: &crate::fuser::FusionPlan) -> f64 {
    p.groups
        .iter()
        .map(|g| prices.get(g.rule.as_str()).copied().unwrap_or(1.0))
        .sum()
}

/// 2026-10-23: Every critical pair of `r` on `c` at decode n=1, priced by `prices`.
fn pairs_of(c: &Circuit, r: &[Rule], prices: &[(&str, f64)]) -> Vec<CriticalPair> {
    let prices: BTreeMap<&str, f64> = prices.iter().copied().collect();
    let avail = AvailableKernels::all_named_by(r);
    let today = plan(c, r, &policy(), 1);
    let current: Vec<Match> = today
        .groups
        .iter()
        .map(|g| Match {
            rule: g.rule.clone(),
            nodes: g.nodes.clone(),
        })
        .collect();
    let matches = match_instances(c, r, &avail, &policy(), (Mode::Decode, 1));
    critical_pairs::<_, ()>(
        &matches,
        (&current, price(&prices, &today)),
        |m| m.rule.clone(),
        |m| {
            Ok(fuse_seeded(
                c,
                r,
                &avail,
                &policy(),
                (Mode::Decode, 1),
                core::slice::from_ref(m),
            )
            .map(|p| Outcome {
                digest: p.digest.clone(),
                cost: price(&prices, &p),
                placeholders: 0,
                launches: p.launches(),
            })
            .map_err(|e| e.to_string()))
        },
    )
    .expect("pairs")
}

fn pair<'a>(ps: &'a [CriticalPair], a: &str, b: &str) -> &'a CriticalPair {
    ps.iter()
        .find(|p| p.a.rule == a && p.b.rule == b)
        .unwrap_or_else(|| panic!("no pair {a} x {b}: {ps:?}"))
}

// 2026-10-23: Two overlapping fusions with different kernels and different prices: priority
// alone picks the plan, and today's plan (the higher priority, a_act_down) is the cheaper side.
#[test]
fn overlapping_rules_of_different_cost_are_priority_decided() {
    let c = circuit(1);
    let r = rules(&(fused("a_act_down", ACT_DOWN, 100) + &fused("b_up_act", UP_ACT, 90)));
    let ps = pairs_of(&c, &r, &[("b_up_act", 2.0)]);
    let p = pair(&ps, "a_act_down", "b_up_act");
    assert_eq!(p.class, Joinability::PriorityDecided);
    assert_eq!(p.current, Pick::A);
    assert!(!p.picks_dearer());
    assert_ne!(
        p.a_first.as_ref().unwrap().digest,
        p.b_first.as_ref().unwrap().digest
    );
}

// 2026-10-23: The same pair with the priorities swapped: today's plan now takes the dearer side,
// which the report must flag.
#[test]
fn a_priority_that_takes_the_dearer_side_is_flagged() {
    let c = circuit(1);
    let r = rules(&(fused("a_act_down", ACT_DOWN, 90) + &fused("b_up_act", UP_ACT, 100)));
    let ps = pairs_of(&c, &r, &[("b_up_act", 2.0)]);
    let p = pair(&ps, "b_up_act", "a_act_down");
    assert_eq!(p.class, Joinability::PriorityDecided);
    assert_eq!(p.current, Pick::A, "b_up_act is tried first and taken");
    assert!(p.picks_dearer());
}

// 2026-10-23: The same overlap at equal prices: different plans, same cost.
#[test]
fn overlapping_rules_of_equal_cost_join_at_the_same_cost() {
    let c = circuit(1);
    let r = rules(&(fused("a_act_down", ACT_DOWN, 100) + &fused("b_up_act", UP_ACT, 90)));
    let p = pair(&pairs_of(&c, &r, &[]), "a_act_down", "b_up_act").clone();
    assert_eq!(p.class, Joinability::SameCost);
    assert!(!p.picks_dearer());
}

// 2026-10-23: A second rule launching the same kernel over the same chain under another id: the
// plan digest names kernels, not rule ids, so both orders give the same plan.
#[test]
fn a_duplicate_rule_under_another_id_is_joinable_identical() {
    let c = circuit(1);
    let dup = fused("a_act_down", ACT_DOWN, 90).replace("id = \"a_act_down\"", "id = \"a2\"");
    let r = rules(&(fused("a_act_down", ACT_DOWN, 100) + &dup));
    let ps = pairs_of(&c, &r, &[]);
    assert_eq!(pair(&ps, "a_act_down", "a2").class, Joinability::Identical);
}

// 2026-10-23: The fused chains overlap the single-op rules they cover, one site per layer; two
// layers share one representative per signature pair, which counts both.
#[test]
fn layers_with_the_same_signatures_share_one_representative() {
    let c = circuit(2);
    let r = rules(&fused("a_act_down", ACT_DOWN, 100));
    let ps = pairs_of(&c, &r, &[]);
    let p = pair(&ps, "a_act_down", "act");
    assert_eq!(p.occurrences, 2);
    assert_eq!(ps.iter().filter(|p| p.b.rule == "act").count(), 1);
}

#[test]
fn classify_covers_every_class() {
    let o = |d: &str, cost: f64, placeholders: usize| -> Result<Outcome, String> {
        Ok(Outcome {
            digest: d.into(),
            cost,
            placeholders,
            launches: 1,
        })
    };
    assert_eq!(
        classify(&o("x", 1.0, 0), &o("x", 2.0, 0)),
        Joinability::Identical
    );
    assert_eq!(
        classify(&o("x", 1.0, 0), &o("y", 1.0, 0)),
        Joinability::SameCost
    );
    // 2026-10-23: Summation-order noise is below the tolerance; a real difference is not.
    assert_eq!(
        classify(&o("x", 1e3, 0), &o("y", 1e3 + 1e-9, 0)),
        Joinability::SameCost
    );
    assert_eq!(
        classify(&o("x", 1e3, 0), &o("y", 1e3 + 1e-3, 0)),
        Joinability::PriorityDecided
    );
    assert_eq!(
        classify(&o("x", 1.0, 0), &o("y", 1.0, 1)),
        Joinability::Forced
    );
    assert_eq!(
        classify(&o("x", 1.0, 0), &Err("refused".into())),
        Joinability::Forced
    );
}

#[test]
fn overlapping_lists_each_sharing_pair_once() {
    let m = |nodes: &[usize]| Match {
        rule: "r".into(),
        nodes: nodes.to_vec(),
    };
    assert_eq!(
        overlapping(&[m(&[0, 1]), m(&[1, 2]), m(&[3]), m(&[0, 1, 2])]),
        vec![(0, 1), (0, 3), (1, 3)]
    );
    assert!(overlapping(&[m(&[0]), m(&[1])]).is_empty());
}

// 2026-10-23: Seeding today's own group reproduces today's plan; a seed whose rule does not apply,
// or over a claimed node, is refused rather than dropped.
#[test]
fn a_seed_is_formed_first_or_refused() {
    let c = circuit(1);
    let r = rules(&fused("a_act_down", ACT_DOWN, 100));
    let avail = AvailableKernels::all_named_by(&r);
    let today = plan(&c, &r, &policy(), 1);
    let g = today
        .groups
        .iter()
        .find(|g| g.rule == "a_act_down")
        .unwrap();
    let own = Match {
        rule: g.rule.clone(),
        nodes: g.nodes.clone(),
    };
    let run = (Mode::Decode, 1);
    let again = fuse_seeded(&c, &r, &avail, &policy(), run, core::slice::from_ref(&own)).unwrap();
    assert_eq!(again.digest, today.digest);
    let act = Match {
        rule: "act".into(),
        nodes: vec![c.node("l0.ffn.act").unwrap()],
    };
    let seeded = fuse_seeded(&c, &r, &avail, &policy(), run, core::slice::from_ref(&act)).unwrap();
    assert_ne!(seeded.digest, today.digest, "the seed displaced a_act_down");
    let unknown = Match {
        rule: "nope".into(),
        nodes: act.nodes.clone(),
    };
    assert!(matches!(
        fuse_seeded(&c, &r, &avail, &policy(), run, &[unknown]),
        Err(FuseError::Seed(_))
    ));
    assert!(matches!(
        fuse_seeded(&c, &r, &avail, &policy(), run, &[own, act]),
        Err(FuseError::Seed(_))
    ));
}
