// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The paired A/B spec: the grammar, the arms each form makes, the palindromic order
//! and the refusals. Owner: model-layers ops tests.

use super::*;

#[test]
fn aa_is_two_identical_a_arms() {
    let s = PairedAbSpec::parse("aa").unwrap();
    assert!(s.names.is_empty());
    assert_eq!(s.arms, vec![AbLevers::NONE, AbLevers::NONE]);
    assert_eq!(s.order(), vec![0, 1, 1, 0]);
    assert_eq!(s.lever_sets(), vec![AbLevers::NONE]);
}

/// 2026-10-23: `aa:graphs` keys its arms apart while no named position is on, so every model
/// lever reads off on both arms.
#[test]
fn aa_graphs_is_two_a_arms_with_distinct_keys() {
    let s = PairedAbSpec::parse("aa:graphs").unwrap();
    assert!(s.names.is_empty());
    assert_eq!(s.arms.len(), 2);
    assert_eq!(s.arms[0], AbLevers::NONE);
    assert_ne!(s.arms[1].bits(), AbLevers::NONE.bits());
    assert!((0..AbLevers::MAX - 1).all(|i| !s.arms[1].has(i)));
    assert_eq!(s.lever_sets().len(), 2);
}

#[test]
fn a_list_is_a_against_all_on_and_attr_adds_one_off_arms() {
    let s = PairedAbSpec::parse("x, y").unwrap();
    assert_eq!(s.names, vec!["x", "y"]);
    assert_eq!(s.arms, vec![AbLevers::NONE, AbLevers::first(2)]);
    let a = PairedAbSpec::parse("x,y,z:attr").unwrap();
    let all = AbLevers::first(3);
    assert_eq!(
        a.arms,
        vec![
            AbLevers::NONE,
            all,
            all.without(0),
            all.without(1),
            all.without(2)
        ]
    );
    assert!(!a.arms[2].has(0) && a.arms[2].has(1) && a.arms[2].has(2));
    assert_eq!(a.order(), vec![0, 1, 2, 3, 4, 4, 3, 2, 1, 0]);
    // 2026-10-23: Every arm appears equally often in the order, once going up and once down.
    for arm in 0..a.arms.len() {
        assert_eq!(a.order().iter().filter(|&&x| x == arm).count(), 2);
    }
}

#[test]
fn empty_duplicate_and_too_many_names_are_refused() {
    for bad in ["", "x,,y", "x,x", "x,y,x:attr"] {
        assert!(PairedAbSpec::parse(bad).is_err(), "`{bad}` must be refused");
    }
    let many: Vec<String> = (0..=AbLevers::MAX).map(|i| format!("l{i}")).collect();
    assert!(PairedAbSpec::parse(&many.join(",")).is_err());
    let max: Vec<String> = (0..AbLevers::MAX).map(|i| format!("l{i}")).collect();
    assert!(PairedAbSpec::parse(&max.join(",")).is_ok());
}

#[test]
fn an_unset_instrument_is_off_and_resolves_once() {
    let p = PairedAb::default();
    assert!(p.get(None).unwrap().is_none());
    // 2026-10-23: Resolved once: a later text does not turn it on.
    assert!(p.get(Some("aa")).unwrap().is_none());
}

#[test]
fn each_cycle_runs_the_order_or_its_mirror_and_both_occur_evenly() {
    let s = PairedAbSpec::parse("x,y:attr").unwrap();
    let n = s.arms.len();
    let (up, down) = (
        s.order(),
        s.order().iter().map(|&a| n - 1 - a).collect::<Vec<_>>(),
    );
    let mut mirrored = 0;
    for c in 0..4000u64 {
        let cyc: Vec<usize> = (0..2 * n as u64)
            .map(|i| s.arm_at(c * 2 * n as u64 + i))
            .collect();
        assert!(cyc == up || cyc == down, "cycle {c}: {cyc:?}");
        mirrored += usize::from(cyc == down && cyc != up);
    }
    assert!(
        (1800..2200).contains(&mirrored),
        "{mirrored} of 4000 cycles mirrored"
    );
    // 2026-10-23: Arm 0 lands on the second position about half the time (the measured bias).
    let a = PairedAbSpec::parse("aa").unwrap();
    let second_is_0 = (0..4000u64).filter(|c| a.arm_at(c * 4 + 1) == 0).count();
    assert!((1800..2200).contains(&second_is_0), "{second_is_0}");
}
