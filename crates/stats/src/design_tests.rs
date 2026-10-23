// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Every load is flanked by idle windows; ABBA cancels a linear drift between arms
//! (each arm's mean position is the same); a random order is reproducible from its seed and
//! differs across seeds.

use alloc::vec::Vec;

use super::*;

fn load_positions(o: &Order, arm: u16) -> Vec<usize> {
    o.arms
        .iter()
        .enumerate()
        .filter(|(_, s)| **s == Step::Load(arm))
        .map(|(i, _)| i)
        .collect()
}

#[test]
fn every_load_is_flanked_and_abba_balances_drift() {
    let o = Order::abba(2, 4);
    assert_eq!(o.render(), "I L0 I L1 I L1 I L0 I L0 I L1 I L1 I L0 I");
    for (i, s) in o.arms.iter().enumerate() {
        if matches!(s, Step::Load(_)) {
            assert_eq!((o.arms[i - 1], o.arms[i + 1]), (Step::Idle, Step::Idle));
        }
    }
    let mean = |v: Vec<usize>| v.iter().sum::<usize>() as f64 / v.len() as f64;
    assert_eq!(mean(load_positions(&o, 0)), mean(load_positions(&o, 1)));
    assert_eq!(o.label(), "abba");
    assert_eq!(abba(1), [Arm::Idle, Arm::Load, Arm::Load, Arm::Idle]);
}

#[test]
fn a_random_order_is_reproducible_from_its_seed() {
    let a = Order::random(4, 6, 99);
    assert_eq!(a, Order::random(4, 6, 99));
    assert_ne!(a.arms, Order::random(4, 6, 100).arms);
    assert_eq!(a.label(), "random:seed=99");
    for arm in 0..4 {
        assert_eq!(load_positions(&a, arm).len(), 6);
    }
}
