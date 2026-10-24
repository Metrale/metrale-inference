// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: `ScaleDictTwins`: readiness over the listed entries the target has, and the entry
//! `pick` launches for a plain or a packed table.

use super::*;

fn h(x: u64) -> KernelHandle {
    KernelHandle(x)
}

/// 2026-10-21: Plain entry `i` has handle `100 + i` and its twin `200 + i`.
fn full() -> [(KernelHandle, KernelHandle); N] {
    std::array::from_fn(|i| (h(100 + i as u64), h(200 + i as u64)))
}

#[test]
fn ready_needs_a_twin_for_every_entry_the_target_has() {
    assert!(ScaleDictTwins::of(full()).ready());
    // 2026-10-21: An entry the target lacks needs no twin (no site reaches it).
    let mut absent = full();
    absent[3] = (h(0), h(0));
    assert!(ScaleDictTwins::of(absent).ready());
    // 2026-10-21: A present entry without its twin is not ready.
    let mut missing = full();
    missing[N - 1].1 = h(0);
    assert!(!ScaleDictTwins::of(missing).ready());
    // 2026-10-21: A target with none of the entries (no W4A4 kernels) is not ready either.
    assert!(!ScaleDictTwins::of([(h(0), h(0)); N]).ready());
}

#[test]
fn pick_launches_the_twin_exactly_over_a_base_table() {
    let t = ScaleDictTwins::of(full());
    let bases = DevicePtr(0xBA5E);
    for i in [0usize, DECODE.len(), N - 1] {
        let (k, b) = t.pick(h(100 + i as u64), DevicePtr::NULL).unwrap();
        assert_eq!((k.0, b), (100 + i as u64, None), "plain table, entry {i}");
        let (k, b) = t.pick(h(100 + i as u64), bases).unwrap();
        assert_eq!(
            (k.0, b),
            (200 + i as u64, Some(bases)),
            "packed table, entry {i}"
        );
    }
}

#[test]
fn pick_refuses_a_packed_table_on_an_unlisted_entry_or_a_missing_twin() {
    let mut pairs = full();
    pairs[2].1 = h(0);
    let t = ScaleDictTwins::of(pairs);
    let bases = DevicePtr(0xBA5E);
    assert!(t.pick(h(999), bases).is_err(), "unlisted entry");
    assert!(
        t.pick(h(102), bases).is_err(),
        "listed entry without a twin"
    );
    assert!(t.pick(h(0), bases).is_err(), "an absent entry");
    assert_eq!(
        t.pick(h(999), DevicePtr::NULL).unwrap().0.0,
        999,
        "plain tables run anything"
    );
}
