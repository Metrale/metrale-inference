// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the busy-union core: islands, gaps, attribution, summary.

use metprof::core::union::{Class, attribute, busy_summary, gaps, islands, union_ns};

const COMPUTE: Option<Class> = Some(Class::Compute);
const COMM: Option<Class> = Some(Class::Comm);

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-7
}

#[test]
fn overlap_touch_and_nesting_merge_disjoint_stays_apart() {
    let got = islands([(30, 40), (0, 10), (10, 15), (2, 5), (50, 60), (55, 58)]).unwrap();
    assert_eq!(got, vec![(0, 15), (30, 40), (50, 60)]);
}

#[test]
fn union_is_not_the_sum() {
    // Negative control for a summing implementation: nested work counts once.
    assert_eq!(union_ns([(0, 100), (10, 20), (30, 90)]).unwrap(), 100);
}

#[test]
fn reversed_interval_is_an_error() {
    assert!(islands([(10, 5)]).is_err());
}

#[test]
fn gap_threshold_is_inclusive() {
    assert_eq!(gaps([(0, 10), (20, 30), (39, 40)], 10).unwrap(), (1, 10));
}

#[test]
fn gap_is_measured_against_the_running_max_end() {
    // The long kernel covers the hole between the two short ones: no gap.
    // Comparing with the previous kernel's end would report a 10-ns gap.
    assert_eq!(gaps([(0, 100), (10, 20), (30, 40)], 5).unwrap(), (0, 0));
}

#[test]
fn non_positive_gap_threshold_is_refused() {
    assert!(gaps([(0, 1)], 0).is_err());
}

#[test]
fn overlap_is_split_equally_and_sums_to_the_union() {
    let got = attribute([(0, 100, "a"), (50, 150, "b"), (200, 210, "a")]).unwrap();
    assert!(close(got["a"], 50.0 + 25.0 + 10.0));
    assert!(close(got["b"], 25.0 + 50.0));
    let total: f64 = got.values().sum();
    assert!(close(
        total,
        union_ns([(0, 100), (50, 150), (200, 210)]).unwrap() as f64
    ));
}

#[test]
fn same_key_twice_keeps_its_share_per_interval() {
    let got = attribute([(0, 90, "a"), (0, 90, "a"), (0, 90, "b")]).unwrap();
    assert!(close(got["a"], 60.0));
    assert!(close(got["b"], 30.0));
}

#[test]
fn zero_length_interval_gets_nothing() {
    let got = attribute([(5, 5, "z"), (0, 10, "a")]).unwrap();
    assert_eq!(got.into_iter().collect::<Vec<_>>(), vec![("a", 10.0)]);
}

#[test]
fn exposed_comm_and_idle() {
    let s = busy_summary(
        &[(0, 100), (50, 150), (200, 210)],
        &[COMPUTE, COMM, COMM],
        20,
    )
    .unwrap();
    assert_eq!(
        (s.span_ns, s.sum_ns, s.u_all_ns, s.u_compute_ns, s.u_comm_ns),
        (210, 210, 160, 100, 110)
    );
    assert_eq!(s.exposed_comm_ns, 60);
    assert_eq!(s.idle_ns, 50);
    assert_eq!((s.gap_count, s.gap_ns), (1, 50));
}

#[test]
fn hidden_comm_is_not_exposed() {
    let s = busy_summary(&[(0, 100), (10, 90)], &[COMPUTE, COMM], 1).unwrap();
    assert_eq!(s.exposed_comm_ns, 0);
    assert_eq!(s.u_comm_ns, 80);
}

#[test]
fn unclassified_time_is_not_reported_as_exposed_comm() {
    let s = busy_summary(&[(0, 100), (100, 150)], &[COMPUTE, None], 1).unwrap();
    assert_eq!(s.exposed_comm_ns, 0);
    assert_eq!(s.u_unclassified_ns, 50);
    assert_eq!(s.u_all_ns, 150);
}

#[test]
fn bad_summary_inputs_are_refused() {
    // An unknown class name cannot be represented: `Class` is a closed enum.
    assert!(busy_summary(&[], &[], 1).is_err());
    assert!(busy_summary(&[(0, 1)], &[], 1).is_err());
    assert!(busy_summary(&[(0, 1)], &[COMPUTE], 0).is_err());
}
