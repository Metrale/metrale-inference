// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Tests for the draft-confidence outcome counts.
//!
//! Owner: telemetry.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn edges_ascend_and_end_at_zero() {
    assert!(EDGES.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(EDGES[EDGES.len() - 1], 0.0);
}

#[test]
fn a_log_probability_lands_in_the_first_bucket_whose_edge_covers_it() {
    assert_eq!(ConfidenceOutcomes::bucket(-9.0), Some(0));
    assert_eq!(ConfidenceOutcomes::bucket(-4.0), Some(0));
    assert_eq!(ConfidenceOutcomes::bucket(-3.9), Some(1));
    assert_eq!(ConfidenceOutcomes::bucket(-0.001), Some(EDGES.len() - 1));
    assert_eq!(ConfidenceOutcomes::bucket(0.0), Some(EDGES.len() - 1));
    assert_eq!(ConfidenceOutcomes::bucket(1e-6), Some(EDGES.len() - 1));
    assert_eq!(ConfidenceOutcomes::bucket(f32::NAN), None);
    assert_eq!(ConfidenceOutcomes::bucket(f32::NEG_INFINITY), None);
}

#[test]
fn outcomes_are_counted_per_bucket_and_verdict() {
    let c = ConfidenceOutcomes::new();
    assert!(c.is_empty());
    c.record(-0.3, true);
    c.record(-0.3, true);
    c.record(-0.3, false);
    c.record(-5.0, false);
    c.record(f32::NAN, true);
    let b = ConfidenceOutcomes::bucket(-0.3).unwrap();
    assert_eq!((c.count(b, true), c.count(b, false)), (2, 1));
    assert_eq!((c.count(0, true), c.count(0, false)), (0, 1));
    assert_eq!(c.overflow.get(), 1);
    assert!(!c.is_empty());
}
