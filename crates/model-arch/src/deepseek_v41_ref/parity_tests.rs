// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: The parity scorer against hand-computed values.
//!
//! Owner: model-arch, DeepSeek-V4.1 reference.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn identical_rows_score_zero_and_agree() {
    let r = [0.5f32, -1.0, 3.0, 2.0, 2.0, 0.0];
    let p = score(&r, &r, 3).unwrap();
    assert_eq!(p.positions.len(), 2);
    assert_eq!(p.max_kl(), 0.0);
    assert_eq!(p.top1_matches(), 2);
    assert_eq!(p.positions[0].reference_top1, 2);
    assert_eq!(
        p.positions[1].reference_top1, 0,
        "the tie goes to the lowest id"
    );
}

#[test]
fn kl_matches_the_closed_form_and_is_shift_invariant() {
    // 2026-10-19: reference p = (0.5, 0.5); candidate q = (e/(1+e), 1/(1+e)).
    let p = score(&[7.0, 7.0], &[1.0, 0.0], 2).unwrap();
    let e = 1f64.exp();
    let (q0, q1) = (e / (1.0 + e), 1.0 / (1.0 + e));
    let want = 0.5 * (0.5 / q0).ln() + 0.5 * (0.5 / q1).ln();
    assert!(
        (p.positions[0].kl - want).abs() < 1e-12,
        "{} vs {want}",
        p.max_kl()
    );
    let shifted = score(&[7.0, 7.0], &[101.0, 100.0], 2).unwrap();
    assert!((shifted.positions[0].kl - want).abs() < 1e-12);
    assert_eq!(p.top1_matches(), 1);
}

#[test]
fn kl_is_taken_from_the_reference_not_the_candidate() {
    let a = [2.0f32, 0.0, -3.0];
    let b = [0.0f32, 1.0, 0.0];
    let ab = score(&a, &b, 3).unwrap().max_kl();
    let ba = score(&b, &a, 3).unwrap().max_kl();
    assert!((ab - ba).abs() > 0.1, "{ab} vs {ba}: KL is not symmetric");
}

#[test]
fn a_disagreeing_arg_max_is_counted_and_the_summary_averages() {
    let r = [1.0f32, 0.0, 0.0, 1.0];
    let c = [1.0f32, 0.0, 1.0, 0.0];
    let p = score(&r, &c, 2).unwrap();
    assert_eq!(p.top1_matches(), 1);
    assert_eq!(p.positions[1].reference_top1, 1);
    assert_eq!(p.positions[1].candidate_top1, 0);
    assert!((p.mean_kl() - p.positions[1].kl / 2.0).abs() < 1e-15);
    assert_eq!(p.max_kl(), p.positions[1].kl);
}

#[test]
fn malformed_inputs_are_refused() {
    let bad = |r: &[f32], c: &[f32], v| {
        let e = score(r, c, v).expect_err("must refuse");
        assert_eq!(e.code(), ErrorCode::InvalidArgument);
        e.to_string()
    };
    assert!(bad(&[0.0; 4], &[0.0; 2], 2).contains("do not form rows"));
    assert!(bad(&[0.0; 3], &[0.0; 3], 2).contains("do not form rows"));
    assert!(bad(&[0.0; 2], &[0.0; 2], 0).contains("do not form rows"));
    let msg = bad(&[0.0, 0.0, 0.0, f32::NAN], &[0.0; 4], 2);
    assert!(msg.contains("position 1"), "{msg}");
    let msg = bad(&[0.0; 2], &[f32::INFINITY, 0.0], 2);
    assert!(msg.contains("position 0"), "{msg}");
}
