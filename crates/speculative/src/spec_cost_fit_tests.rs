// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Tests for fitting and writing the acceptance calibration.
//!
//! Owner: speculative.
//! Invariants: none beyond the types.

use super::*;

fn drafter() -> DrafterKey {
    DrafterKey {
        weights_sha256: "ab12".into(),
        vocab: 248_320,
        quantization: "bf16".into(),
        context: true,
    }
}

fn steps(rows: &[((usize, usize), u64)]) -> BTreeMap<(usize, usize), u64> {
    rows.iter().copied().collect()
}

#[test]
fn well_populated_buckets_take_their_own_rate_and_the_file_reads_back() {
    let s = steps(&[((2, 0), 100), ((2, 1), 100), ((2, 2), 200)]);
    let c = AcceptanceCalibration::fit(drafter(), &[-1.0, 0.0], &[30, 180], &[70, 20], &s).unwrap();
    assert_eq!(c.p_given_lp(-2.0), 0.3);
    assert_eq!(c.p_given_lp(-0.01), 0.9);
    // 2026-10-04: prior(1) = 300/400; prior(2) = 200/300.
    assert_eq!(c.prior(1), 0.75);
    assert!((c.prior(2) - 2.0 / 3.0).abs() < 1e-12);
    let text = c.render().unwrap();
    assert_eq!(AcceptanceCalibration::parse(&text).unwrap(), c);
}

#[test]
fn sparse_buckets_pool_towards_higher_confidence_and_a_low_remainder_joins_the_last_pool() {
    let s = steps(&[((1, 1), 100)]);
    // 2026-10-04: From the top: bucket 3 (100) closes a pool; buckets 2+1 (40+60) close the
    // next; bucket 0 (10) is a remainder and joins the 2+1 pool: (20+30+5)/(40+60+10).
    let c = AcceptanceCalibration::fit(
        drafter(),
        &[-2.0, -1.0, -0.5, 0.0],
        &[5, 30, 20, 90],
        &[5, 30, 20, 10],
        &s,
    )
    .unwrap();
    assert_eq!(c.p_given_lp(-0.1), 0.9);
    let pooled = 55.0 / 110.0;
    for lp in [-3.0, -1.5, -0.7] {
        assert_eq!(c.p_given_lp(lp), pooled, "lp {lp}");
    }
}

#[test]
fn priors_stop_at_the_first_thin_position() {
    // 2026-10-04: Position 2 is reached by 50 steps only, under MIN_OUTCOMES.
    let s = steps(&[((2, 0), 150), ((2, 1), 30), ((2, 2), 20)]);
    let c = AcceptanceCalibration::fit(drafter(), &[0.0], &[100], &[100], &s).unwrap();
    assert_eq!(c.prior(1), 0.25);
    assert_eq!(
        c.prior(2),
        0.25,
        "past the fitted positions the last prior holds"
    );
}

#[test]
fn too_few_observations_are_refused() {
    let enough = steps(&[((1, 1), 200)]);
    let err = AcceptanceCalibration::fit(drafter(), &[0.0], &[50], &[49], &enough).unwrap_err();
    assert!(err.contains("99 reached drafts"), "{err}");
    let thin = steps(&[((1, 1), 99)]);
    let err = AcceptanceCalibration::fit(drafter(), &[0.0], &[100], &[0], &thin).unwrap_err();
    assert!(err.contains("reached the first draft"), "{err}");
    let err = AcceptanceCalibration::fit(drafter(), &[0.0, 1.0], &[1], &[1], &enough).unwrap_err();
    assert!(err.contains("differ in length"), "{err}");
}
