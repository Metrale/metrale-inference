// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Exact recovery of a synthetic affine model (linear and rate), least squares over
//! extra points, and refusal of points that do not determine every signature.
//!
//! Owner: metrale-ml-utils.
//! Invariants: none beyond the types.

use super::*;

fn p(units: &[f64], value: f64) -> Point {
    Point {
        units: units.to_vec(),
        value,
    }
}

#[test]
fn two_signatures_need_three_points_and_are_recovered_exactly() {
    // 2026-10-03: TTFT = 50 + 30 * a + 70 * b; the full model has 14 a and 2 b.
    let f = |a: f64, b: f64| 50.0 + 30.0 * a + 70.0 * b;
    let pts = [
        p(&[1.0, 1.0], f(1.0, 1.0)),
        p(&[2.0, 1.0], f(2.0, 1.0)),
        p(&[1.0, 2.0], f(1.0, 2.0)),
    ];
    let e = extrapolate(&pts, &[14.0, 2.0], Scaling::Linear).unwrap();
    assert!((e.full - f(14.0, 2.0)).abs() < 1e-6);
    assert!((e.fixed - 50.0).abs() < 1e-9 && (e.per_unit[1] - 70.0).abs() < 1e-9);
    assert!(e.max_residual < 1e-12);
    assert!(extrapolate(&pts[..2], &[14.0, 2.0], Scaling::Linear).is_err());
}

#[test]
fn a_rate_is_fitted_as_its_reciprocal() {
    // 2026-10-03: 2 ms fixed + 1.5 ms per unit per token; tok/s is the reciprocal.
    let rate = |u: f64| 1.0 / (0.002 + 0.0015 * u);
    let e = extrapolate(
        &[p(&[1.0], rate(1.0)), p(&[2.0], rate(2.0))],
        &[10.0],
        Scaling::Rate,
    )
    .unwrap();
    assert!((e.full - rate(10.0)).abs() / rate(10.0) < 1e-9);
    assert!(extrapolate(&[p(&[1.0], 0.0), p(&[2.0], 1.0)], &[10.0], Scaling::Rate).is_err());
}

#[test]
fn collinear_points_are_refused_and_extra_points_are_least_squares() {
    let pts = [
        p(&[1.0, 1.0], 3.0),
        p(&[2.0, 2.0], 5.0),
        p(&[3.0, 3.0], 7.0),
    ];
    let err = extrapolate(&pts, &[4.0, 4.0], Scaling::Linear).unwrap_err();
    assert!(err.to_string().contains("do not determine"), "{err}");
    let noisy = [p(&[1.0], 10.0), p(&[2.0], 12.1), p(&[3.0], 13.9)];
    let e = extrapolate(&noisy, &[10.0], Scaling::Linear).unwrap();
    assert!(e.max_residual > 0.0 && e.max_residual < 0.02);
    assert!(Scaling::parse("log").is_err());
}

#[test]
fn resolved_specs_give_units_and_mismatched_sources_are_refused() {
    let (config, index) = crate::testkit::moe_fp8();
    let mk = |per: &str| {
        crate::plan::plan_mock(&crate::plan::MockInputs {
            source_id: "toy/model",
            revision: None,
            config_json: &config,
            hf_quant_config: None,
            index: &index,
            spec: &crate::testkit::spec(per, "mode = \"uniform\""),
            routing: None,
            calibration: None,
            stats: None,
        })
        .unwrap()
    };
    let one = units_of_resolved(&mk("1").resolved).unwrap();
    let two = units_of_resolved(&mk("2").resolved).unwrap();
    assert_eq!((one.kept.clone(), one.full.clone()), (vec![1.0], vec![2.0]));
    assert_eq!(two.kept, vec![2.0]);
    let (pts, full) = points_of(&[(one.clone(), 100.0), (two, 160.0)]).unwrap();
    let e = extrapolate(&pts, &full, Scaling::Linear).unwrap();
    assert!(
        (e.full - 160.0).abs() < 1e-9,
        "the toy's full model is two units"
    );
    let mut other = one.clone();
    other.source.1 = "x".into();
    assert!(points_of(&[(one, 1.0), (other, 2.0)]).is_err());
}
