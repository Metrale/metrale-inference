// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: NNLS recovers a non-negative truth exactly and clamps a negative one to 0; the
//! collinearity checks pass orthogonal designs and flag nearly collinear ones; the bootstrap
//! intervals cover the generating coefficients.

use alloc::vec;
use alloc::vec::Vec;

use super::*;

fn design(rows: usize, cols: &[&dyn Fn(usize) -> f64]) -> Matrix {
    let data = (0..rows)
        .flat_map(|r| cols.iter().map(move |c| c(r)))
        .collect();
    Matrix::new(rows, cols.len(), data).unwrap()
}

fn tri(r: usize, p: usize) -> f64 {
    (r % p) as f64 / p as f64
}

#[test]
fn recovers_non_negative_coefficients_exactly() {
    let a = design(60, &[&|_| 1.0, &|r| tri(r, 7), &|r| tri(r * 3 + 1, 11)]);
    let truth = [12.5, 30.0, 4.0];
    let b: Vec<f64> = (0..60)
        .map(|r| (0..3).map(|j| a.get(r, j) * truth[j]).sum())
        .collect();
    let x = nnls(&a, &b).unwrap();
    for (got, want) in x.iter().zip(truth) {
        assert!((got - want).abs() < 1e-9, "{x:?}");
    }
}

#[test]
fn a_negative_truth_is_clamped_to_zero() {
    let a = design(50, &[&|_| 1.0, &|r| tri(r, 5), &|r| tri(r * 7 + 2, 13)]);
    let b: Vec<f64> = (0..50)
        .map(|r| 10.0 + 5.0 * a.get(r, 1) - 3.0 * a.get(r, 2))
        .collect();
    let x = nnls(&a, &b).unwrap();
    assert_eq!(x[2], 0.0);
    assert!(x.iter().all(|&v| v >= 0.0));
}

#[test]
fn collinearity_is_flagged_and_orthogonality_passes() {
    let orth = design(
        4,
        &[
            &|r| [1.0, 1.0, 1.0, 1.0][r],
            &|r| [1.0, -1.0, 1.0, -1.0][r],
            &|r| [1.0, 1.0, -1.0, -1.0][r],
        ],
    );
    assert!((condition_number(&orth).unwrap() - 1.0).abs() < 1e-9);
    let col = design(
        40,
        &[&|_| 1.0, &|r| tri(r, 9), &|r| {
            tri(r, 9) * 2.0 + 1e-4 * tri(r, 4)
        }],
    );
    assert!(condition_number(&col).unwrap() > MAX_CONDITION);
    let v = vif(&col).unwrap();
    assert_eq!(v[0], 1.0);
    assert!(v[1] > MAX_VIF && v[2] > MAX_VIF, "{v:?}");
    let ok = design(40, &[&|_| 1.0, &|r| tri(r, 9), &|r| tri(r * 5 + 3, 7)]);
    assert!(vif(&ok).unwrap().iter().all(|&x| x < 2.0));
}

#[test]
fn bootstrap_intervals_cover_the_generating_coefficients() {
    let mut r = crate::rng::SplitMix64::new(81);
    let a = design(300, &[&|_| 1.0, &|r| tri(r, 7), &|r| tri(r * 3 + 1, 11)]);
    let truth = [12.0, 20.0, 8.0];
    let mut hits = [0usize; 3];
    let trials: usize = 60;
    for t in 0..trials {
        let b: Vec<f64> = (0..300)
            .map(|i| (0..3).map(|j| truth[j] * a.get(i, j)).sum::<f64>() + 0.5 * r.normal())
            .collect();
        let ci = bootstrap_coefficients(&a, &b, 200, 0.95, 5 + u64::try_from(t).unwrap()).unwrap();
        for j in 0..3 {
            hits[j] += usize::from(ci[j].contains(truth[j]));
            assert!(ci[j].width() < 1.0);
        }
    }
    for h in hits {
        assert!(h * 100 >= 85 * trials, "{hits:?} of {trials}");
    }
}

#[test]
fn shapes_that_cannot_be_solved_are_refused() {
    use metrale_error::ErrorCode;
    assert_eq!(
        Matrix::new(2, 2, vec![1.0; 3]).unwrap_err().code(),
        ErrorCode::StatsInvalid
    );
    let a = Matrix::new(1, 2, vec![1.0, 2.0]).unwrap();
    assert_eq!(
        nnls(&a, &[1.0]).unwrap_err().code(),
        ErrorCode::StatsInsufficient
    );
}
