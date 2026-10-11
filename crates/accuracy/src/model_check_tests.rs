// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The metrics against hand-computed values, the pin and shape refusals, the
//! composition formula, and attribution to the first stage that leaves its budget.

use super::*;

fn pins(c: &str) -> Pins {
    Pins {
        corpus_sha256: c.into(),
        reference_sha256: "r".into(),
    }
}

#[test]
fn metrics_match_hand_values() {
    let same = token(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
    assert_eq!((same.kl, same.top1, same.max_dlogit), (0.0, true, 0.0));
    // 2026-10-09: A uniform shift of the logits changes no probability.
    let shifted = token(&[11.0, 12.0, 13.0], &[1.0, 2.0, 3.0]);
    assert!(shifted.kl.abs() < 1e-12 && shifted.top1 && shifted.max_dlogit == 10.0);
    // 2026-10-09: Two outcomes, p = (0.5, 0.5), q = softmax(0, ln 3) = (0.25, 0.75):
    // KL = 0.5 ln 2 + 0.5 ln(2/3).
    let t = token(&[0.0, 3f64.ln()], &[0.0, 0.0]);
    assert!((t.kl - (0.5 * 2f64.ln() + 0.5 * (2.0f64 / 3.0).ln())).abs() < 1e-12);
    assert!(!token(&[0.0, 1.0], &[1.0, 0.0]).top1);
}

#[test]
fn comparisons_refuse_other_pins_shapes_and_nan() {
    let r = [0.0, 1.0, 1.0, 0.0];
    let (per, s) = compare(&r, &r, 2, &pins("a"), &pins("a")).unwrap();
    assert_eq!((per.len(), s.top1, s.max_kl), (2, 1.0, 0.0));
    assert!(matches!(
        compare(&r, &r, 2, &pins("a"), &pins("b")),
        Err(ModelCheckError::Pins { .. })
    ));
    assert!(matches!(
        compare(&r, &r[..3], 2, &pins("a"), &pins("a")),
        Err(ModelCheckError::Shape(_))
    ));
    assert!(matches!(
        compare(&[0.0, f64::NAN], &[0.0, 0.0], 2, &pins("a"), &pins("a")),
        Err(ModelCheckError::NonFinite(0))
    ));
}

fn stage(name: &str, eps: f64, factor: f64) -> Stage {
    Stage {
        name: name.into(),
        eps,
        amplification: Amplification {
            factor,
            measured_by: "test".into(),
        },
    }
}

#[test]
fn composition_and_attribution() {
    let chain = [
        stage("l0", 1e-3, 1.0),
        stage("l1", 1e-3, 2.0),
        stage("l2", 1e-3, 3.0),
    ];
    let b = compose(&chain);
    assert!(
        (b[0] - 1e-3).abs() < 1e-15 && (b[1] - 3e-3).abs() < 1e-15 && (b[2] - 10e-3).abs() < 1e-15
    );
    assert_eq!(attribute(&chain, &[]), Attribution::Unavailable);
    assert_eq!(
        attribute(&chain, &[5e-4, 2e-3, 9e-3]),
        Attribution::WithinBudgets
    );
    match attribute(&chain, &[5e-4, 4e-3, 9e-3]) {
        Attribution::Stage { name, .. } => assert_eq!(name, "l1"),
        other => panic!("{other:?}"),
    }
}
