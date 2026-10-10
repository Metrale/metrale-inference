// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The dump metrics against hand values: tie-aware agreement, the KL lower bound
//! with the floor for missing tokens, decode divergence margins, pins, and the refusal of an
//! empty leg.

use super::*;

fn top(p: &[(&str, f64)]) -> Option<BTreeMap<String, f64>> {
    Some(p.iter().map(|(k, v)| (k.to_string(), *v)).collect())
}

fn seq(tokens: &[&str], lp: &[f64], tops: Vec<Option<BTreeMap<String, f64>>>) -> Seq {
    Seq {
        tokens: tokens.iter().map(|s| s.to_string()).collect(),
        lp: lp.iter().map(|v| Some(*v)).collect(),
        top: tops,
    }
}

fn dump(tf: Vec<Seq>, dec: Vec<Seq>) -> Dump {
    Dump {
        corpus_sha256: "c".into(),
        tf,
        dec1: dec.clone(),
        dec4: dec,
    }
}

#[test]
fn position_metrics_match_hand_values() {
    let r = top(&[("a", (0.5f64).ln()), ("b", (0.5f64).ln())]).unwrap();
    let t = top(&[("b", (0.5f64).ln()), ("a", (0.5f64).ln())]).unwrap();
    let (agree, kl, d) = position(&r, &t, Some(-0.1), Some(-0.3));
    assert!(agree, "tied sets intersect whatever the listing order");
    assert!(kl.abs() < 1e-12);
    assert!((d - 0.2).abs() < 1e-12);
    // 2026-10-09: `c` is missing from the test's list: it takes the test's smallest logprob.
    let r2 = top(&[("a", (0.75f64).ln()), ("c", (0.25f64).ln())]).unwrap();
    let t2 = top(&[("a", (0.9f64).ln()), ("b", (0.1f64).ln())]).unwrap();
    let (agree2, kl2, _) = position(&r2, &t2, None, None);
    let want = 0.75 * ((0.75f64).ln() - (0.9f64).ln()) + 0.25 * ((0.25f64).ln() - (0.1f64).ln());
    assert!(agree2 && (kl2 - want).abs() < 1e-12);
}

#[test]
fn decode_legs_stop_at_the_first_divergence_and_report_its_margin() {
    let tops = || {
        vec![
            top(&[("x", -0.1), ("y", -2.4)]),
            top(&[("y", -0.2), ("z", -0.9)]),
            top(&[("q", -0.1)]),
        ]
    };
    let r = dump(
        vec![seq(&["x"], &[-0.1], vec![top(&[("x", -0.1)])])],
        vec![seq(&["x", "y", "q"], &[-0.1, -0.2, -0.1], tops())],
    );
    let t = dump(
        r.tf.clone(),
        vec![seq(&["x", "z", "q"], &[-0.1, -0.9, -0.1], tops())],
    );
    let m = leg_metrics(&r, &t, "dec1");
    assert_eq!(
        (m.positions, m.diverged, m.unmeasured_divergences),
        (1, 1, 0)
    );
    assert!((m.max_margin_at_divergence.unwrap() - 0.7).abs() < 1e-12);
    let l = Limits {
        tf_min_top1: 1.0,
        tf_max_kl: 0.0,
        tf_max_dlp_p99: 0.0,
        dec_max_kl: 0.0,
        dec_max_dlp_p99: 0.0,
        max_divergence_margin: 0.5,
        max_unmeasured_divergences: 0.0,
    };
    assert!(
        !judge("dec1", &m, &l),
        "a divergence at margin 0.7 exceeds 0.5"
    );
    assert!(judge(
        "dec1",
        &m,
        &Limits {
            max_divergence_margin: 0.75,
            ..l
        }
    ));
    assert_eq!(
        exact(&r, &t),
        vec![
            ("dec1".to_string(), 0, Some(1)),
            ("dec4".to_string(), 0, Some(1))
        ]
    );
    assert!(exact(&r, &r).is_empty());
}

#[test]
fn pins_are_enforced_and_an_empty_leg_fails() {
    let d = dump(vec![], vec![]);
    let bytes = serde_json::to_vec(
        &serde_json::json!({"corpus_sha256": "c", "tf": [], "dec1": [], "dec4": []}),
    )
    .unwrap();
    let sha: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert!(load(&bytes, &sha, &bytes).is_ok());
    assert!(matches!(
        load(&bytes, "00", &bytes),
        Err(DumpError::Reference { .. })
    ));
    let other = serde_json::to_vec(
        &serde_json::json!({"corpus_sha256": "other", "tf": [], "dec1": [], "dec4": []}),
    )
    .unwrap();
    assert!(matches!(
        load(&bytes, &sha, &other),
        Err(DumpError::Corpus { .. })
    ));
    let l = Limits {
        tf_min_top1: 0.0,
        tf_max_kl: 1.0,
        tf_max_dlp_p99: 1.0,
        dec_max_kl: 1.0,
        dec_max_dlp_p99: 1.0,
        max_divergence_margin: 1.0,
        max_unmeasured_divergences: 1.0,
    };
    assert!(
        !judge("tf", &leg_metrics(&d, &d, "tf"), &l),
        "nothing measured is never a pass"
    );
}
