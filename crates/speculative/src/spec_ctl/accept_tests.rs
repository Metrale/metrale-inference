// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The acceptance model: what a verify observes, decay, the prior blend, and the
//! rate form's equality with a seeded EWMA (the MTP rung's estimator).
//!
//! Owner: speculative.
//! Invariants: none beyond the types.

use super::*;

const STEP: AcceptParams = AcceptParams {
    decay: 0.95,
    prior_weight: 4.0,
    cold_weight: 0.0,
    seed: SeedWeight::Unit,
};

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

/// 2026-10-10: A partial accept observes accepts before the first reject and the reject; a
/// full accept observes no reject and nothing past the verified drafts.
#[test]
fn a_verify_observes_accepts_then_the_reject_and_nothing_past_it() {
    let mut r = AcceptCounts::default();
    r.observe_step(&STEP, 4, 2);
    let c = r.rates(&STEP, None, &ColdPrior::None).unwrap();
    assert_eq!(
        &c[..4],
        &[1.0, 1.0, 0.0, 0.0],
        "position 4 takes position 3's rate"
    );
    let mut full = AcceptCounts::default();
    full.observe_step(&STEP, 3, 3);
    assert_eq!(
        &full.rates(&STEP, None, &ColdPrior::None).unwrap()[..4],
        &[1.0; 4]
    );
    assert!(!full.observed(4), "no observation past the verified drafts");
    assert!(
        AcceptCounts::default()
            .rates(&STEP, None, &ColdPrior::None)
            .is_none()
    );
}

/// 2026-10-10: Counts decay per observation, so recent steps dominate; the prior weighs
/// `prior_weight` observations and stands alone where the sequence has none.
#[test]
fn rates_decay_and_blend_with_the_prior() {
    let mut r = AcceptCounts::default();
    for _ in 0..200 {
        r.observe_step(&STEP, 1, 0);
    }
    for _ in 0..20 {
        r.observe_step(&STEP, 1, 1);
    }
    let c0 = r.rates(&STEP, None, &ColdPrior::None).unwrap()[0];
    assert!(
        c0 > 0.6 && c0 < 0.7,
        "20 recent accepts over a decayed past: {c0}"
    );

    let mut prior = AcceptCounts::default();
    prior.observe_step(&STEP, 1, 0);
    prior.observe_step(&STEP, 1, 1);
    let prior_rate = 1.0 / 1.95;
    let mut seq = AcceptCounts::default();
    assert!(approx(
        seq.rates(&STEP, Some(&prior), &ColdPrior::None).unwrap()[0],
        prior_rate
    ));
    seq.observe_step(&STEP, 1, 1);
    let blended = seq.rates(&STEP, Some(&prior), &ColdPrior::None).unwrap()[0];
    assert!(approx(blended, (1.0 + 4.0 * prior_rate) / 5.0));
    seq.soften(0.0);
    assert!(approx(
        seq.rates(&STEP, Some(&prior), &ColdPrior::None).unwrap()[0],
        prior_rate
    ));
}

/// 2026-10-10: Mutation guard for a stale EMA: a model that stops decaying (decay 1) keeps
/// 200 old rejects against 20 new accepts and reads below 0.1; the decaying one reads above
/// 0.6. A decay that is skipped in `fold` fails the previous test's bounds.
#[test]
fn a_model_that_does_not_decay_cannot_follow_a_regime_change() {
    let stale = AcceptParams { decay: 1.0, ..STEP };
    let mut r = AcceptCounts::default();
    for _ in 0..200 {
        r.observe_step(&stale, 1, 0);
    }
    for _ in 0..20 {
        r.observe_step(&stale, 1, 1);
    }
    assert!(r.rates(&stale, None, &ColdPrior::None).unwrap()[0] < 0.1);
}

/// 2026-10-10: The rate form with a steady-state seed equals a seeded EWMA of the samples
/// (alpha = 1 - decay) at every step, including the first.
#[test]
fn the_rate_form_is_a_seeded_ewma() {
    for alpha in [0.5, 0.15, 1.0] {
        let p = AcceptParams {
            decay: 1.0 - alpha,
            prior_weight: 0.0,
            cold_weight: 0.0,
            seed: SeedWeight::SteadyState,
        };
        let mut r = AcceptCounts::default();
        let mut ewma: Option<f64> = None;
        for (t, x) in [0.84, 0.77, 0.73, 0.917, 0.2, 0.6].into_iter().enumerate() {
            r.observe_rate(&p, 1, x);
            let e = ewma.map_or(x, |e| alpha * x + (1.0 - alpha) * e);
            ewma = Some(e);
            let got = r.rate(1).unwrap();
            assert!((got - e).abs() < 1e-12, "alpha {alpha} t {t}: {got} vs {e}");
        }
    }
}

#[test]
fn a_rate_outside_the_positions_or_not_finite_is_ignored() {
    let p = AcceptParams {
        decay: 0.5,
        prior_weight: 0.0,
        cold_weight: 0.0,
        seed: SeedWeight::SteadyState,
    };
    let mut r = AcceptCounts::default();
    r.observe_rate(&p, 0, 0.5);
    r.observe_rate(&p, MAX_POSITIONS + 1, 0.5);
    r.observe_rate(&p, 1, f64::NAN);
    assert_eq!(r, AcceptCounts::default());
    r.observe_rate(&p, 2, 1.5);
    assert_eq!(r.rate(2), Some(1.0), "clamped to a probability");
    assert!(!r.observed(1));
}

/// 2026-10-10: Cold-start rates stand in where nothing was observed and weigh `cold_weight`
/// observations against data: one reject at position 2 no longer reads as "never accepted".
#[test]
fn cold_rates_smooth_a_first_observation() {
    let p = AcceptParams {
        cold_weight: 4.0,
        ..STEP
    };
    let cold = ColdPrior::Rates(vec![0.6, 0.5]);
    let empty = AcceptCounts::default();
    let c = empty.rates(&p, None, &cold).unwrap();
    assert_eq!(&c[..3], &[0.6, 0.5, 0.5]);
    let mut r = AcceptCounts::default();
    r.observe_step(&p, 7, 1);
    let c = r.rates(&p, None, &cold).unwrap();
    assert!(approx(c[0], (1.0 + 2.4) / 5.0) && approx(c[1], 2.0 / 5.0));
    assert_eq!(
        r.rates(&STEP, None, &cold).unwrap()[1],
        0.0,
        "cold_weight 0 ignores them"
    );
}

/// 2026-10-10: A chained cold prior pulls a deeper position towards the one before it; the
/// first position has nothing to chain from.
#[test]
fn a_chained_prior_pulls_each_position_towards_the_previous() {
    let p = AcceptParams {
        cold_weight: 4.0,
        ..STEP
    };
    let mut r = AcceptCounts::default();
    for _ in 0..4 {
        r.observe_step(&p, 2, 2);
    }
    r.observe_step(&p, 2, 1);
    let own = r.rates(&p, None, &ColdPrior::None).unwrap();
    let chained = r.rates(&p, None, &ColdPrior::Chained).unwrap();
    assert_eq!(chained[0], own[0], "position 1 has no predecessor");
    assert!(
        chained[1] > own[1] && chained[1] < chained[0],
        "{chained:?} vs {own:?}"
    );
    assert_eq!(
        chained[2], chained[1],
        "unobserved: the previous position's rate"
    );
}
