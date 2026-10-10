// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The adaptive DFlash draft count: what a verify observes, the blended rates,
//! expected tokens, the cost table, the choice and its refusal to speculate at a loss.
//!
//! Owner: speculative.
//! Invariants: none beyond the types.

use super::*;

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

/// 2026-10-10: Rates from conditional values, the rest equal to the last.
fn flat(prefix: &[f64]) -> [f64; MAX_POSITIONS] {
    let mut c = [*prefix.last().unwrap(); MAX_POSITIONS];
    c[..prefix.len()].copy_from_slice(prefix);
    c
}

/// 2026-10-10: A partial accept observes accepts before the first reject and the reject; a
/// full accept observes no reject and nothing past the verified drafts.
#[test]
fn a_verify_observes_accepts_then_the_reject_and_nothing_past_it() {
    let mut r = AcceptRates::default();
    r.observe(4, 2);
    let c = r.rates(None).unwrap();
    assert_eq!(
        &c[..4],
        &[1.0, 1.0, 0.0, 0.0],
        "position 3 takes position 2's rate"
    );
    let mut full = AcceptRates::default();
    full.observe(3, 3);
    assert_eq!(&full.rates(None).unwrap()[..4], &[1.0; 4]);
    assert_eq!(full.obs[3], 0.0, "no observation past the verified drafts");
    assert!(AcceptRates::default().rates(None).is_none());
}

/// 2026-10-10: Counts decay per observation, so recent steps dominate; the prior weighs
/// PRIOR_WEIGHT observations and stands alone where the sequence has none.
#[test]
fn rates_decay_and_blend_with_the_prior() {
    let mut r = AcceptRates::default();
    for _ in 0..200 {
        r.observe(1, 0);
    }
    for _ in 0..20 {
        r.observe(1, 1);
    }
    let c0 = r.rates(None).unwrap()[0];
    assert!(
        c0 > 0.6 && c0 < 0.7,
        "20 recent accepts over a decayed past: {c0}"
    );

    let mut prior = AcceptRates::default();
    prior.observe(1, 0);
    prior.observe(1, 1);
    let prior_rate = 1.0 / 1.95;
    let mut seq = AcceptRates::default();
    assert!(approx(seq.rates(Some(&prior)).unwrap()[0], prior_rate));
    seq.observe(1, 1);
    let blended = seq.rates(Some(&prior)).unwrap()[0];
    assert!(approx(
        blended,
        (1.0 + PRIOR_WEIGHT * prior_rate) / (1.0 + PRIOR_WEIGHT)
    ));
    seq.soften(0.0);
    assert!(approx(seq.rates(Some(&prior)).unwrap()[0], prior_rate));
}

#[test]
fn expected_tokens_is_one_plus_the_survival_sum() {
    let c = flat(&[0.6, 0.5, 0.4]);
    assert!(approx(expected_tokens(&c, 0), 1.0));
    assert!(approx(expected_tokens(&c, 1), 1.6));
    assert!(approx(expected_tokens(&c, 2), 1.9));
    assert!(approx(expected_tokens(&c, 3), 2.02));
}

#[test]
fn the_cost_table_parses_consecutive_positive_entries_only() {
    let t = CostModel::parse("0:32, 1:44,2:50").unwrap();
    assert_eq!(
        (t.max_drafts(), t.cost(0), t.cost(2), t.cost(3)),
        (2, Some(32.0), Some(50.0), None)
    );
    let t = CostModel::parse("1:44,2:50").unwrap();
    assert_eq!((t.cost(0), t.cost(1)), (None, Some(44.0)));
    for bad in [
        "",
        "0:32",
        "2:50",
        "0:32,2:50",
        "1:-1",
        "1:0",
        "1:x",
        "a:1",
        "1-44",
    ] {
        assert!(CostModel::parse(bad).is_err(), "{bad:?} accepted");
    }
}

/// 2026-10-10: A measured step moves the shared scale SCALE_ALPHA of the way to its ratio
/// to the table, and every priced count follows it.
#[test]
fn a_measured_step_rescales_every_count() {
    let mut t = CostModel::parse("0:30,1:40").unwrap();
    t.observe_step(1, 60.0);
    assert!(approx(t.scale(), 1.0 + SCALE_ALPHA * 0.5));
    assert!(approx(t.cost(0).unwrap(), 30.0 * t.scale()));
    t.observe_step(5, 1e9);
    t.observe_step(1, f64::NAN);
    assert!(
        approx(t.scale(), 1.0 + SCALE_ALPHA * 0.5),
        "unpriced or invalid steps ignored"
    );
}

/// 2026-10-10: At the reference acceptance (0.611 / 0.312 / 0.117 cumulative) and a 6 ms
/// per-row step slope, two drafts win; at a flat 0.2 plain decode wins (0), and without a
/// plain-decode entry the best count is kept; the cap bounds the choice.
#[test]
fn the_choice_maximises_tokens_per_ms_and_suspends_at_a_loss() {
    let costs = CostModel::parse("0:32,1:44,2:50,3:56,4:62").unwrap();
    let reference = flat(&[0.611, 0.312 / 0.611, 0.117 / 0.312, 0.028 / 0.117]);
    assert_eq!(choose(&costs, &reference, 7), 2);
    assert_eq!(choose(&costs, &reference, 1), 1);
    let poor = flat(&[0.2]);
    assert_eq!(choose(&costs, &poor, 7), 0);
    let no_decode = CostModel::parse("1:44,2:50").unwrap();
    assert_eq!(choose(&no_decode, &poor, 7), 1);
    let rich = flat(&[0.95]);
    assert_eq!(
        choose(&costs, &rich, 7),
        4,
        "capped by the widest priced count"
    );
}

/// 2026-10-10: Plain decode must win by more than SUSPEND_MARGIN: one draft at 1/32 ms per
/// token minus 2% keeps speculating, minus 4% suspends.
#[test]
fn a_near_tie_keeps_speculating() {
    let costs = CostModel::parse("0:32,1:44").unwrap();
    let tie = |gap: f64| {
        let tput = (1.0 - gap) / 32.0;
        let c1 = tput * 44.0 - 1.0;
        choose(&costs, &flat(&[c1]), 1)
    };
    assert_eq!(tie(0.02), 1);
    assert_eq!(tie(0.04), 0);
}

/// 2026-10-10: With nothing measured the cap stands; afterwards every EXPLORE_EVERY-th step
/// verifies one draft more than the choice, within the cap.
#[test]
fn the_controller_starts_at_the_cap_and_explores_one_deeper() {
    let mut k = AdaptiveK::new(CostModel::parse("0:32,1:44,2:50,3:56,4:62").unwrap());
    let mut seq = SeqAdaptiveK::default();
    assert_eq!(k.drafts(&seq, 5), 5);
    let mut seen = Vec::new();
    for _ in 0..EXPLORE_EVERY {
        k.observe(&mut seq, 3, 1);
        seen.push(k.drafts(&seq, 3));
    }
    // 2026-10-10: Accept-1-of-3 rates (1, 0, 0) choose one draft; the step whose counter
    // reaches EXPLORE_EVERY - 1 verifies two.
    let mut want = vec![1usize; EXPLORE_EVERY as usize];
    want[EXPLORE_EVERY as usize - 2] = 2;
    assert_eq!(seen, want);
    assert_eq!(seq.steps, EXPLORE_EVERY);
    assert_eq!(
        k.global, seq.rates,
        "one sequence: the prior holds the same counts"
    );
}
