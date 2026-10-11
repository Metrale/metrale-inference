// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The decision rule against hand-computed optima: argmax per objective, the
//! margins (hysteresis and suspension), the energy floor, monotonicity in acceptance and cost,
//! and the per-sequence cut.
//!
//! Owner: speculative.
//! Invariants: none beyond the types.

use super::super::chain::expected_tokens;
use super::*;

/// 2026-10-10: Candidates at a flat conditional rate `p` for depths `0..costs.len()` with
/// `costs[k] = (ms, joules)`, one stream.
fn flat(p: f64, costs: &[(f64, f64)]) -> Vec<Candidate> {
    costs
        .iter()
        .enumerate()
        .map(|(k, &(ms, j))| {
            let e = expected_tokens(&[p], k);
            Candidate {
                k,
                tokens: e,
                slowest: e,
                cost: StepCost { ms, j: Some(j) },
            }
        })
        .collect()
}

const T: Objective = Objective::Throughput;

/// 2026-10-10: Hand-computed: p = 0.8, costs 10, 14, 17, 21 ms. E = 1, 1.8, 2.44, 2.952;
/// tokens/ms = 0.1, 0.1286, 0.1435, 0.1406, so 2 drafts.
#[test]
fn throughput_picks_the_hand_computed_optimum() {
    let c = flat(0.8, &[(10.0, 1.0), (14.0, 1.0), (17.0, 1.0), (21.0, 1.0)]);
    assert_eq!(choose(&T, &Margins::NONE, None, &c), Some(2));
    assert_eq!(choose(&T, &Margins::NONE, None, &[]), None);
}

/// 2026-10-10: Mutation guard for a wrong-sign cost: a rule that maximised tokens x ms (or
/// minimised tokens/ms) picks 3 or 0 here, not 2; and a cheaper deep step must never move the
/// choice shallower.
#[test]
fn a_cheaper_step_never_loses_depth_and_more_acceptance_never_loses_depth() {
    let base = [(10.0, 1.0), (14.0, 1.0), (17.0, 1.0), (21.0, 1.0)];
    let mut last = 0;
    for deep_ms in [30.0, 25.0, 21.0, 18.0, 17.5] {
        let mut costs = base;
        costs[3].0 = deep_ms;
        let k = choose(&T, &Margins::NONE, None, &flat(0.8, &costs)).unwrap();
        assert!(
            k >= last,
            "cheaper depth 3 ({deep_ms} ms) chose {k} after {last}"
        );
        last = k;
    }
    assert_eq!(last, 3);
    let mut last = 0;
    for p in [0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0] {
        let k = choose(&T, &Margins::NONE, None, &flat(p, &base)).unwrap();
        assert!(k >= last, "p {p} chose {k} after {last}");
        last = k;
    }
    assert_eq!(choose(&T, &Margins::NONE, None, &flat(0.0, &base)), Some(0));
    assert_eq!(last, 3);
}

/// 2026-10-10: Energy and latency disagree on a table where depth costs time but saves
/// joules: Throughput stays at 1, Energy with slack 0.5 goes to 3, with slack 0 it may not
/// fall below depth 1's rate.
#[test]
fn energy_and_latency_objectives_disagree_where_they_should() {
    let c = flat(0.9, &[(10.0, 5.0), (12.0, 4.0), (20.0, 3.0), (28.0, 2.5)]);
    assert_eq!(choose(&T, &Margins::NONE, None, &c), Some(1));
    assert_eq!(
        choose(&Objective::Latency, &Margins::NONE, None, &c),
        Some(1)
    );
    let loose = Objective::Energy {
        slack: 0.5,
        floor: FloorRef::Depth(1),
    };
    assert_eq!(choose(&loose, &Margins::NONE, None, &c), Some(3));
    let tight = Objective::Energy {
        slack: 0.0,
        floor: FloorRef::Depth(1),
    };
    assert_eq!(choose(&tight, &Margins::NONE, None, &c), Some(1));
    let best = Objective::Energy {
        slack: 0.0,
        floor: FloorRef::Best,
    };
    assert_eq!(choose(&best, &Margins::NONE, None, &c), Some(1));
    let no_j: Vec<Candidate> = c
        .iter()
        .map(|x| Candidate {
            cost: StepCost { j: None, ..x.cost },
            ..*x
        })
        .collect();
    assert_eq!(
        choose(&loose, &Margins::NONE, None, &no_j),
        Some(1),
        "no joules: the floor"
    );
}

/// 2026-10-10: Latency follows the slowest stream: two streams, one that accepts and one that
/// never does; depth helps the sum but not the slowest, so Latency stays shallow.
#[test]
fn latency_follows_the_slowest_stream() {
    let costs = [(10.0, 1.0), (12.0, 1.0), (14.0, 1.0)];
    let c: Vec<Candidate> = costs
        .iter()
        .enumerate()
        .map(|(k, &(ms, j))| {
            let (a, b) = (expected_tokens(&[0.95], k), expected_tokens(&[0.0], k));
            Candidate {
                k,
                tokens: a + b,
                slowest: a.min(b),
                cost: StepCost { ms, j: Some(j) },
            }
        })
        .collect();
    assert_eq!(choose(&T, &Margins::NONE, None, &c), Some(2));
    assert_eq!(
        choose(&Objective::Latency, &Margins::NONE, None, &c),
        Some(0)
    );
}

/// 2026-10-10: Hysteresis: from depth 1 the deeper candidate must reach `1 + deeper` times
/// depth 1's value (reaching it switches); from depth 2 the shallower one must reach
/// `1 + shallower`; between the two the incumbent holds.
#[test]
fn margins_hold_the_incumbent_inside_the_band() {
    let m = Margins {
        deeper: 0.02,
        shallower: 0.02,
        suspend: 0.0,
    };
    let pair = |v2: f64| {
        let one = Candidate {
            k: 1,
            tokens: 1.0,
            slowest: 1.0,
            cost: StepCost { ms: 1.0, j: None },
        };
        vec![
            one,
            Candidate {
                k: 2,
                tokens: v2,
                slowest: v2,
                ..one
            },
        ]
    };
    assert_eq!(choose(&T, &m, Some(1), &pair(1.01)), Some(1));
    assert_eq!(choose(&T, &m, Some(2), &pair(1.01)), Some(2));
    assert_eq!(
        choose(&T, &m, Some(1), &pair(1.02)),
        Some(2),
        "reaching the margin switches"
    );
    assert_eq!(choose(&T, &m, Some(2), &pair(0.99)), Some(2));
    assert_eq!(choose(&T, &m, Some(2), &pair(0.97)), Some(1));
    assert_eq!(
        choose(&T, &Margins::NONE, None, &pair(1.0)),
        Some(1),
        "ties keep the shallower"
    );
}

/// 2026-10-10: Plain decode must beat the best speculative depth by more than `suspend`.
#[test]
fn suspension_needs_its_margin() {
    let m = Margins {
        deeper: 0.0,
        shallower: 0.0,
        suspend: 0.03,
    };
    let two = |v0: f64| {
        let spec = Candidate {
            k: 1,
            tokens: 1.0,
            slowest: 1.0,
            cost: StepCost { ms: 1.0, j: None },
        };
        vec![
            Candidate {
                k: 0,
                tokens: v0,
                slowest: v0,
                ..spec
            },
            spec,
        ]
    };
    assert_eq!(choose(&T, &m, None, &two(1.02)), Some(1));
    assert_eq!(choose(&T, &m, None, &two(1.04)), Some(0));
    assert_eq!(choose(&T, &m, Some(1), &two(1.04)), Some(0));
}

fn linear(row_ms: f64) -> impl Fn(usize) -> Option<StepCost> {
    move |rows| {
        Some(StepCost {
            ms: 10.0 + row_ms * rows as f64,
            j: Some(1.0 + 0.1 * rows as f64),
        })
    }
}

/// 2026-10-10: Cheap rows fill every proposed draft; caps on depth, held drafts and the row
/// budget all bind; a budget below the mandatory first drafts verifies none.
#[test]
fn the_cut_respects_every_cap() {
    let r = |_: usize, _: usize| 0.9;
    let d = cut_depths(&T, r, &[3, 2, 0, 3], 3, 64, linear(0.01));
    assert_eq!(d, vec![3, 2, 0, 3]);
    assert!(
        cut_depths(&T, r, &[3, 2, 0, 3], 2, 64, linear(0.01))
            .iter()
            .all(|&k| k <= 2)
    );
    assert!(
        cut_depths(&T, r, &[3, 2, 0, 3], 3, 5, linear(0.01))
            .iter()
            .sum::<usize>()
            <= 5
    );
    assert_eq!(
        cut_depths(&T, r, &[3, 2, 0, 3], 3, 2, linear(0.01)),
        vec![0; 4]
    );
    assert_eq!(
        cut_depths(&T, r, &[3, 2, 0, 3], 0, 64, linear(0.01)),
        vec![0; 4]
    );
}

/// 2026-10-10: Expensive rows keep one draft each; a contested row goes to the sequence with
/// the larger marginal gain.
#[test]
fn the_cut_gives_rows_by_gain_and_stops_where_they_stop_paying() {
    let r = |_: usize, _: usize| 0.9;
    assert_eq!(cut_depths(&T, r, &[3; 4], 3, 64, linear(50.0)), vec![1; 4]);
    let skew = |i: usize, _: usize| if i == 0 { 0.1 } else { 0.9 };
    assert_eq!(
        cut_depths(&T, skew, &[3, 3], 3, 3, linear(0.01)),
        vec![1, 2]
    );
}
