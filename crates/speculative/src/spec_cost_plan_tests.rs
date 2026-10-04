// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Tests for the measured planner, on synthetic tables whose answer is known.
//!
//! Owner: speculative.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use super::super::{SCHEMA, TableKey};
use super::*;

/// 2026-10-04: A table over widths 1, 2, 4, 8 and depths 0..=3: a step at width `n`, depth `k`
/// takes `10 + n + row_ms·n·k` ms and `1 + 0.1n + row_j·n·k` J to verify, and `k` ms and
/// `0.05k` J to draft.
fn table(row_ms: f64, row_j: f64) -> CostTable {
    let key = TableKey {
        schema: SCHEMA,
        box_class: "gb10".into(),
        recipe: "r".into(),
        plan_digests: BTreeMap::from([("verify".to_string(), "d".to_string())]),
    };
    let mut cells = Vec::new();
    for n in [1usize, 2, 4, 8] {
        for k in 0..=3usize {
            let (nf, kf) = (n as f64, k as f64);
            cells.push(Cell {
                n,
                k,
                verify_ms: 10.0 + nf + row_ms * nf * kf,
                verify_j: 1.0 + 0.1 * nf + row_j * nf * kf,
                draft_ms: kf,
                draft_j: 0.05 * kf,
            });
        }
    }
    CostTable::parse(&CostTable::render(&key, &cells).unwrap()).unwrap()
}

/// 2026-10-04: Buckets `lp <= -1` → 0.2, `<= -0.1` → 0.6, `<= 0` → 0.95; priors 0.8, 0.6, 0.5.
fn cal() -> AcceptanceCalibration {
    AcceptanceCalibration::parse(
        "[drafter]\nweights_sha256 = \"x\"\nvocab = 1\nquantization = \"bf16\"\ncontext = true\n\
         [acceptance]\nedges = [-1.0, -0.1, 0.0]\np_accept = [0.2, 0.6, 0.95]\n\
         prior_by_position = [0.8, 0.6, 0.5]\n",
    )
    .unwrap()
}

#[test]
fn cheap_rows_go_deep_and_a_second_row_that_does_not_pay_stays_at_one() {
    assert_eq!(propose_depth(&table(0.1, 0.01), &cal(), 2, 0.0), 3);
    // 2026-10-04: At n = 2 a first draft pays (tokens per joule) and a second does not when a
    // row costs between ~0.19 and ~0.45 J; 0.3 is inside.
    assert_eq!(propose_depth(&table(1.0, 0.3), &cal(), 2, 0.0), 1);
}

#[test]
fn no_drafting_when_it_is_neither_faster_nor_cheaper() {
    // 2026-10-04: A draft row costs a whole step: depth 0 is at least as fast and cheaper.
    assert_eq!(propose_depth(&table(40.0, 4.0), &cal(), 1, 0.0), 0);
}

#[test]
fn slack_admits_a_slower_depth_that_saves_energy() {
    // 2026-10-04: Rows cost no energy and 12 ms each. At n = 1 depth 3 has the most tokens per
    // joule (2.52 / 1.25) but 2.52 / 50 tokens per ms, under depth 1's 1.8 / 24; depth 0 is
    // faster but the least per joule. Slack 0 must stop at depth 1; slack 0.9 admits depth 3.
    let t = table(12.0, 0.0);
    assert_eq!(propose_depth(&t, &cal(), 1, 0.0), 1);
    assert_eq!(propose_depth(&t, &cal(), 1, 0.9), 3);
}

#[test]
fn depths_respect_every_cap() {
    let t = table(0.1, 0.01);
    let c = cal();
    let conf: Vec<Vec<f32>> = vec![vec![-0.01; 3], vec![-0.01; 2], vec![], vec![-0.01; 3]];
    let refs: Vec<&[f32]> = conf.iter().map(Vec::as_slice).collect();
    let known = [3, 2, 0, 3];
    let d = sequence_depths(&t, &c, &refs, &known, 3, 64, 0.0);
    assert_eq!(d, vec![3, 2, 0, 3], "cheap rows fill every proposed draft");
    let d = sequence_depths(&t, &c, &refs, &known, 2, 64, 0.0);
    assert!(d.iter().all(|&k| k <= 2), "{d:?}");
    let d = sequence_depths(&t, &c, &refs, &known, 3, 5, 0.0);
    assert!(d.iter().sum::<usize>() <= 5, "{d:?}");
    assert_eq!(
        sequence_depths(&t, &c, &refs, &known, 3, 2, 0.0),
        vec![0; 4]
    );
    assert_eq!(
        sequence_depths(&t, &c, &refs, &known, 0, 64, 0.0),
        vec![0; 4]
    );
}

#[test]
fn a_contested_row_goes_to_the_more_confident_sequence_and_ties_to_the_earlier() {
    // 2026-10-04: Rows are cheap, so only the budget (one row beyond one draft each) binds.
    let t = table(0.1, 0.01);
    let conf: Vec<Vec<f32>> = vec![vec![-2.0; 3], vec![-0.01; 3]];
    let refs: Vec<&[f32]> = conf.iter().map(Vec::as_slice).collect();
    assert_eq!(
        sequence_depths(&t, &cal(), &refs, &[3, 3], 3, 3, 0.0),
        vec![1, 2]
    );
    let same: Vec<Vec<f32>> = vec![vec![-0.5; 3]; 2];
    let refs: Vec<&[f32]> = same.iter().map(Vec::as_slice).collect();
    assert_eq!(
        sequence_depths(&t, &cal(), &refs, &[3, 3], 3, 3, 0.0),
        vec![2, 1]
    );
}

#[test]
fn rows_go_to_the_confident_sequences_first_and_the_plan_is_deterministic() {
    let t = table(1.5, 0.12);
    let c = cal();
    let conf: Vec<Vec<f32>> = vec![vec![-2.0; 3], vec![-0.01; 3]];
    let refs: Vec<&[f32]> = conf.iter().map(Vec::as_slice).collect();
    let d = sequence_depths(&t, &c, &refs, &[3, 3], 3, 64, 0.0);
    assert!(d[1] >= d[0], "{d:?}");
    assert_eq!(d, sequence_depths(&t, &c, &refs, &[3, 3], 3, 64, 0.0));
}

#[test]
fn raising_a_sequence_s_confidence_never_shortens_it() {
    let t = table(1.5, 0.12);
    let c = cal();
    let levels = [-3.0f32, -0.5, -0.05];
    for &other in &levels {
        let mut last = 0;
        for &mine in &levels {
            let conf: Vec<Vec<f32>> = vec![vec![other; 3], vec![mine; 3], vec![-0.5; 3]];
            let refs: Vec<&[f32]> = conf.iter().map(Vec::as_slice).collect();
            let d = sequence_depths(&t, &c, &refs, &[3, 3, 3], 3, 64, 0.0);
            assert!(d[1] >= last, "other {other} mine {mine}: {d:?}");
            last = d[1];
        }
    }
}

#[test]
fn rows_that_never_pay_keep_one_draft_each() {
    let t = table(8.0, 0.8);
    let conf: Vec<Vec<f32>> = vec![vec![-0.01; 3]; 4];
    let refs: Vec<&[f32]> = conf.iter().map(Vec::as_slice).collect();
    assert_eq!(
        sequence_depths(&t, &cal(), &refs, &[3; 4], 3, 64, 0.0),
        vec![1; 4]
    );
}
