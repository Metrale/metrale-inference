// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Tests for the bucket ladder over a small rule set.
//!
//! Owner: metrale-circuit.
//! Invariants: none beyond the types.

use super::*;
use crate::rules::parse_rules;

fn rule(id: &str, rows: [u64; 2], modes: &str) -> String {
    format!(
        r#"
[[rule]]
id = "{id}"
pattern = [{{ op = "rms_norm" }}]
kernels = [{{ module = "norm", func = "rms_norm" }}]
repeat = "once"
emitter = "rms_norm"
rows = [{}, {}]
modes = [{modes}]
numerics = "reference"
priority = 10
cite = "test"
"#,
        rows[0], rows[1]
    )
}

fn ladder(rules: &str, mode: Mode, max: u64) -> Vec<(u64, u64)> {
    let rules = parse_rules(&format!("schema = 1\n{rules}")).unwrap();
    bucket_ladder(&rules, &[], mode, max)
        .into_iter()
        .map(|b| (b.lo, b.hi))
        .collect()
}

#[test]
fn the_ladder_splits_at_every_edge_of_the_modes_rules_and_tiles_the_range() {
    let rules = [
        rule("small", [1, 16], r#""multi_seq""#),
        rule("mid", [17, 64], r#""multi_seq""#),
        rule("wide", [65, 128], r#""multi_seq""#),
        rule("overlap", [33, 96], r#""multi_seq", "verify""#),
        rule("other_mode", [5, 7], r#""verify""#),
    ]
    .concat();
    let l = ladder(&rules, Mode::MultiSeq, 128);
    assert_eq!(l, [(1, 16), (17, 32), (33, 64), (65, 96), (97, 128)]);
    for w in l.windows(2) {
        assert_eq!(w[0].1 + 1, w[1].0, "no gap, no overlap");
    }
    assert_eq!(
        ladder(&rules, Mode::Verify, 10),
        [(1, 4), (5, 7), (8, 10)],
        "edges past the top are dropped, and only the mode's rules split"
    );
}

#[test]
fn a_mode_without_rules_is_one_bucket_and_max_zero_is_none() {
    let rules = rule("small", [1, 16], r#""multi_seq""#);
    assert_eq!(ladder(&rules, Mode::Draft, 300), [(1, 300)]);
    assert!(ladder(&rules, Mode::MultiSeq, 0).is_empty());
    let l = bucket_ladder(&parse_rules(&format!("schema = 1\n{rules}")).unwrap(), &[], Mode::MultiSeq, 40);
    assert_eq!(bucket_of(&l, 16), Some(Bucket { lo: 1, hi: 16 }));
    assert_eq!(bucket_of(&l, 17), Some(Bucket { lo: 17, hi: 40 }));
    assert_eq!(bucket_of(&l, 41), None);
}
