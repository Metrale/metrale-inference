// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: Row tables and per-run selection.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

use super::*;

fn k(func: &str) -> KernelId {
    KernelId {
        module: "m".into(),
        func: func.into(),
    }
}

// 2026-09-30: Runs are maximal groups of equal k, as `batched_conv_gdn_route` forms them, and
// each asks the contiguity predicate once with its first sequence and length. Mutation:
// splitting on every sequence, or passing the wrong span, fails.
#[test]
fn runs_are_maximal_and_carry_their_contiguity() {
    let asked = std::cell::RefCell::new(Vec::new());
    let t = RowTable::from_seqs(&[4, 4, 4, 3, 2, 2], true, |first, n| {
        asked.borrow_mut().push((first, n));
        first != 3
    })
    .unwrap();
    assert_eq!(t.text(), "4x3 3x1! 2x2");
    let one = RowTable::from_seqs(&[4, 2, 2], true, |_, _| true).unwrap();
    assert_eq!(one.text(), "4x1! 2x2", "a run of one is never batched");
    assert!(RowTable::parse("4x1 2x2").is_err());
    assert_eq!(asked.into_inner(), vec![(0, 3), (4, 2)]);
    assert_eq!((t.rows(), t.seqs()), (19, 6));
    assert_eq!(RowTable::parse("4x3 3x1! 2x2").unwrap(), t);
    assert!(RowTable::from_seqs(&[], true, |_, _| true).is_err());
    assert!(RowTable::from_seqs(&[2, 0], true, |_, _| true).is_err());
    let u = RowTable::parse("uncarried: 4x8! 2x8!").unwrap();
    assert!(!u.carried && u.runs.iter().all(|r| !r.contiguous));
    assert_eq!(u.text(), "uncarried: 4x8! 2x8!");
    for bad in ["", "4x", "4x0", "0x2", "4x2 4x1", "4y2"] {
        assert!(RowTable::parse(bad).is_err(), "{bad}");
    }
}

// 2026-09-30: The first serving selector wins; a run no selector serves leaves the rule
// inapplicable; counts follow `Times`. Mutation: taking the last match, or ignoring `n` or
// `contiguous`, fails.
#[test]
fn each_run_takes_the_first_selector_that_serves_it() {
    let sel = |kk: (u64, u64), n: (u64, u64), c: Option<bool>, f: &str| RunSelect {
        k: kk,
        n,
        contiguous: c,
        carried: None,
        launches: vec![(k(f), Times::Once), (k("row"), Times::PerRow)],
        copies: (c == Some(false)).then_some(Times::PerSeqRowButLast),
    };
    let selects = vec![
        sel((2, 4), (1, 7), Some(true), "eager"),
        sel((2, 4), (8, 128), Some(true), "lazy"),
        sel((2, 4), (1, 128), Some(false), "frag"),
        sel((2, 4), (1, 128), None, "never"),
    ];
    let t = RowTable::parse("4x8 3x2! 2x7").unwrap();
    let got = resolve_runs(&t, &selects).unwrap();
    let names: Vec<&str> = got.iter().map(|r| r.launches[0].0.func.as_str()).collect();
    assert_eq!(names, ["lazy", "frag", "eager"]);
    assert_eq!(got[0].launch_count(), 1 + 32);
    assert_eq!((got[1].launch_count(), got[1].copy_count()), (1 + 6, 4));
    assert_eq!(got[2].copy_count(), 0);
    assert!(resolve_runs(&RowTable::parse("5x2").unwrap(), &selects).is_none());
    for t in [
        Times::Once,
        Times::PerSeq,
        Times::PerRow,
        Times::PerSeqRowButLast,
    ] {
        assert_eq!(Times::parse(t.name()), Some(t));
    }
}
