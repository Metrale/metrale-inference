// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Parallel layouts: the engine's split, the property over every N <= 8 and every
//! d <= N, mixed degrees with a sub-group, and every refusal.

use super::*;

fn split(unit: u64, pad: bool, remainder: Remainder, min: u64, max: u64) -> DimSplit {
    DimSplit {
        unit,
        pad,
        remainder,
        scope: Scope::Every,
        degree: DegreeBounds { min, max },
        tie: None,
        why: "test".into(),
    }
}

fn map<V: Clone>(pairs: &[(&str, V)]) -> BTreeMap<String, V> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
}

/// 2026-10-23: `Front` is the engine's `tp_split` at every degree, index and unit.
#[test]
fn front_is_the_engines_tp_split() {
    for d in 1..=8u64 {
        for unit in [1u64, 2, 8, 64, 256] {
            for units in d..=40 {
                let total = units * unit;
                for i in 0..d {
                    let ours = rank_slice(total, d, i, unit, Remainder::Front).unwrap();
                    let us = |x: u64| usize::try_from(x).unwrap();
                    let theirs = metrale_config::tp_split(us(total), us(d), us(i), us(unit)).unwrap();
                    assert_eq!(ours, (theirs.start as u64, theirs.len as u64), "{total}/{d}/{i}/{unit}");
                }
            }
        }
    }
}

/// 2026-10-23: For N in 1..=8 and every d <= N, under both remainders, with and without padding:
/// a placed part's group slices tile the padded dim in order, each a whole number of units and
/// none empty; the ranks outside the group hold nothing; a replicated part is whole everywhere;
/// a degree the split cannot serve is refused, never sliced.
#[test]
fn every_degree_at_every_rank_count_tiles_the_dim_or_is_refused() {
    for n in 1..=8u64 {
        for d in 1..=n {
            for remainder in [Remainder::Front, Remainder::Back, Remainder::Exact] {
                for (total, unit, pad) in [(2048, 64, true), (64, 1, false), (64, 2, false), (100, 64, true), (12288, 8, false), (5, 1, false)] {
                    let spec = map(&[("x", split(unit, pad, remainder, 1, 8))]);
                    let dims = map(&[("x", total)]);
                    let placed = place(&spec, &dims, n, &map(&[("x", d)]), Chosen::Instance);
                    let units = total.div_ceil(unit);
                    let servable = d == 1 || rank_slice(total, d, d - 1, unit, remainder).is_some();
                    let Ok(l) = placed else {
                        assert!(!servable, "n={n} d={d} {remainder:?} {total}/{unit}: refused a servable split");
                        continue;
                    };
                    assert!(servable);
                    let p = &l.parts["x"];
                    assert_eq!(p.degree, d);
                    let mut next = 0;
                    for rank in 0..n {
                        let s = slices(&spec, &dims, RankOf { layout: &l, rank }).unwrap()["x"];
                        if d == 1 {
                            assert_eq!(s, Some((0, total)), "replicated");
                            continue;
                        }
                        match (p.group.contains(&rank), s) {
                            (true, Some((start, len))) => {
                                assert_eq!(start, next, "contiguous");
                                assert!(len > 0 && len % unit == 0, "whole units");
                                next = start + len;
                            }
                            (false, None) => {}
                            other => panic!("n={n} d={d} rank {rank}: {other:?}"),
                        }
                    }
                    if d > 1 {
                        assert_eq!(next, units * unit, "n={n} d={d}: the group covers the padded dim");
                        assert_eq!(p.group, (0..d).collect::<Vec<_>>());
                    }
                }
            }
        }
    }
}

/// 2026-10-23: One layout with a replicated part, a part split unevenly over all three ranks and
/// a part on ranks {0, 1} only: rank 2 idles for the last (its dim is 0 in its shape).
#[test]
fn mixed_degrees_in_one_layout() {
    let spec = map(&[
        ("index_heads", split(1, false, Remainder::Front, 1, 8)),
        ("moe_inter", split(64, true, Remainder::Front, 1, 8)),
        ("shared_inter", split(256, false, Remainder::Front, 1, 8)),
    ]);
    let dims = map(&[("index_heads", 32), ("moe_inter", 2048), ("shared_inter", 2048), ("hidden", 4096)]);
    let degrees = map(&[("index_heads", 1), ("moe_inter", 3), ("shared_inter", 2)]);
    let l = place(&spec, &dims, 3, &degrees, Chosen::Instance).unwrap();
    assert_eq!(l.parts["shared_inter"].group, [0, 1]);
    assert_eq!(l.parts["index_heads"].group, [0, 1, 2]);
    let at = |rank| rank_dims(&spec, &dims, RankOf { layout: &l, rank }).unwrap();
    let col = |d: &str| [at(0)[d], at(1)[d], at(2)[d]];
    assert_eq!(col("index_heads"), [32, 32, 32]);
    assert_eq!(col("moe_inter"), [704, 704, 640]);
    assert_eq!(col("shared_inter"), [1024, 1024, 0]);
    assert_eq!(col("hidden"), [4096, 4096, 4096]);
}

/// 2026-10-23: The remainder policy is the declared one: the same 64 heads over 3 ranks are
/// 22/21/21 at `front` and 22/22/20 at `back`.
#[test]
fn the_remainder_policy_decides_the_uneven_slices() {
    let lens = |r| (0..3).map(|i| rank_slice(64, 3, i, 1, r).unwrap().1).collect::<Vec<_>>();
    assert_eq!(lens(Remainder::Front), [22, 21, 21]);
    assert_eq!(lens(Remainder::Back), [22, 22, 20]);
    assert!(rank_slice(64, 3, 0, 1, Remainder::Exact).is_none(), "64 heads do not divide by 3");
    assert_eq!(rank_slice(64, 4, 3, 1, Remainder::Exact), Some((48, 16)));
}

#[test]
fn a_degree_the_part_forbids_is_refused() {
    let spec = map(&[("q_heads", split(1, false, Remainder::Front, 2, 4))]);
    let dims = map(&[("q_heads", 64)]);
    let try_d = |n, d| place(&spec, &dims, n, &map(&[("q_heads", d)]), Chosen::Instance);
    assert!(try_d(3, 3).is_ok());
    for (n, d, why) in [(3, 1, "below the part's min"), (8, 5, "above its max"), (3, 4, "above N"), (3, 0, "zero")] {
        assert!(matches!(try_d(n, d), Err(ParallelError::Split { .. })), "{why}");
    }
    assert!(matches!(
        place(&spec, &dims, 3, &BTreeMap::new(), Chosen::Instance),
        Err(ParallelError::Split { .. })
    ), "a part without a stated degree");
    assert!(matches!(
        place(&spec, &dims, 3, &map(&[("q_heads", 3), ("nope", 2)]), Chosen::Instance),
        Err(ParallelError::UnknownDim(_))
    ));
    assert!(matches!(place(&spec, &dims, 0, &map(&[("q_heads", 1)]), Chosen::Instance), Err(ParallelError::Rank { .. })));
}

#[test]
fn malformed_splits_are_refused() {
    let dims = map(&[("x", 100)]);
    let d3 = map(&[("x", 3)]);
    let refused = |s: DimSplit| place(&map(&[("x", s)]), &dims, 3, &d3, Chosen::Instance).is_err();
    assert!(refused(split(0, false, Remainder::Front, 1, 8)), "unit 0");
    assert!(refused(split(64, false, Remainder::Front, 1, 8)), "100 is not whole 64-units and does not pad");
    assert!(refused(split(64, true, Remainder::Front, 1, 8)), "two units for three ranks");
    assert!(refused(split(1, false, Remainder::Front, 3, 2)), "min above max");
    let mut no_why = split(1, false, Remainder::Front, 1, 8);
    no_why.why = " ".into();
    assert!(refused(no_why));
    assert!(!refused(split(1, false, Remainder::Front, 1, 8)));
}

#[test]
fn only_linear_ops_carry_a_partial() {
    assert!(passes_partial(&OpKind::Blend, &[true, false, true]));
    assert!(!passes_partial(&OpKind::Blend, &[true, false, false]), "a whole shared expert");
    assert!(!passes_partial(&OpKind::Blend, &[true, true, true]), "partial weights");
    assert!(passes_partial(&OpKind::ResidualAdd, &[true, true]));
    assert!(!passes_partial(&OpKind::ResidualAdd, &[false, true]), "x + partial");
    assert!(!passes_partial(&OpKind::HcPost, &[true, false, false, false]));
}

/// 2026-10-23: A tied dim takes its part's placement and is not stated by the deployment; a tie
/// to another number of units, or to a tied dim, is refused.
#[test]
fn a_tied_dim_follows_its_part() {
    let mut v = split(2, false, Remainder::Front, 1, 32);
    v.tie = Some("k".into());
    let spec = map(&[("k", split(2, false, Remainder::Front, 1, 32)), ("v", v.clone())]);
    let dims = map(&[("k", 64), ("v", 64)]);
    let l = place(&spec, &dims, 3, &map(&[("k", 3)]), Chosen::Instance).unwrap();
    assert_eq!(l.parts["v"], l.parts["k"]);
    assert_eq!(parts(&spec).collect::<Vec<_>>(), ["k"]);
    assert_eq!(part_of(&spec, "v"), "k");
    assert!(place(&spec, &dims, 3, &map(&[("k", 3), ("v", 3)]), Chosen::Instance).is_err(), "a tied dim is not stated");
    let uneven = map(&[("k", 64), ("v", 32)]);
    assert!(place(&spec, &uneven, 3, &map(&[("k", 3)]), Chosen::Instance).is_err(), "unequal units");
}
