// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Row ownership, the gather shard lists, the measured table and the split decision.

use super::*;

/// 2026-10-14: How many ranks own each row of a `k`-row pass, from every rank's [`owned`].
fn owners_per_row(k: usize, world: usize) -> Vec<usize> {
    let mut n = vec![0usize; k];
    for r in 0..world {
        for (lo, hi) in owned(k, world, r).unwrap() {
            for c in &mut n[lo..hi] {
                *c += 1;
            }
        }
    }
    n
}

/// 2026-10-14: Every row has exactly one owner, slabs are non-empty and contiguous in order,
/// and each rank's two slabs differ in size by at most one row from any other slab.
#[test]
fn every_row_has_exactly_one_owner() {
    for world in 2..=4 {
        for k in SLABS_PER_RANK * world..=300 {
            assert!(
                owners_per_row(k, world).iter().all(|&n| n == 1),
                "k {k} world {world}"
            );
            let s = slabs(k, world);
            assert_eq!(s.len(), SLABS_PER_RANK * world);
            assert_eq!((s[0].0, s[s.len() - 1].1), (0, k));
            assert!(s.windows(2).all(|w| w[0].1 == w[1].0 && w[0].0 < w[0].1));
            let (mn, mx) = s.iter().fold((usize::MAX, 0), |(a, b), &(lo, hi)| {
                (a.min(hi - lo), b.max(hi - lo))
            });
            assert!(mx - mn <= 1, "k {k} world {world}: slab sizes {mn}..{mx}");
        }
    }
}

/// 2026-10-14: Gather list `i` holds rank `r`'s `i`-th owned slab at index `r`.
#[test]
fn gather_shards_put_each_rank_slab_at_its_index() {
    for world in 2..=4 {
        for k in [SLABS_PER_RANK * world, 97, 4096, 8192, 8208] {
            let g = gather_shards(k, world);
            for r in 0..world {
                let own = owned(k, world, r).unwrap();
                assert_eq!((g[0][r], g[1][r]), (own[0], own[1]), "k {k} world {world}");
            }
        }
    }
}

/// 2026-10-14: Too few rows for every slab, or a rank outside the world, is an error, never
/// an empty shard the gather would refuse mid-pass.
#[test]
fn owned_refuses_empty_slabs_and_bad_ranks() {
    assert!(owned(5, 3, 0).is_err());
    assert!(owned(6, 3, 0).is_ok());
    assert!(owned(600, 3, 3).is_err());
}

/// 2026-10-14: Under causality a row's scores cost grows with its position; the zigzag pairs
/// keep the slowest rank within 0.1% of an even third at a full chunk, where contiguous thirds
/// leave the last rank ~1.67x of it.
#[test]
fn zigzag_balances_causal_cost() {
    let (k, world) = (8192usize, 3usize);
    let cost = |lo: usize, hi: usize| (lo..hi).map(|r| (r + 1) as f64).sum::<f64>();
    let ideal = cost(0, k) / world as f64;
    let zig = (0..world)
        .map(|r| {
            owned(k, world, r)
                .unwrap()
                .iter()
                .map(|&(lo, hi)| cost(lo, hi))
                .sum::<f64>()
        })
        .fold(0.0f64, f64::max);
    let contig = shards(k, world)
        .iter()
        .map(|&(lo, hi)| cost(lo, hi))
        .fold(0.0f64, f64::max);
    assert!(zig / ideal < 1.001, "zigzag {}", zig / ideal);
    assert!(contig / ideal > 1.6, "contiguous {}", contig / ideal);
}

/// 2026-10-14: An empty table or a k that does not strictly increase is refused.
#[test]
fn table_refuses_empty_and_unordered_rows() {
    assert!(MinPoolsTable::new(vec![]).is_err());
    assert!(MinPoolsTable::new(vec![(4096, 2048), (4096, 1024)]).is_err());
    assert!(MinPoolsTable::new(vec![(8192, 2048), (4096, 1024)]).is_err());
    assert!(MinPoolsTable::new(vec![(4096, 2048), (8192, 1024)]).is_ok());
}

/// 2026-10-14: Below the smallest and above the largest measured k never split; between two
/// measured k the smaller one's threshold applies; a pass exactly at a threshold splits and one
/// pool below does not.
#[test]
fn table_splits_only_at_measured_points() {
    let t = MinPoolsTable::new(vec![(2048, 4096), (4096, 3000), (8192, 2000)]).unwrap();
    assert!(!t.split(2047, 1 << 20), "below the measured k range");
    assert!(!t.split(8193, 1 << 20), "above the measured k range");
    assert!(
        t.split(2048, 4096) && !t.split(2048, 4095),
        "at / under k 2048's threshold"
    );
    assert!(
        t.split(6000, 3000) && !t.split(6000, 2999),
        "k 6000 takes k 4096's threshold"
    );
    assert!(
        t.split(8192, 2000) && !t.split(8192, 1999),
        "at / under k 8192's threshold"
    );
}

/// 2026-10-14: No decode or verify step (`is_prefill` false) ever splits, whatever the other
/// inputs; a prefill pass splits exactly when the lever, the world, the slabs and the table
/// allow it.
#[test]
fn applies_is_false_off_prefill_and_follows_the_table_on_it() {
    let t = MinPoolsTable::new(vec![(1, 1), (8192, 1)]).unwrap();
    for on in [false, true] {
        for world in 1..=4 {
            for k in [1usize, 2, 5, 6, 7, 8, 64, 4096, 8192] {
                for pools in [0usize, 1, 2052, 8196] {
                    assert!(!applies(on, world, k, pools, false, &t));
                    let want = on && world >= 2 && k >= SLABS_PER_RANK * world && pools >= 1;
                    assert_eq!(
                        applies(on, world, k, pools, true, &t),
                        want,
                        "on {on} world {world} k {k} pools {pools}"
                    );
                }
            }
        }
    }
    let strict = MinPoolsTable::new(vec![(1, 4096), (8192, 4096)]).unwrap();
    assert!(applies(true, 3, 8192, 4096, true, &strict));
    assert!(
        !applies(true, 3, 8192, 4095, true, &strict),
        "the table vetoes a prefill pass"
    );
    assert!(
        !applies(true, 3, 8193, 1 << 20, true, &strict),
        "an unmeasured k never splits"
    );
}
