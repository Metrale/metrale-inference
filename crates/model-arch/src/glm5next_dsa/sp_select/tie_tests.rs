// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The split selection's top-k bytes equal the unsplit selection's, ties included.
//!
//! Reference: `glm5next_dsa_ref::topk_pools` over all rows (score descending, then pool index
//! ascending; `-0.0 == +0.0` under `partial_cmp`, as the radix keys fold them). Split path:
//! every rank runs the same top-k on its two owned slabs only, writes them into its copy of the
//! `[q, select_k]` buffer, and the two in-place gathers are replayed byte for byte from the
//! production `all_gather_plan`. The fixtures force ties at the top-k cutoff in the rows on both
//! sides of every slab boundary, a `-0.0` / `+0.0` tie, rows with fewer candidates than
//! `select_k`, and uneven slab sizes. Five negative controls must each break the equality.

use super::*;
use crate::glm5next_dsa_ref::topk_pools;
use crate::glm5next_sp_mhc::all_gather_plan;

/// 2026-10-14: The canonical tie rule and the wrong ones the controls substitute for it.
#[derive(Clone, Copy, Debug)]
enum Rule {
    Canonical,
    PoolDescending,
    TotalOrder,
}

/// 2026-10-14: How a pool-axis split (the rejected design) merges its per-rank top-k lists.
#[derive(Clone, Copy, Debug)]
enum PoolMerge {
    Global,
    LocalOrdinal,
    Concat,
}

struct Fx {
    q: usize,
    p: usize,
    k: usize,
    scores: Vec<f32>,
    cand: Vec<u8>,
}

const SENTINEL: u8 = 0xA5;

/// 2026-10-14: A seeded fixture of `q` rows over `p` pools at `world` ranks.
/// - Row r has candidates `0 ..= r * p / q` (causal: early rows have fewer than `k`).
/// - Scores take 6 values, so ties are everywhere; the two rows on either side of every slab
///   boundary hold the same tie-heavy vector (2.0 on every fifth pool, else 1.0), so the cutoff
///   falls inside a run of equal scores on both sides of the boundary.
/// - The first row of slab 1 scores pool 0 `-0.0`, pool 1 `+0.0`, every other pool `-1.0`.
fn fixture(q: usize, p: usize, k: usize, world: usize, seed: u64) -> Fx {
    let mut x = seed | 1;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let mut scores: Vec<f32> = (0..q * p).map(|_| (next() % 6) as f32 * 0.5).collect();
    let mut cand = vec![0u8; q * p];
    for r in 0..q {
        for c in &mut cand[r * p..r * p + (r * p / q + 1).min(p)] {
            *c = 1;
        }
    }
    let s = slabs(q, world);
    for &(lo, _) in &s[1..] {
        for row in [lo - 1, lo] {
            for (j, v) in scores[row * p..(row + 1) * p].iter_mut().enumerate() {
                *v = if j % 5 == 0 { 2.0 } else { 1.0 };
            }
        }
    }
    let z = s[1].0;
    for (j, v) in scores[z * p..(z + 1) * p].iter_mut().enumerate() {
        *v = match j {
            0 => -0.0,
            1 => 0.0,
            _ => -1.0,
        };
    }
    Fx {
        q,
        p,
        k,
        scores,
        cand,
    }
}

/// 2026-10-14: Per-row top-k of `rows` rows under `rule`; non-candidates score `f32::MIN`.
fn topk(rule: Rule, scores: &[f32], cand: &[u8], p: usize, rows: usize, k: usize) -> Vec<i32> {
    let cmp = |a: &(f32, usize), b: &(f32, usize)| match rule {
        Rule::Canonical => unreachable!("the canonical rule is the reference's own"),
        Rule::PoolDescending => b.0.partial_cmp(&a.0).unwrap().then(b.1.cmp(&a.1)),
        Rule::TotalOrder => b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)),
    };
    if matches!(rule, Rule::Canonical) {
        return topk_pools(scores, cand, p, rows, k);
    }
    let mut out = Vec::with_capacity(rows * k);
    for r in 0..rows {
        let mut v: Vec<(f32, usize)> = (0..p)
            .map(|j| {
                let s = if cand[r * p + j] != 0 {
                    scores[r * p + j]
                } else {
                    f32::MIN
                };
                (s, j)
            })
            .collect();
        v.sort_by(cmp);
        out.extend(v.iter().take(k).map(|&(_, j)| j as i32));
    }
    out
}

fn bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// 2026-10-14: The in-place gather of `shards`, replayed on every rank's buffer from the
/// production plan: what rank s sends to peer d lands where d's plan receives from s.
fn gather(bufs: &mut [Vec<u8>], shards: &[(usize, usize)], row_bytes: usize) {
    let world = bufs.len();
    let plans: Vec<_> = (0..world)
        .map(|r| all_gather_plan(r, shards, row_bytes))
        .collect();
    let mut writes = Vec::new();
    for (s, plan) in plans.iter().enumerate() {
        for x in plan {
            let y = plans[x.peer].iter().find(|y| y.peer == s).unwrap();
            assert!(x.send_bytes > 0 && x.send_bytes == y.recv_bytes);
            let data = bufs[s][x.send_off..x.send_off + x.send_bytes].to_vec();
            writes.push((x.peer, y.recv_off, data));
        }
    }
    for (d, off, data) in writes {
        bufs[d][off..off + data.len()].copy_from_slice(&data);
    }
}

/// 2026-10-14: Every rank's buffer after the split selection over the slab plan `s` (rank r
/// owns `s[r]` and `s[2 world - 1 - r]`) and its two gathers.
fn split_select(rule: Rule, fx: &Fx, world: usize, s: &[(usize, usize)]) -> Vec<Vec<u8>> {
    let rb = fx.k * 4;
    let last = SLABS_PER_RANK * world - 1;
    let mut bufs = vec![vec![SENTINEL; fx.q * rb]; world];
    for (r, buf) in bufs.iter_mut().enumerate() {
        for (lo, hi) in [s[r], s[last - r]] {
            let sel = topk(
                rule,
                &fx.scores[lo * fx.p..hi * fx.p],
                &fx.cand[lo * fx.p..hi * fx.p],
                fx.p,
                hi - lo,
                fx.k,
            );
            buf[lo * rb..hi * rb].copy_from_slice(&bytes(&sel));
        }
    }
    for list in [
        (0..world).map(|r| s[r]).collect::<Vec<_>>(),
        (0..world).map(|r| s[last - r]).collect(),
    ] {
        gather(&mut bufs, &list, rb);
    }
    bufs
}

fn reference(fx: &Fx) -> Vec<u8> {
    bytes(&topk_pools(&fx.scores, &fx.cand, fx.p, fx.q, fx.k))
}

fn agrees(bufs: &[Vec<u8>], want: &[u8]) -> bool {
    bufs.iter().all(|b| b == want)
}

/// 2026-10-14: A pool-axis split: rank r takes the top-k of pools `shards(p, world)[r]`, then
/// the lists are merged by `merge`.
fn pool_split(fx: &Fx, world: usize, merge: PoolMerge) -> Vec<u8> {
    let parts = shards(fx.p, world);
    let mut out = Vec::with_capacity(fx.q * fx.k);
    for r in 0..fx.q {
        let mut all: Vec<(f32, usize, usize)> = Vec::new();
        for &(plo, phi) in &parts {
            let n = phi - plo;
            let row_s: Vec<f32> = fx.scores[r * fx.p + plo..r * fx.p + phi].to_vec();
            let row_c: Vec<u8> = fx.cand[r * fx.p + plo..r * fx.p + phi].to_vec();
            for j in topk_pools(&row_s, &row_c, n, 1, fx.k.min(n)) {
                let g = plo + j as usize;
                let s = if fx.cand[r * fx.p + g] != 0 {
                    fx.scores[r * fx.p + g]
                } else {
                    f32::MIN
                };
                all.push((s, g, j as usize));
            }
        }
        match merge {
            PoolMerge::Global => {
                all.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)))
            }
            PoolMerge::LocalOrdinal => {
                all.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.2.cmp(&b.2)))
            }
            PoolMerge::Concat => {}
        }
        out.extend(all.iter().take(fx.k).map(|&(_, g, _)| g as i32));
    }
    bytes(&out)
}

const SHAPES: [(usize, usize); 3] = [(96, 128), (97, 128), (101, 130)];
const K: usize = 16;

/// 2026-10-14: The split selection equals the unsplit one byte for byte on every rank, for
/// every fixture and world size, and the fixtures do contain cutoff ties at slab boundaries.
#[test]
fn split_equals_unsplit_with_ties_at_every_boundary() {
    for world in 2..=4 {
        for (i, &(q, p)) in SHAPES.iter().enumerate() {
            let fx = fixture(q, p, K, world, 0x9E37_79B9 + i as u64);
            let want = reference(&fx);
            let bufs = split_select(Rule::Canonical, &fx, world, &slabs(q, world));
            assert!(agrees(&bufs, &want), "q {q} p {p} world {world}");
            for &(lo, _) in &slabs(q, world)[1..] {
                let row = &fx.scores[lo * p..(lo + 1) * p];
                let kth = {
                    let mut v: Vec<f32> = row
                        .iter()
                        .zip(&fx.cand[lo * p..(lo + 1) * p])
                        .filter(|(_, c)| **c != 0)
                        .map(|(s, _)| *s)
                        .collect();
                    v.sort_by(|a, b| b.partial_cmp(a).unwrap());
                    v.get(K - 1).copied()
                };
                if let Some(t) = kth {
                    let tied = row.iter().zip(&fx.cand[lo * p..(lo + 1) * p]);
                    let n = tied.filter(|(s, c)| **c != 0 && **s == t).count();
                    assert!(
                        n > 1,
                        "row {lo}: no tie at the cutoff, the fixture is too weak"
                    );
                }
            }
        }
    }
}

/// 2026-10-14: Control (i): ties by pool index descending.
#[test]
fn control_pool_descending_tie_rule_fails() {
    let fx = fixture(96, 128, K, 3, 0x9E37_79B9);
    let bufs = split_select(Rule::PoolDescending, &fx, 3, &slabs(96, 3));
    assert!(!agrees(&bufs, &reference(&fx)));
}

/// 2026-10-14: Control (ii): a pool-axis split merged with ties by the rank-local pool ordinal
/// (the boundary tie problem the row split avoids); merged by the global index it agrees.
#[test]
fn control_pool_split_local_ordinal_merge_fails() {
    let fx = fixture(96, 128, K, 3, 0x9E37_79B9);
    let want = reference(&fx);
    assert_eq!(pool_split(&fx, 3, PoolMerge::Global), want);
    assert_ne!(pool_split(&fx, 3, PoolMerge::LocalOrdinal), want);
}

/// 2026-10-14: Control (iii): an IEEE total order, which ranks `-0.0` below `+0.0`.
#[test]
fn control_total_order_on_signed_zero_fails() {
    let fx = fixture(96, 128, K, 3, 0x9E37_79B9);
    let bufs = split_select(Rule::TotalOrder, &fx, 3, &slabs(96, 3));
    assert!(!agrees(&bufs, &reference(&fx)));
}

/// 2026-10-14: Control (iv): a slab plan with a one-row gap at a boundary leaves a row no rank
/// selects (the buffers keep the sentinel there).
#[test]
fn control_slab_gap_fails() {
    let fx = fixture(96, 128, K, 3, 0x9E37_79B9);
    let mut s = slabs(96, 3);
    s[1].0 += 1;
    let bufs = split_select(Rule::Canonical, &fx, 3, &s);
    assert!(!agrees(&bufs, &reference(&fx)));
}

/// 2026-10-14: Control (v): a pool-axis split whose per-rank lists are concatenated, not merged.
#[test]
fn control_pool_split_concat_fails() {
    let fx = fixture(96, 128, K, 3, 0x9E37_79B9);
    assert_ne!(pool_split(&fx, 3, PoolMerge::Concat), reference(&fx));
}
