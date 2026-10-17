// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The sharded keys against the whole-row feed argmax.
//!
//! Owner: metrale-sampling.
//! Invariants: none beyond the types.
//!
//! Every case splits a row into contiguous slices (one to four, uneven, and the GLM-5.3
//! three-rank split), combines the slices' keys by `max` and checks the resolved token
//! against `feed_argmax_f32` over the whole row: planted ties across slice boundaries and
//! inside one thread's stride, `±0`, the `-1e30` floor, `-inf`, NaN, and masks that hit the
//! plain argmax.

use super::feed_argmax::{BLOCK, NO_ID, PLAIN_FLOOR, bf16_to_f32, feed_argmax_f32};
use super::feed_argmax_key::{masked_key, order_bits, plain_index, plain_key, resolve, shard_keys};
use rand::{Rng, SeedableRng, rngs::StdRng};

/// 2026-10-10: The combined token for `row` cut at `cuts` (slice starts after 0).
fn sharded(row: &[f32], cuts: &[usize], mask: [u32; 2]) -> u32 {
    let mut starts = vec![0];
    starts.extend_from_slice(cuts);
    starts.push(row.len());
    let (mut p, mut q) = (0u64, 0u64);
    for w in starts.windows(2) {
        let (kp, kq) = shard_keys(&row[w[0]..w[1]], w[0] as u32, mask);
        p = p.max(kp);
        q = q.max(kq);
    }
    resolve(p, q, mask)
}

fn check(row: &[f32], mask: [u32; 2], what: &str) {
    let want = feed_argmax_f32(row, mask);
    let n = row.len();
    let mut splits: Vec<Vec<usize>> = vec![vec![], vec![n / 2], vec![n / 3, 2 * n / 3]];
    splits.push(vec![1.min(n), n.saturating_sub(1)]);
    splits.push(vec![n / 4, n / 2, 3 * n / 4]);
    for cuts in splits {
        let mut cuts = cuts;
        cuts.retain(|&c| c > 0 && c < n);
        cuts.dedup();
        assert_eq!(
            sharded(row, &cuts, mask),
            want,
            "{what}: n={n} cuts={cuts:?} mask={mask:?}"
        );
    }
}

/// 2026-10-10: A value drawn from a small set, so ties are common.
fn tie_value(rng: &mut StdRng) -> f32 {
    const SET: [f32; 9] = [
        -1e31,
        PLAIN_FLOOR,
        -3.5,
        -0.0,
        0.0,
        1.25,
        7.0,
        f32::NEG_INFINITY,
        7.0,
    ];
    SET[rng.random_range(0..SET.len())]
}

#[test]
fn the_order_map_is_monotonic_and_folds_negative_zero() {
    let xs = [
        f32::NEG_INFINITY,
        -1e30,
        -2.0,
        -1e-30,
        0.0,
        1e-30,
        3.0,
        f32::INFINITY,
    ];
    for w in xs.windows(2) {
        assert!(order_bits(w[0]) < order_bits(w[1]), "{w:?}");
    }
    assert_eq!(order_bits(-0.0), order_bits(0.0));
}

#[test]
fn a_plain_key_round_trips_its_index() {
    for i in [0u32, 1, 511, 512, 1023, 1024, 1025, 51_647, 51_648, 154_855] {
        assert_eq!(plain_index(plain_key(1.0, i)), i);
    }
    assert_eq!(plain_index(0), 0);
    assert_eq!(plain_key(PLAIN_FLOOR, 3), 0);
    assert_eq!(plain_key(f32::NAN, 3), 0);
    assert_eq!(masked_key(f32::NAN, 3, [NO_ID, NO_ID]), 0);
    assert_ne!(masked_key(f32::NEG_INFINITY, 3, [NO_ID, NO_ID]), 0);
}

#[test]
fn ties_follow_the_kernel_tree_not_the_lowest_index() {
    // 2026-10-10: Threads 1 and 2 tie: the tree keeps the even thread (bit 0 decides first).
    let mut row = vec![0.0f32; 2 * BLOCK];
    row[1] = 5.0;
    row[2] = 5.0;
    assert_eq!(feed_argmax_f32(&row, [NO_ID, NO_ID]), 2);
    check(&row, [NO_ID, NO_ID], "threads 1 and 2");
    // 2026-10-10: Inside one thread's stride the first index wins.
    let mut row = vec![0.0f32; 3 * BLOCK];
    row[BLOCK + 7] = 5.0;
    row[2 * BLOCK + 7] = 5.0;
    check(&row, [NO_ID, NO_ID], "one stride");
    // 2026-10-10: A tie across a three-rank boundary.
    let mut row = vec![-1.0f32; 154_856];
    row[51_647] = 9.0;
    row[51_648] = 9.0;
    row[154_855] = 9.0;
    check(&row, [NO_ID, NO_ID], "rank boundary");
}

#[test]
fn rows_without_a_candidate_answer_as_the_kernel_does() {
    check(&[PLAIN_FLOOR; 70], [NO_ID, NO_ID], "all at the floor");
    check(&[f32::NEG_INFINITY; 70], [NO_ID, NO_ID], "all -inf");
    check(&[f32::NAN; 70], [NO_ID, NO_ID], "all NaN");
    // 2026-10-10: p is 0 (no candidate); a mask on 0 sends the answer to q.
    check(&[f32::NEG_INFINITY; 70], [0, NO_ID], "all -inf, mask on 0");
    check(&[f32::NAN; 70], [0, 5], "all NaN, mask on 0");
}

#[test]
fn a_masked_plain_argmax_takes_the_last_masked_maximum() {
    let mut row = vec![0.0f32; 3000];
    row[10] = 8.0;
    row[2000] = 6.0;
    row[2999] = 6.0;
    check(&row, [10, NO_ID], "mask hits p");
    check(&row, [10, 2999], "both ids masked");
    check(&row, [2000, 10], "mask order swapped");
}

#[test]
fn random_rows_with_dense_ties_match_on_every_split() {
    let mut rng = StdRng::seed_from_u64(0x05EE_DA26);
    for case in 0..4000 {
        let n = match case % 4 {
            0 => rng.random_range(1..40),
            1 => rng.random_range(1000..2100),
            2 => rng.random_range(3000..5000),
            _ => rng.random_range(1..9000),
        };
        let mut row: Vec<f32> = (0..n).map(|_| tie_value(&mut rng)).collect();
        if case % 7 == 0 {
            row[rng.random_range(0..n)] = f32::NAN;
        }
        let pick = |rng: &mut StdRng| match rng.random_range(0..4) {
            0 => NO_ID,
            1 => (n + 3) as u32,
            _ => rng.random_range(0..n) as u32,
        };
        let mask = [pick(&mut rng), pick(&mut rng)];
        check(&row, mask, "random");
    }
}

#[test]
fn random_bf16_rows_of_the_glm_vocab_match_the_three_rank_split() {
    let mut rng = StdRng::seed_from_u64(0xB16);
    for _ in 0..20 {
        let row: Vec<f32> = (0..154_856)
            .map(|_| bf16_to_f32(rng.random_range(0x3F00u16..0x4180)))
            .collect();
        let want = feed_argmax_f32(&row, [NO_ID, NO_ID]);
        let got = sharded(&row, &[51_648, 103_296], [NO_ID, NO_ID]);
        assert_eq!(got, want);
    }
}
