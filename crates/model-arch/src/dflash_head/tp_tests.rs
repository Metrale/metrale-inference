// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The tensor-parallel drafter's split and its propose header.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

use super::*;

/// 2026-10-10: The GLM-5.3 DFlash2 drafter (32 q / 8 KV heads, 12288 MLP, 154880 vocab) over
/// three ranks: whole GQA groups 12/12/8, MLP 4096 each, vocab 51648/51648/51584.
#[test]
fn the_glm_drafter_splits_in_whole_groups() {
    let p: Vec<TpPlan> = (0..3)
        .map(|r| plan(32, 8, 12288, 154880, 3, r).unwrap())
        .collect();
    assert_eq!(
        p.iter().map(|x| x.kv_heads).collect::<Vec<_>>(),
        [(0, 3), (3, 3), (6, 2)]
    );
    assert_eq!(
        p.iter().map(|x| x.q_heads).collect::<Vec<_>>(),
        [(0, 12), (12, 12), (24, 8)]
    );
    assert_eq!(
        p.iter().map(|x| x.inter).collect::<Vec<_>>(),
        [(0, 4096), (4096, 4096), (8192, 4096)]
    );
    assert_eq!(
        p.iter().map(|x| x.vocab).collect::<Vec<_>>(),
        [(0, 51648), (51648, 51648), (103296, 51584)]
    );
}

/// 2026-10-10: Over 1..=4 ranks and vocabularies that are not 64-row multiples, every range
/// is contiguous, in rank order and covers its dimension exactly once.
#[test]
fn every_dimension_is_covered_once_in_rank_order() {
    for world in 1..=4 {
        for vocab in [154880usize, 151937, 32768 + 5] {
            let p: Vec<TpPlan> = (0..world)
                .map(|r| plan(32, 8, 12288, vocab, world, r).unwrap())
                .collect();
            let check = |f: &dyn Fn(&TpPlan) -> (usize, usize), total: usize| {
                let mut next = 0;
                for x in &p {
                    let (s, l) = f(x);
                    assert_eq!(s, next, "world {world}: a gap or an overlap");
                    assert!(l > 0);
                    next = s + l;
                }
                assert_eq!(next, total, "world {world}");
            };
            check(&|x| x.q_heads, 32);
            check(&|x| x.kv_heads, 8);
            check(&|x| x.inter, 12288);
            check(&|x| x.vocab, vocab);
        }
    }
}

#[test]
fn an_impossible_split_is_refused() {
    assert!(
        plan(30, 8, 12288, 154880, 3, 0).is_err(),
        "heads that do not group"
    );
    assert!(
        plan(32, 2, 12288, 154880, 3, 0).is_err(),
        "fewer KV heads than ranks"
    );
    assert!(
        plan(32, 8, 12288 + 64, 154880, 3, 0).is_err(),
        "MLP off its unit"
    );
    assert!(
        plan(32, 8, 12288, 154880, 3, 3).is_err(),
        "rank past the world"
    );
}

/// 2026-10-10: A header round-trips with its positions inline up to HEADER_INLINE; past it
/// the rest is the overflow; the abort decodes to None and a foreign tag is an error.
#[test]
fn the_header_round_trips_and_spills_long_tails() {
    let short = Header {
        width: 3,
        anchor: 154_000,
        position: 9000,
        ctx_count: 8999,
        ship_start: 8996,
        positions: vec![8995, 8996, 8997],
    };
    let w = short.encode();
    assert_eq!(Header::decode(&w).unwrap(), Some(short.clone()));
    assert_eq!(Header::overflow_len(&w), 0);
    assert!(short.overflow().is_empty());

    let long = Header {
        positions: (0..2048).collect(),
        ..short
    };
    let w = long.encode();
    let got = Header::decode(&w).unwrap().unwrap();
    assert_eq!(got.positions, (0..HEADER_INLINE as i32).collect::<Vec<_>>());
    assert_eq!(Header::overflow_len(&w), 2048 - HEADER_INLINE);
    assert_eq!(long.overflow(), &long.positions[HEADER_INLINE..]);

    assert_eq!(Header::decode(&Header::abort_words()).unwrap(), None);
    let mut bad = w;
    bad[0] = 7;
    assert!(Header::decode(&bad).is_err());
    let mut empty = short.encode();
    empty[1] = 0;
    assert!(Header::decode(&empty).is_err());
}

/// 2026-10-10: Validation rejects a decoded width or context span before that rank can reach
/// the overflow/context-row broadcast sequence.
#[test]
fn decoded_header_bounds_are_checked_before_collectives_continue() {
    let good = Header {
        width: 3,
        anchor: 7,
        position: 12,
        ctx_count: 10,
        ship_start: 8,
        positions: vec![7, 8],
    };
    assert!(super::super::tp_propose::validate_tp_header(&good, 2, 8, 16).is_ok());

    let bad_width = Header {
        width: 9,
        ..good.clone()
    };
    assert!(super::super::tp_propose::validate_tp_header(&bad_width, 2, 8, 16).is_err());

    let bad_span = Header {
        ship_start: 15,
        ..good
    };
    assert!(super::super::tp_propose::validate_tp_header(&bad_span, 2, 8, 16).is_err());
}
