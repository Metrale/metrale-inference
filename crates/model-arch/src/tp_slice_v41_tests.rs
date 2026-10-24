// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: Tensor-parallel byte slicing: the kept bytes are exactly the requested block, the
//! shares tile the total in order with the larger shares first, and bad requests are refused.
//!
//! Owner: model-arch, DeepSeek-V4.1.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn a_block_keeps_exactly_its_rows_and_columns() {
    // 2026-10-19: Byte (r, c) of a [5, 7] matrix is 10 * r + c.
    let src: Vec<u8> = (0..5u8)
        .flat_map(|r| (0..7u8).map(move |c| 10 * r + c))
        .collect();
    let got = slice2d(&src, 7, 1..4, 2..5).unwrap();
    assert_eq!(got, [12, 13, 14, 22, 23, 24, 32, 33, 34]);
    assert_eq!(slice2d(&src, 7, 0..5, 0..7).unwrap(), src);
    assert!(slice2d(&src, 7, 2..2, 0..7).unwrap().is_empty());
}

#[test]
fn a_ragged_source_or_an_out_of_range_block_is_refused() {
    let src = vec![0u8; 35];
    assert!(
        slice2d(&src, 6, 0..1, 0..1).is_err(),
        "35 is not whole rows of 6"
    );
    assert!(slice2d(&src, 7, 0..6, 0..7).is_err());
    assert!(slice2d(&src, 7, 0..5, 3..8).is_err());
    assert!(slice2d(&src, 0, 0..0, 0..0).is_err());
}

#[test]
fn shares_tile_the_total_in_order_with_the_larger_shares_first() {
    let heads: Vec<_> = (0..3).map(|r| rank_share(64, 8, 3, r).unwrap()).collect();
    assert_eq!(heads, [0..24, 24..48, 48..64]);
    let inter: Vec<_> = (0..3)
        .map(|r| rank_share(2304, 32, 3, r).unwrap())
        .collect();
    assert_eq!(inter, [0..768, 768..1536, 1536..2304]);
    for (total, unit, world) in [(96, 8, 5), (40, 4, 3), (12, 4, 4), (8, 8, 1)] {
        let shares: Vec<_> = (0..world)
            .map(|r| rank_share(total, unit, world, r).unwrap())
            .collect();
        assert_eq!(shares[0].start, 0);
        assert_eq!(shares[world - 1].end, total);
        for w in shares.windows(2) {
            assert_eq!(w[0].end, w[1].start);
            assert!(w[0].len() >= w[1].len());
            assert!(w[0].len() - w[1].len() <= unit);
        }
    }
}

#[test]
fn a_total_not_in_whole_units_or_a_rank_past_the_world_is_refused() {
    assert!(rank_share(100, 8, 3, 0).is_err());
    assert!(rank_share(96, 8, 3, 3).is_err());
    assert!(rank_share(96, 0, 3, 0).is_err());
}
