// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The tensor-parallel propose's exchange: every rank resolves the gathered blocks
//! the same way, any rank not ready cancels the propose on all of them, and rank 0's header
//! comes through with its overflow count.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

use super::super::tp::HEADER_INLINE;
use super::*;

fn header(positions: usize) -> Header {
    Header {
        width: 8,
        anchor: 42,
        position: 900,
        ctx_count: 899,
        ship_start: 899 - positions,
        positions: (0..positions as i32).collect(),
    }
}

fn gathered(blocks: &[[u32; EXCHANGE_WORDS]]) -> Vec<u32> {
    blocks.iter().flat_map(|b| b.iter().copied()).collect()
}

/// 2026-10-10: All ready: every rank gets rank 0's header (inline positions) and its overflow
/// count; the other ranks' header words are ignored.
#[test]
fn a_ready_exchange_gives_every_rank_the_root_header() {
    let h = header(3);
    let b = gathered(&[
        exchange_block(true, Some(&h)),
        exchange_block(true, None),
        exchange_block(true, None),
    ]);
    assert_eq!(resolve_exchange(&b, 3).unwrap(), (h, 0));
    let long = header(HEADER_INLINE + 9);
    let b = gathered(&[
        exchange_block(true, Some(&long)),
        exchange_block(true, None),
    ]);
    let (got, overflow) = resolve_exchange(&b, 2).unwrap();
    assert_eq!(overflow, 9);
    assert_eq!(got.positions, long.positions[..HEADER_INLINE].to_vec());
}

/// 2026-10-10: Any rank not ready, rank 0 included, cancels the propose with the same error on
/// every rank; a foreign status word counts as not ready.
#[test]
fn any_rank_not_ready_cancels_on_every_rank() {
    let h = header(3);
    for failed in 0..3 {
        let blocks: Vec<[u32; EXCHANGE_WORDS]> = (0..3)
            .map(|r| exchange_block(r != failed, (r == 0 && failed != 0).then_some(&h)))
            .collect();
        let e = resolve_exchange(&gathered(&blocks), 3)
            .unwrap_err()
            .to_string();
        assert!(e.contains(&format!("[{failed}]")), "{e}");
    }
    let mut b = gathered(&[exchange_block(true, Some(&h)), exchange_block(true, None)]);
    b[EXCHANGE_WORDS] = 7;
    assert!(resolve_exchange(&b, 2).is_err());
}

/// 2026-10-10: A ready rank 0 that sent no header, or a block count that is not the world's, is
/// an error.
#[test]
fn a_missing_header_or_a_short_gather_is_refused() {
    let b = gathered(&[exchange_block(true, None), exchange_block(true, None)]);
    assert!(resolve_exchange(&b, 2).is_err());
    let b = gathered(&[exchange_block(true, Some(&header(1)))]);
    assert!(resolve_exchange(&b, 2).is_err());
}
