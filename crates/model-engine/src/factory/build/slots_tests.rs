// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: `balance_slots` in its three regimes (the pool never binds, it binds exactly at
//! the ceiling, it binds below it), on the dense default tier's measured sizes, and its refusals;
//! and the `--max-batch-size` request's spelling.
//!
//! Owner: metrale-model-engine.
//! Invariants: none beyond the types.

use super::*;

/// 2026-10-01: The dense Qwen3.8-27B default tier as served on GB10 (`--weight-quantization
/// declared`, f32 h, fp8 KV, `--max-seq-len 2048`): 151.5 MB of SSM state per slot, KV blocks of
/// 0.5 MB (16 attention layers x 32 KiB), and 19,947 MB that the per-slot state and the KV pool
/// share once everything else is placed (serve logs, 2026-10-01). In MB-scaled integers.
fn dense_default_tier(slots: usize) -> Result<usize> {
    let shared_mb_x10 = 199_470usize;
    let state = (slots + 1) * 1_515;
    anyhow::ensure!(state < shared_mb_x10, "no memory left for KV");
    Ok((shared_mb_x10 - state) / 5)
}

/// 2026-10-01: 2048 tokens of 16-token blocks, plus the one spare the admission reserves.
const FULL_LENGTH_BLOCKS: usize = 129;

#[test]
fn the_pool_never_binds_and_every_slot_is_kept() {
    // 2026-10-01: Plenty of KV at every count: the ceiling is the answer.
    let b = balance_slots(8, FULL_LENGTH_BLOCKS, |_| Ok(100_000)).unwrap();
    assert_eq!((b.slots, b.admitted), (8, 8));
}

#[test]
fn the_pool_binds_exactly_at_the_ceiling() {
    // 2026-10-01: Each slot costs one sequence's blocks; at 10 slots the pool holds exactly 10.
    let kv = |s: usize| Ok((20 - s) * FULL_LENGTH_BLOCKS);
    let b = balance_slots(10, FULL_LENGTH_BLOCKS, kv).unwrap();
    assert_eq!((b.slots, b.admitted), (10, 10));
    // 2026-10-01: One more allowed slot would admit fewer (11 slots hold only 9).
    let b = balance_slots(11, FULL_LENGTH_BLOCKS, kv).unwrap();
    assert_eq!((b.slots, b.admitted), (10, 10));
}

#[test]
fn the_dense_default_tier_balances_at_91_slots() {
    let b = balance_slots(128, FULL_LENGTH_BLOCKS, dense_default_tier).unwrap();
    assert_eq!(b.slots, 91);
    assert_eq!(b.admitted, 91);
    assert!(b.kv_blocks >= 91 * FULL_LENGTH_BLOCKS);
    // 2026-10-01: Today's 128 slots admit 6 full-length sequences.
    let at_128 = dense_default_tier(128).unwrap();
    assert_eq!(at_128 / FULL_LENGTH_BLOCKS, 6);
    // 2026-10-01: Neither neighbour admits as many.
    for s in [90, 92] {
        let kv = dense_default_tier(s).unwrap();
        assert!(s.min(kv / FULL_LENGTH_BLOCKS) < 91, "{s} slots");
    }
}

#[test]
fn a_tie_keeps_the_smaller_count_which_its_pool_holds() {
    // 2026-10-01: The pool holds 5 full-length sequences whatever the count: 5 slots, not 32.
    let b = balance_slots(32, FULL_LENGTH_BLOCKS, |_| Ok(5 * FULL_LENGTH_BLOCKS)).unwrap();
    assert_eq!((b.slots, b.admitted), (5, 5));
    // 2026-10-01: The GB10 boot (2026-10-01): each slot costs 303 blocks (151.5 MB of state), and
    // 91 slots leave 11,624, so 90 and 91 slots both admit 90 full-length sequences; 91 would
    // boot overcommitted by one.
    let kv = |s: usize| Ok(39_197 - s * 303);
    let b = balance_slots(128, FULL_LENGTH_BLOCKS, kv).unwrap();
    assert_eq!((b.slots, b.admitted), (90, 90));
    assert!(b.kv_blocks / FULL_LENGTH_BLOCKS >= b.slots);
}

#[test]
fn the_chosen_count_always_fits_its_pool_at_full_length() {
    // 2026-10-01: Over pools that shrink with the slot count at different rates, the chosen
    // count's pool holds that many full-length sequences.
    for per_slot in [0usize, 1, 37, 129, 303, 1_000] {
        for base in [129usize, 5_000, 20_000] {
            let kv = |s: usize| {
                anyhow::ensure!(s * per_slot < base, "none left");
                Ok(base - s * per_slot)
            };
            let Ok(b) = balance_slots(128, FULL_LENGTH_BLOCKS, kv) else {
                // 2026-10-01: Refused only when not even one slot's pool holds a sequence.
                assert!(base.saturating_sub(per_slot) < FULL_LENGTH_BLOCKS);
                continue;
            };
            assert!(
                b.kv_blocks / FULL_LENGTH_BLOCKS >= b.slots,
                "per_slot {per_slot}, base {base}: {b:?}"
            );
        }
    }
}

#[test]
fn counts_that_leave_no_kv_are_skipped_and_none_is_an_error() {
    let b = balance_slots(128, FULL_LENGTH_BLOCKS, |s| {
        anyhow::ensure!(s <= 3, "too many");
        Ok(10 * FULL_LENGTH_BLOCKS)
    })
    .unwrap();
    assert_eq!((b.slots, b.admitted), (3, 3));
    assert!(balance_slots(4, FULL_LENGTH_BLOCKS, |_| anyhow::bail!("never")).is_err());
    // 2026-10-01: A pool too small for one full-length sequence at any count is refused, not
    // built at one slot that overcommits.
    let err = balance_slots(4, FULL_LENGTH_BLOCKS, |_| Ok(FULL_LENGTH_BLOCKS - 1)).unwrap_err();
    assert!(err.to_string().contains("lower --max-seq-len"), "{err}");
    assert!(balance_slots(0, FULL_LENGTH_BLOCKS, |_| Ok(1)).is_err());
}

#[test]
fn the_request_spells_a_count_or_auto() {
    assert_eq!("auto".parse::<SlotRequest>(), Ok(SlotRequest::Auto));
    assert_eq!("128".parse::<SlotRequest>(), Ok(SlotRequest::Count(128)));
    assert!("Auto".parse::<SlotRequest>().is_err());
    assert!("-1".parse::<SlotRequest>().is_err());
    assert_eq!(SlotRequest::Auto.to_string(), "auto");
    assert_eq!(SlotRequest::Count(8).to_string(), "8");
    assert_eq!(SlotRequest::Count(8).ceiling(), 8);
    assert_eq!(SlotRequest::Auto.ceiling(), AUTO_MAX_SLOTS);
    assert_eq!(AUTO_MAX_SLOTS, 128);
}
