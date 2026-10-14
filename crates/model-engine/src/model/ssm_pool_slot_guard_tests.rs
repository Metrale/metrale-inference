// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for `SlotGuard`: release on drop, `take`, and migration.
//!
//! Owner: model-engine SSM state pool.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: A pool with no device memory: every pointer vector is empty and no
/// SSM layer exists. `claim_slot`, `release_slot` and the guard touch only
/// `free_slots`, so these tests need no GPU.
fn bare_pool(max_slots: usize) -> Arc<SsmStatePool> {
    Arc::new(SsmStatePool {
        owned_allocations: Vec::new(),
        h_state_pools: Vec::new(),
        conv_state_pools: Vec::new(),
        h_intermediate_pools: Vec::new(),
        conv_intermediate_pools: Vec::new(),
        h_checkpoint_pools: Vec::new(),
        conv_checkpoint_pools: Vec::new(),
        h_bytes: 0,
        h_stored_bytes: 0,
        h_prefill_stage_pool: None,
        conv_bytes: 0,
        max_slots,
        mtp_slots: 0,
        num_ssm_layers: 0,
        has_mtp: false,
        num_intermediates: 0,
        h_inter_counts: Vec::new(),
        h_inter_offsets: Vec::new(),
        rollback_mode: metrale_model_layers::ssm_reserve::SsmRollbackMode::Snapshot,
        replay_input_rings: Vec::new(),
        free_slots: Mutex::new((0..max_slots).rev().collect()),
    })
}

fn free_count(pool: &SsmStatePool) -> usize {
    pool.free_slots.lock().len()
}

#[test]
fn guard_releases_on_drop() {
    let pool = bare_pool(2);
    let claimed;
    {
        let g = pool.claim_guarded().unwrap();
        // 2026-09-25: The free list is `(0..max).rev()`, so `pop()` returns 0 first.
        claimed = g.idx().expect("guard owns a slot");
        assert_eq!(claimed, 0);
        assert_eq!(free_count(&pool), 1);
    }
    assert_eq!(
        free_count(&pool),
        2,
        "drop must return the slot exactly once"
    );
    assert!(pool.free_slots.lock().contains(&claimed));
}

#[test]
fn take_neutralizes_drop_no_double_release() {
    let pool = bare_pool(2);
    let mut g = pool.claim_guarded().unwrap();
    let idx = g.take().expect("guard owns a slot");
    pool.release_slot(idx).unwrap();
    assert_eq!(free_count(&pool), 2);
    drop(g);
    assert_eq!(
        free_count(&pool),
        2,
        "take() must make Drop a no-op (no double-release)"
    );
}

#[test]
fn migration_releases_old_once_then_owns_new() {
    // 2026-09-25: Two claimed slots, so the migration target is owned by another
    // guard rather than sitting on the free list.
    let pool = bare_pool(2);
    let mut survivor = pool.claim_guarded().unwrap();
    let donor = pool.claim_guarded().unwrap();
    assert_eq!(free_count(&pool), 0);
    let donor_slot = donor.idx().unwrap();

    // 2026-09-25: The survivor releases its old slot and migrates onto the donor's.
    let old = survivor.take().unwrap();
    pool.release_slot(old).unwrap();
    // 2026-09-25: The donor gives up its slot without releasing it.
    let mut donor = donor;
    let _ = donor.take();
    drop(donor);
    survivor.migrate(donor_slot);
    assert_eq!(survivor.idx(), Some(donor_slot));

    let final_idx = survivor.take().unwrap();
    pool.release_slot(final_idx).unwrap();
    drop(survivor);

    let free = pool.free_slots.lock();
    let mut sorted = free.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, vec![0, 1], "both slots free exactly once, no dupes");
}

/// 2026-10-14: DEF-1. A second release of a slot is refused and leaves the free list
/// as it was; under the old `debug_assert!` a release build pushed the slot twice, and
/// two later claims both received it.
#[test]
fn a_double_release_is_refused_and_never_hands_one_slot_to_two_claims() {
    let pool = bare_pool(2);
    let mut g = pool.claim_guarded().unwrap();
    let idx = g.take().unwrap();
    pool.release_slot(idx).unwrap();
    let err = pool
        .release_slot(idx)
        .expect_err("second release of the same slot");
    assert!(format!("{err:#}").contains("already free"), "{err:#}");
    let a = pool.claim_slot().unwrap();
    let b = pool.claim_slot().unwrap();
    assert_ne!(
        a, b,
        "two claims after a refused double release must differ"
    );
    assert!(
        pool.claim_slot().is_err(),
        "a 2-slot pool has exactly two claimable slots"
    );
}

/// 2026-10-14: The dummy slot (`max_slots`) and anything above it never enter the free list.
#[test]
fn releasing_a_slot_outside_the_pool_is_refused() {
    let pool = bare_pool(2);
    assert!(pool.release_slot(2).is_err());
    assert_eq!(free_count(&pool), 2);
}

/// 2026-10-14: `claim_specific` takes only a free slot; a slot a guard owns is refused,
/// which is what `compact_sequence` relies on to refuse an aliasing move.
#[test]
fn claim_specific_refuses_an_owned_slot() {
    let pool = bare_pool(2);
    let owner = pool.claim_guarded().unwrap();
    let owned = owner.idx().unwrap();
    assert!(!pool.claim_specific(owned));
    let other = 1 - owned;
    assert!(pool.claim_specific(other));
    assert!(
        !pool.claim_specific(other),
        "a claimed slot is no longer free"
    );
    assert_eq!(free_count(&pool), 0);
}
