// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The pool's commit-swap addressing (`ssm_pool_swap.rs`) on a pool of made-up
//! addresses: no device memory is touched.

use parking_lot::Mutex;

use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::ssm_reserve::{SlotSwaps, SsmRollbackMode};

use super::super::ssm_batched_copy::{StateCopy, copy_plan_as_strided_run};
use super::SsmStatePool;

const LAYERS: usize = 3;
const SLOTS: usize = 4;
const MTP_SLOTS: usize = 2;
const NI: usize = 2;
const H: usize = 0x100;
const C: usize = 0x10;

/// 2026-10-21: Every family one contiguous block, layer stride = the family's per-layer size, as
/// `alloc_layer_pools` lays them out. `MTP_SLOTS` slots carry `NI - 1` h intermediates.
fn pool(mode: SsmRollbackMode) -> SsmStatePool {
    let fam = |base: u64, per_layer: usize| -> Vec<DevicePtr> {
        (0..LAYERS)
            .map(|l| DevicePtr(base + (l * per_layer) as u64))
            .collect()
    };
    let counts: Vec<usize> = vec![NI - 1; MTP_SLOTS + 1];
    let offsets: Vec<usize> = (0..=MTP_SLOTS + 1).map(|s| s * (NI - 1)).collect();
    SsmStatePool {
        owned_allocations: Vec::new(),
        h_state_pools: fam(0x10_0000, (SLOTS + 1) * H),
        conv_state_pools: fam(0x20_0000, (SLOTS + 1) * C),
        h_intermediate_pools: fam(0x30_0000, (MTP_SLOTS + 1) * (NI - 1) * H),
        conv_intermediate_pools: fam(0x40_0000, (MTP_SLOTS + 1) * NI * C),
        h_checkpoint_pools: fam(0x50_0000, (MTP_SLOTS + 1) * H),
        conv_checkpoint_pools: fam(0x60_0000, (MTP_SLOTS + 1) * C),
        h_bytes: H,
        h_stored_bytes: H,
        h_prefill_stage_pool: None,
        conv_bytes: C,
        max_slots: SLOTS,
        mtp_slots: MTP_SLOTS,
        num_ssm_layers: LAYERS,
        has_mtp: true,
        num_intermediates: NI,
        h_inter_counts: counts,
        h_inter_offsets: offsets,
        rollback_mode: mode,
        replay_input_rings: Vec::new(),
        replay_slot_bytes: 0,
        free_slots: Mutex::new((0..SLOTS).rev().collect()),
        swaps: SlotSwaps::new(SLOTS),
    }
}

type Addrs = Vec<(DevicePtr, DevicePtr, DevicePtr, DevicePtr, DevicePtr)>;

/// 2026-10-21: Per layer: live h, live conv, h inter 0, conv inter 0, conv inter 1.
fn addrs(p: &SsmStatePool, slot: usize) -> Addrs {
    (0..LAYERS)
        .map(|i| {
            (
                p.h_state(i, slot),
                p.conv_state(i, slot),
                p.h_intermediate(i, slot, 0),
                p.conv_intermediate(i, slot, 0),
                p.conv_intermediate(i, slot, 1),
            )
        })
        .collect()
}

/// 2026-10-21: A swap exchanges live with intermediate 0 for h and conv in every layer, leaves
/// intermediate 1, the checkpoints and the other slots alone, and a second swap is the identity.
#[test]
fn a_swap_exchanges_live_and_intermediate_zero_in_every_layer() {
    let p = pool(SsmRollbackMode::Snapshot);
    let home = addrs(&p, 1);
    let other = addrs(&p, 0);
    let ck = p.h_checkpoint(2, 1);
    p.swap_roles(1).unwrap();
    assert!(p.is_swapped(1));
    let sw = addrs(&p, 1);
    for (a, b) in home.iter().zip(&sw) {
        assert_eq!((b.0, b.1, b.2, b.3, b.4), (a.2, a.3, a.0, a.1, a.4));
    }
    assert_eq!(addrs(&p, 0), other, "slot 0 untouched");
    assert_eq!(p.h_checkpoint(2, 1), ck);
    p.swap_roles(1).unwrap();
    assert_eq!(addrs(&p, 1), home);
}

/// 2026-10-21: The commit and the device step issue a plan of one blob per layer as one pitched
/// copy; a swapped slot's resolved addresses keep one stride per role, so it still collapses.
#[test]
fn swapped_addresses_still_form_one_pitched_copy_per_family() {
    let p = pool(SsmRollbackMode::Snapshot);
    p.swap_roles(0).unwrap();
    let h: Vec<StateCopy> = (0..LAYERS)
        .map(|i| StateCopy {
            src: p.h_intermediate(i, 0, 0),
            dst: p.h_state(i, 0),
            bytes: H,
        })
        .collect();
    let run = copy_plan_as_strided_run(&h).expect("collapses");
    assert_eq!(run.height, LAYERS);
    assert_eq!(run.src, p.h_home(0, 0));
}

/// 2026-10-21: Only a covered slot of a snapshot-mode MTP pool swaps; the dummy, an uncovered
/// slot and a replay-mode pool refuse and stay home.
#[test]
fn only_a_covered_snapshot_slot_swaps() {
    let p = pool(SsmRollbackMode::Snapshot);
    for slot in [MTP_SLOTS, SLOTS - 1, SLOTS] {
        assert!(!p.can_swap(slot), "slot {slot}");
        assert!(p.swap_roles(slot).is_err(), "slot {slot}");
        assert!(!p.is_swapped(slot));
    }
    let r = pool(SsmRollbackMode::Replay);
    assert!(!r.can_swap(0));
    assert!(r.swap_roles(0).is_err());
}

/// 2026-10-21: Releasing a swapped slot returns it to the home layout (its bytes are the next
/// owner's), so a fresh sequence gets the home addresses.
#[test]
fn a_released_slot_is_home_again() {
    let p = pool(SsmRollbackMode::Snapshot);
    let s = p.claim_slot().unwrap();
    let home = addrs(&p, s);
    p.swap_roles(s).unwrap();
    p.swaps.arm(s);
    assert_ne!(addrs(&p, s), home);
    p.release_slot(s);
    assert_eq!(addrs(&p, s), home);
    assert!(!p.swaps.take_arm(s), "the arm is cleared too");
}
