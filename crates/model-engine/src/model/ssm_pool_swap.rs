// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The SSM pool's commit swap (`ssm_reserve::commit_swap`): the home addresses of
//! a slot's live state and intermediate 0, the rule that a slot may swap, and the swap itself.
//! The resolving accessors (`h_state`, `conv_state`, `h_intermediate`, `conv_intermediate` in
//! `ssm_pool_slots.rs`) read the role bit through these.
//!
//! Owner: model-engine SSM state pool.
//! Invariants:
//! - Only a slot with its own intermediate 0 of both kinds swaps (snapshot mode, MTP on, below
//!   `mtp_slots`), so a swapped slot never resolves onto the shared MTP dummy.
//! - A swap moves every SSM layer of the slot at once: the per-layer addresses of one role stay
//!   one stride apart, so the copy plans built from them still collapse to one pitched copy.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::ssm_reserve::{SsmRollbackMode, resolve_pair};

use super::ssm_pool::SsmStatePool;

impl SsmStatePool {
    /// 2026-10-21: Slot `slot`'s home live h buffer in SSM layer `i`.
    pub(super) fn h_home(&self, i: usize, slot: usize) -> DevicePtr {
        self.h_state_pools[i].offset(slot * self.h_stored_bytes)
    }

    /// 2026-10-21: Slot `slot`'s home live conv buffer in SSM layer `i`.
    pub(super) fn conv_home(&self, i: usize, slot: usize) -> DevicePtr {
        self.conv_state_pools[i].offset(slot * self.conv_bytes)
    }

    /// 2026-10-21: Home address of H intermediate `t` of `slot` (after the MTP slot mapping).
    /// `t` must be below that slot's `h_inter_counts` entry; debug builds assert it.
    pub(super) fn h_inter_home(&self, i: usize, slot: usize, t: usize) -> DevicePtr {
        let slot = self.mtp_slot(slot);
        debug_assert!(
            t < self.h_inter_counts[slot],
            "h_intermediate: token_idx {t} >= slot {slot}'s tiered capacity {}",
            self.h_inter_counts[slot],
        );
        self.h_intermediate_pools[i].offset((self.h_inter_offsets[slot] + t) * self.h_stored_bytes)
    }

    /// 2026-10-21: Home address of conv intermediate `t` of `slot` (after the MTP slot mapping).
    pub(super) fn conv_inter_home(&self, i: usize, slot: usize, t: usize) -> DevicePtr {
        let slot = self.mtp_slot(slot);
        self.conv_intermediate_pools[i]
            .offset((slot * self.num_intermediates + t) * self.conv_bytes)
    }

    /// 2026-10-21: `(live, intermediate 0)` of `slot`'s h in layer `i`, by its role bit; `None`
    /// when the slot is not swapped (the home addresses then hold, and intermediate 0 may not
    /// exist).
    pub(super) fn h_swapped_pair(&self, i: usize, slot: usize) -> Option<(DevicePtr, DevicePtr)> {
        self.swaps
            .is_swapped(slot)
            .then(|| resolve_pair(self.h_home(i, slot), self.h_inter_home(i, slot, 0), true))
    }

    /// 2026-10-21: [`Self::h_swapped_pair`] for conv.
    pub(super) fn conv_swapped_pair(
        &self,
        i: usize,
        slot: usize,
    ) -> Option<(DevicePtr, DevicePtr)> {
        self.swaps.is_swapped(slot).then(|| {
            resolve_pair(
                self.conv_home(i, slot),
                self.conv_inter_home(i, slot, 0),
                true,
            )
        })
    }

    /// 2026-10-21: Whether `slot` has its own intermediate 0 of both kinds to swap with.
    pub(crate) fn can_swap(&self, slot: usize) -> bool {
        self.has_mtp
            && self.rollback_mode == SsmRollbackMode::Snapshot
            && self.num_ssm_layers > 0
            && slot < self.mtp_slots
            && self.h_inter_count(slot) >= 1
            && self.conv_inter_count() >= 1
    }

    /// 2026-10-21: Whether `slot`'s live state sits in its intermediate-0 buffers.
    pub(crate) fn is_swapped(&self, slot: usize) -> bool {
        self.swaps.is_swapped(slot)
    }

    /// 2026-10-21: Exchange `slot`'s live and intermediate-0 roles in every layer. The caller
    /// exchanges the sequence's cached pointers with it (`SsmLayerState::swap_live_with_inter0`).
    pub(super) fn swap_roles(&self, slot: usize) -> Result<()> {
        if !self.can_swap(slot) || !self.swaps.toggle(slot) {
            bail!("SSM commit swap: slot {slot} has no intermediate 0 of its own to swap with");
        }
        Ok(())
    }
}

