// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Commit a rejected draft by swapping two state buffers instead of copying one over
//! the other (`METRALE_SSM_COMMIT_SWAP`, default on, `0` off).
//!
//! A K-row verify steps a slot's live state in place and writes the state after row 0 to the
//! slot's intermediate 0. A commit of one accepted row (the draft rejected) used to copy
//! intermediate 0 over the live state: at GLM-5.3-Flash TP=3, 34 layers of 1.4 MB h plus the
//! conv window, ~54 MB per rejected step. Both buffers are the slot's own and have the same
//! width, so the commit can instead exchange their roles: intermediate 0 becomes the live
//! state and the old live buffer becomes intermediate 0, which the next verify overwrites.
//! No byte moves and no arithmetic changes, so every later kernel reads the bits the copy
//! would have left.
//!
//! [`SlotSwaps`] holds the per-slot role bit. Every address of a live state or of intermediate
//! 0 must go through [`resolve_pair`] with that bit; the pool's accessors do, and a sequence's
//! cached pointers are exchanged with the bit (`SsmLayerState::swap_live_with_inter0`). A
//! CUDA graph bakes addresses, so a graph keyed by slot is keyed by the bit too.
//!
//! The arm bit marks a slot whose last verify was a single-sequence one (its graphs keyed by
//! the role bit); only an armed commit swaps, so a multi-sequence verify keeps copying.
//!
//! Owner: model-layers (SSM reserve).
//! Invariants:
//! - A slot outside the table is never swapped and never armed.
//! - `clear` resets both bits; a released slot holds no state worth keeping, so clearing it
//!   moves nothing.

use std::sync::atomic::{AtomicBool, Ordering};

use metrale_gpu_runtime::gpu::DevicePtr;

/// 2026-10-21: The per-slot role bit (live state in the home buffer or in intermediate 0) and
/// the per-slot arm bit, for `slots` slots.
#[derive(Debug, Default)]
pub struct SlotSwaps {
    swapped: Vec<AtomicBool>,
    armed: Vec<AtomicBool>,
}

impl SlotSwaps {
    /// 2026-10-21: `slots` slots, none swapped, none armed.
    pub fn new(slots: usize) -> Self {
        Self {
            swapped: (0..slots).map(|_| AtomicBool::new(false)).collect(),
            armed: (0..slots).map(|_| AtomicBool::new(false)).collect(),
        }
    }

    /// 2026-10-21: Whether `slot`'s live state sits in its intermediate-0 buffer.
    pub fn is_swapped(&self, slot: usize) -> bool {
        self.swapped
            .get(slot)
            .is_some_and(|b| b.load(Ordering::Acquire))
    }

    /// 2026-10-21: Exchange `slot`'s roles. `false` (nothing changed) for a slot outside the
    /// table.
    pub fn toggle(&self, slot: usize) -> bool {
        match self.swapped.get(slot) {
            Some(b) => {
                b.fetch_xor(true, Ordering::AcqRel);
                true
            }
            None => false,
        }
    }

    /// 2026-10-21: Mark `slot`'s pending commit as one that may swap.
    pub fn arm(&self, slot: usize) {
        if let Some(b) = self.armed.get(slot) {
            b.store(true, Ordering::Release);
        }
    }

    /// 2026-10-21: Whether `slot` was armed, disarming it.
    pub fn take_arm(&self, slot: usize) -> bool {
        self.armed
            .get(slot)
            .is_some_and(|b| b.swap(false, Ordering::AcqRel))
    }

    /// 2026-10-21: Back to the home layout, disarmed (a released slot).
    pub fn clear(&self, slot: usize) {
        if let Some(b) = self.swapped.get(slot) {
            b.store(false, Ordering::Release);
        }
        if let Some(b) = self.armed.get(slot) {
            b.store(false, Ordering::Release);
        }
    }
}

/// 2026-10-21: `(live, intermediate 0)` of a slot whose home live buffer is `home` and whose
/// home intermediate-0 buffer is `inter0`.
pub fn resolve_pair(home: DevicePtr, inter0: DevicePtr, swapped: bool) -> (DevicePtr, DevicePtr) {
    if swapped {
        (inter0, home)
    } else {
        (home, inter0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-21: Swapped exchanges the two addresses, unswapped keeps them; two toggles are
    /// the identity.
    #[test]
    fn a_swap_exchanges_live_and_intermediate_zero() {
        let (h, i) = (DevicePtr(0x1000), DevicePtr(0x9000));
        assert_eq!(resolve_pair(h, i, false), (h, i));
        assert_eq!(resolve_pair(h, i, true), (i, h));
        let s = SlotSwaps::new(3);
        assert!(!s.is_swapped(1));
        assert!(s.toggle(1));
        assert!(s.is_swapped(1));
        assert!(!s.is_swapped(0) && !s.is_swapped(2), "only slot 1 moved");
        assert!(s.toggle(1));
        assert!(!s.is_swapped(1));
    }

    /// 2026-10-21: A slot past the table is never swapped or armed, and toggling it reports
    /// that nothing changed.
    #[test]
    fn a_slot_outside_the_table_never_swaps() {
        let s = SlotSwaps::new(2);
        assert!(!s.toggle(2));
        assert!(!s.is_swapped(2));
        s.arm(2);
        assert!(!s.take_arm(2));
    }

    /// 2026-10-21: An arm is consumed once; `clear` resets the role and the arm.
    #[test]
    fn an_arm_is_taken_once_and_clear_resets_everything() {
        let s = SlotSwaps::new(2);
        assert!(!s.take_arm(0));
        s.arm(0);
        assert!(s.take_arm(0));
        assert!(!s.take_arm(0), "taken");
        s.arm(1);
        s.toggle(1);
        s.clear(1);
        assert!(!s.is_swapped(1));
        assert!(!s.take_arm(1));
    }
}
