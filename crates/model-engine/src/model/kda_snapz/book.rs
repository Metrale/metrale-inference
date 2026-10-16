// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: SNAPZ bookkeeping, pure: which slots have a deferred commit (their state is
//! still in their first snapshot slot), which batch positions of the next batched verify read
//! it there (the flag table), which deferred commits must land first, and which slots the last
//! verify armed for a deferral. The model (`kda_snapz.rs`) does the I/O.
//!
//! Owner: model-engine speculative decoding.
//! Invariants:
//! - A pending slot's plan is the eager commit that was skipped; it lands (is returned by
//!   `begin` or `flush_all`) before anything but a flagged batched verify reads that slot.
//! - The flag table sets position `i` only for a pending slot that sits at `i` alone and
//!   below the table ([`eligible`]); the layer gives exactly such a position its defer.

/// 2026-10-16: Whether batch position `i` of `slots` can read a deferred state: below the
/// flag table (`cap` positions) and the slot's only position in the batch. The GLM layer's
/// `defers_from` applies the same rule to the slots' snapshot pointers.
pub(crate) fn eligible(slots: &[usize], i: usize, cap: usize) -> bool {
    i < cap && slots.iter().filter(|&&s| s == slots[i]).count() == 1
}

/// 2026-10-16: What a batched verify needs before its launches: the deferred commits to land
/// first, in pending order, and the flag table to upload when it changed (`None`: the device
/// copy already holds it).
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Begin<P> {
    pub flush: Vec<P>,
    pub upload: Option<Vec<u32>>,
}

/// 2026-10-16: See the module header. `P` is a deferred commit's plan.
#[derive(Debug)]
pub(crate) struct Book<P> {
    cap: usize,
    pending: Vec<(usize, P)>,
    armed: Vec<usize>,
    mirror: Vec<u32>,
    synced: bool,
}

impl<P> Book<P> {
    pub(crate) fn new(cap: usize) -> Self {
        Self {
            cap,
            pending: Vec::new(),
            armed: Vec::new(),
            mirror: vec![0; cap],
            synced: false,
        }
    }

    pub(crate) fn is_idle(&self) -> bool {
        self.pending.is_empty() && self.armed.is_empty()
    }

    /// 2026-10-16: Before a batched verify over `slots`: a pending slot at an eligible
    /// position is flagged there; one in the batch at no eligible position lands now; one
    /// outside the batch stays pending (nothing in this verify reads it). Clears the arms of
    /// the previous verify.
    pub(crate) fn begin(&mut self, slots: &[usize]) -> Begin<P> {
        self.armed.clear();
        let mut flags = vec![0u32; self.cap];
        let mut flush = Vec::new();
        for (slot, plan) in std::mem::take(&mut self.pending) {
            match slots.iter().position(|&s| s == slot) {
                Some(i) if eligible(slots, i, self.cap) => {
                    flags[i] = 1;
                    self.pending.push((slot, plan));
                }
                Some(_) => flush.push(plan),
                None => self.pending.push((slot, plan)),
            }
        }
        let upload = (!self.synced || flags != self.mirror).then(|| {
            self.mirror.clone_from(&flags);
            flags
        });
        self.synced = true;
        Begin { flush, upload }
    }

    /// 2026-10-16: After that verify. `Ok`: its flagged slots read their deferred state and
    /// wrote the live one, so they are no longer pending, and every slot at an eligible
    /// position is armed. `Err`: nothing is armed and the device flag table is no longer
    /// trusted (the next `begin` uploads).
    pub(crate) fn end(&mut self, ok: bool, slots: &[usize]) {
        if !ok {
            self.armed.clear();
            self.synced = false;
            return;
        }
        self.pending.retain(|(s, _)| !slots.contains(s));
        self.armed = (0..slots.len())
            .filter(|&i| eligible(slots, i, self.cap))
            .map(|i| slots[i])
            .collect();
    }

    /// 2026-10-16: Consume `slot`'s arm: `true` when its commit may be deferred.
    pub(crate) fn take_armed(&mut self, slot: usize) -> bool {
        match self.armed.iter().position(|&s| s == slot) {
            Some(p) => {
                self.armed.swap_remove(p);
                true
            }
            None => false,
        }
    }

    /// 2026-10-16: Record `slot`'s skipped commit. Errors when one is already pending: its
    /// first snapshot then holds only the newer one, and the older state is lost.
    pub(crate) fn defer(&mut self, slot: usize, plan: P) -> Result<(), String> {
        if self.pending.iter().any(|(s, _)| *s == slot) {
            return Err(format!("SNAPZ: slot {slot} already has a deferred commit"));
        }
        self.pending.push((slot, plan));
        Ok(())
    }

    /// 2026-10-16: Every pending commit, to land now, in pending order; clears the arms.
    pub(crate) fn flush_all(&mut self) -> Vec<P> {
        self.armed.clear();
        self.pending.drain(..).map(|(_, p)| p).collect()
    }
}

#[cfg(test)]
#[path = "book_tests.rs"]
mod tests;
