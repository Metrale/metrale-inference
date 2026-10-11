// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: Process-wide counts of the host round trips the CUDA backend makes: every
//! blocking stream sync (`synchronize`, and the sync inside `copy_h2d` and the blocking D2H
//! copies), every host-to-device and device-to-host copy. Always on (one relaxed add per
//! event); readers difference two [`snapshot`]s, e.g. the MTP step timing's per-step counts.
//!
//! Owner: gpu-runtime.
//! Invariants: the counters only grow.

use std::sync::atomic::{AtomicU64, Ordering};

static SYNCS: AtomicU64 = AtomicU64::new(0);
static H2D: AtomicU64 = AtomicU64::new(0);
static D2H: AtomicU64 = AtomicU64::new(0);

/// 2026-10-11: The counters at one instant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostSyncCounts {
    pub syncs: u64,
    pub h2d: u64,
    pub d2h: u64,
}

impl HostSyncCounts {
    /// 2026-10-11: The counts since `earlier`.
    pub fn since(self, earlier: Self) -> Self {
        Self {
            syncs: self.syncs.saturating_sub(earlier.syncs),
            h2d: self.h2d.saturating_sub(earlier.h2d),
            d2h: self.d2h.saturating_sub(earlier.d2h),
        }
    }
}

pub fn snapshot() -> HostSyncCounts {
    HostSyncCounts {
        syncs: SYNCS.load(Ordering::Relaxed),
        h2d: H2D.load(Ordering::Relaxed),
        d2h: D2H.load(Ordering::Relaxed),
    }
}

#[inline]
pub fn sync() {
    SYNCS.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub fn h2d() {
    H2D.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub fn d2h() {
    D2H.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-11: Each event moves its own counter, and `since` differences two snapshots.
    #[test]
    fn events_move_their_own_counter() {
        let a = snapshot();
        sync();
        h2d();
        h2d();
        d2h();
        let d = snapshot().since(a);
        // 2026-10-11: Other tests in the process may add events concurrently, so the counts
        // are lower bounds.
        assert!(d.syncs >= 1 && d.h2d >= 2 && d.d2h >= 1, "{d:?}");
        assert_eq!(a.since(snapshot()), HostSyncCounts::default());
    }
}
