// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: `PinnedMetaStaging` itself, plus the `Send`/`Sync` safety assertions for
//! `TransformerModel` that justify its single `UnsafeCell` field. Split out of `types.rs`
//! to keep that file under the 500-line cap; no behavior changes.
//!
//! Owner: model-engine.

use super::TransformerModel;

/// 2026-09-25: Pinned host staging buffer plus reusable metadata `Vec`s.
pub(crate) struct PinnedMetaStaging {
    /// 2026-09-25: Page-locked host buffer from `alloc_host_pinned`.
    pub(in crate::model) ptr: *mut u8,
    /// 2026-09-25: Size of `ptr`'s region in bytes.
    pub(in crate::model) bytes: usize,
    pub(in crate::model) positions: Vec<u32>,
    pub(in crate::model) positions_h: Vec<u32>,
    pub(in crate::model) positions_w: Vec<u32>,
    pub(in crate::model) slots: Vec<i64>,
}

// 2026-09-25: SAFETY: TransformerModel is constructed on one thread and then
// moved to the scheduler thread as a `Box<dyn Model>`; every later call
// (prefill, decode, verify) is made from that thread. The `Model` trait
// requires Send + Sync for the move. `UnsafeCell<PinnedMetaStaging>` is not
// Sync, so single-thread access is an assumption of the scheduler design, not
// something the types enforce. The pinned pointer is valid from any thread.
unsafe impl Send for TransformerModel {}
// 2026-09-25: SAFETY: Model methods are only called from the scheduler thread; there is no concurrent `&self` access.
unsafe impl Sync for TransformerModel {}
