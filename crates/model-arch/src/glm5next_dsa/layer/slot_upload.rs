// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: How a drafter row's KV slot reaches the device (`METRALE_GLM_MTP_ROW_NOSYNC`).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: `StreamOrdered` is used only off capture (`write_kv_row` refuses it under one).

/// 2026-10-21: How `write_kv_row` uploads the row's KV slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotUpload {
    /// 2026-10-21: `copy_h2d`: the host waits for the stream, so the drafter launches behind it.
    Blocking,
    /// 2026-10-21: `copy_h2d_async` on the row's stream (`METRALE_GLM_MTP_ROW_NOSYNC`): the
    /// source is copied out before the call returns (`GpuBackend::copy_h2d_async`), and the
    /// latent write that reads the slot is the next op on the same stream.
    StreamOrdered,
}

impl SlotUpload {
    /// 2026-10-21: The mode for the lever's value.
    pub fn from_lever(row_nosync: bool) -> Self {
        if row_nosync {
            Self::StreamOrdered
        } else {
            Self::Blocking
        }
    }
}
