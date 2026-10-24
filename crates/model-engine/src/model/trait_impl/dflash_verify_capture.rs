// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The DFlash hidden capture of a single-sequence verify forward, one rule for
//! every verify width (K=2, K=3, K=4, the fused decode + verify and K=γ).
//!
//! The scheduler commits verify rows `0..=num_accepted` to the drafter context
//! (`ModelDraft::commit_ctx`) after every verify on a DFlash serve, whatever its width, so
//! every verify forward must leave row `t` of the pass in row `t` of `dflash_hidden_save`.
//! Before this rule the K=2/3/4 forwards captured only their last row (and the fused one only
//! row 0) into row 0, so their verdicts appended a draft's hidden at the anchor's position, a
//! stale row after it, or nothing for the accepted rows.
//!
//! Owner: model-engine (speculative verify).
//! Invariants:
//! - With [`TransformerModel::capture_all_rows`] on, rows `0..k` of the pass land in rows `0..k`; otherwise only
//!   `legacy_row` lands, in row 0, as before the unified context existed.

use anyhow::Result;

use super::super::types::TransformerModel;

impl TransformerModel {
    /// 2026-10-09: Whether a verify forward captures all its rows: unless
    /// `METRALE_DFLASH_EAGLE_FIX` or `METRALE_DFLASH_UNIFIED_CTX` is off, the levers that select
    /// the scheduler's older context appends. Resolved once in the settings, so a captured graph
    /// cannot bake a changed value.
    fn capture_all_rows(&self) -> bool {
        self.settings.speculative.dflash_eagle_fix && self.settings.speculative.dflash_unified_ctx
    }

    /// 2026-10-09: DFlash capture after layer `layer_idx` of a `k`-row single-sequence verify:
    /// rows `0..k` into rows `0..k` of `dflash_hidden_save`, or, under the older context levers,
    /// row `legacy_row` into row 0. A no-op without DFlash, off rank 0, or for a layer the
    /// drafter does not read.
    pub(super) fn dflash_capture_verify_rows(
        &self,
        layer_idx: usize,
        k: usize,
        legacy_row: usize,
        stream: u64,
    ) -> Result<()> {
        if self.capture_all_rows() {
            self.try_dflash_capture_all(layer_idx, k, stream)
        } else {
            self.try_dflash_capture(layer_idx, legacy_row, stream)
        }
    }
}
