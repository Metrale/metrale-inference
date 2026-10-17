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
//! - With `capture_all_rows` on, rows `0..k` of the pass land in rows `0..k`; otherwise only
//!   `legacy_row` lands, in row 0, as before the unified context existed.

use anyhow::Result;

use super::super::types::TransformerModel;

/// 2026-10-09: Whether a verify forward captures all its rows: unless
/// `METRALE_DFLASH_EAGLE_FIX=0` or `METRALE_DFLASH_UNIFIED_CTX=0`, the levers that select the
/// scheduler's older context appends. Read once, so a captured graph cannot bake a changed
/// value.
fn capture_all_rows() -> bool {
    static CAPTURE_ALL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CAPTURE_ALL.get_or_init(|| {
        capture_all_rows_from(
            std::env::var("METRALE_DFLASH_EAGLE_FIX").ok().as_deref(),
            std::env::var("METRALE_DFLASH_UNIFIED_CTX").ok().as_deref(),
        )
    })
}

/// 2026-10-09: `capture_all_rows` over the two raw lever values.
fn capture_all_rows_from(eagle_fix: Option<&str>, unified_ctx: Option<&str>) -> bool {
    eagle_fix != Some("0") && unified_ctx != Some("0")
}

impl TransformerModel {
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
        if capture_all_rows() {
            self.try_dflash_capture_all(layer_idx, k, stream)
        } else {
            self.try_dflash_capture(layer_idx, legacy_row, stream)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::capture_all_rows_from;

    /// 2026-10-09: Every row is captured by default and when either lever is anything but `0`;
    /// either lever at `0` keeps the one-row capture its older context append reads.
    #[test]
    fn all_rows_unless_an_older_context_lever_is_off() {
        assert!(capture_all_rows_from(None, None));
        assert!(capture_all_rows_from(Some("1"), Some("1")));
        assert!(!capture_all_rows_from(Some("0"), None));
        assert!(!capture_all_rows_from(None, Some("0")));
    }
}
