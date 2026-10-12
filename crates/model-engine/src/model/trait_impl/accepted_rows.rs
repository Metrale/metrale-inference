// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: `ModelLevers::mtp_accepted_rows` (`METRALE_MTP_ACCEPTED_ROWS=1`): the drafter rows
//! of accepted drafts, written before the next propose from the verify forward's live rows.
//!
//! A propose writes one drafter row, pair key `position - 1`: `(hidden_{position-1},
//! token_position)`. After a verify that accepted `a` drafts, `position` has moved `a + 1`, so the
//! `a` keys in between, `(hidden_k, token_{k+1})` for the accepted positions, are never written:
//! at K=1 and p1 ~0.66 about two rows in five. vLLM's drafter runs every accepted position. Their
//! target hiddens are the verify forward's rows just before the one `save_hidden_for_mtp` saved,
//! and their tokens are already in `seq.tokens`.
//!
//! Owner: model-engine speculative decoding.
//! Invariants:
//! - Only a dense-row drafter is fed (GLM-5.3's `Glm5NextMtpHead`): its newest pair key is
//!   `drafter_rows - 1`, and rows append at `drafter_rows`. 2026-10-12: When an older key was never
//!   written the appended keys sit past their rows (as a propose's own row then does); the
//!   bootstrap's key is written by the drafter prefill instead (`ensure_drafter_context`).
//! - Rows are read only while `mtp_hidden_live` holds (the target hidden came from the live
//!   `hidden_states` rows) and only rows below the saved one, so a serial decode step (saved row
//!   0) or a stash-sourced propose writes nothing.
//! - Drafter-only: the target verifies every draft, so acceptance can change, emitted tokens
//!   cannot. Every rank runs this in `run_mtp_propose_inner` with the same latched row, the same
//!   `position` and the same tokens, so the ranks' drafter rows stay identical.

use metrale_model_layers::layer::ForwardContext;
use metrale_model_layers::speculative::{DraftProposer, ProposerState};

use super::super::types::TransformerModel;

/// 2026-10-11: The missing rows: pair keys `first_key..first_key + rows`, whose hiddens are
/// `hidden_states` rows `hidden_row0..hidden_row0 + rows`.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct AcceptedRows {
    pub first_key: usize,
    pub rows: usize,
    pub hidden_row0: usize,
    /// 2026-10-12: The drafter row the first one is appended at (its row count): the key when
    /// the drafter has every earlier key, past it when an older key was never written.
    pub row_base: usize,
}

/// 2026-10-11: The rows to write before a propose at `position`, or `None`. `last_key` is the
/// drafter's newest pair key (`None` when it has no rows), `drafter_rows` its row count,
/// `hidden_idx` the `hidden_states` row the target hidden was saved from, `live` whether that
/// save read the live rows, `tokens_len` the sequence's token count.
pub(super) fn plan_accepted_rows(
    drafter_rows: usize,
    last_key: Option<usize>,
    position: usize,
    hidden_idx: usize,
    live: bool,
    tokens_len: usize,
) -> Option<AcceptedRows> {
    let want = position.checked_sub(1)?;
    // 2026-10-11: Dense rows only: a drafter whose keys are not its row indices has no
    // contiguous key to append at.
    if last_key.map_or(drafter_rows != 0, |k| k + 1 != drafter_rows) {
        return None;
    }
    let gap = want.checked_sub(drafter_rows).filter(|&g| g > 0)?;
    // 2026-10-12: Only this verify's rows below the saved one are this sequence's: a gap wider
    // than them (keys no verify covered: the bootstrap decode's, a gated-off stretch) writes the
    // newest `hidden_idx` keys, the accepted drafts', appended after the drafter's rows. Measured
    // 2026-10-12 (nOfeed): every C1 serve opened with that one-key bootstrap gap, so the
    // gap-equals-accepted rule never fired.
    let rows = gap.min(hidden_idx);
    if !live || rows == 0 || tokens_len <= want {
        return None;
    }
    Some(AcceptedRows {
        first_key: want - rows,
        rows,
        hidden_row0: hidden_idx - rows,
        row_base: drafter_rows,
    })
}

impl TransformerModel {
    /// 2026-10-11: Write the accepted drafts' rows (see the module docs). A failure is logged and
    /// leaves the drafter as it was: it costs acceptance, not correctness.
    pub(in crate::model) fn feed_accepted_rows(
        &self,
        proposer: &dyn DraftProposer,
        tokens: &[u32],
        position: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) {
        use std::sync::atomic::Ordering;
        let Some(plan) = plan_accepted_rows(
            proposer.drafter_rows(state),
            proposer.last_pair_key(state),
            position,
            self.last_mtp_hidden_idx.load(Ordering::Relaxed),
            self.mtp_hidden_live.load(Ordering::Relaxed),
            tokens.len(),
        ) else {
            return;
        };
        let h = self.config.hidden_size;
        // 2026-10-11: Row r is key `first_key + r` = `(embed(tokens[first_key + r + 1]),
        // hidden row hidden_row0 + r)`: the window starts at `first_key` (`rows + 1` tokens).
        let window = &tokens[plan.first_key..=plan.first_key + plan.rows];
        let hiddens = self
            .buffers
            .hidden_states()
            .offset(plan.hidden_row0 * h * 2);
        match proposer.catchup_drafter(
            window,
            hiddens,
            plan.row_base,
            plan.first_key + 1,
            state,
            ctx,
            stream,
        ) {
            Ok(w) if w == plan.rows => {
                static LOGGED: std::sync::Once = std::sync::Once::new();
                LOGGED.call_once(|| {
                    tracing::info!(
                        "GLM MTP feed: ACCEPTED_ROWS applied (single-sequence propose, first at \
                         key {})",
                        plan.first_key
                    )
                });
            }
            Ok(w) => tracing::debug!("MTP accepted rows: wrote {w}/{} ({plan:?})", plan.rows),
            Err(e) => tracing::warn!("MTP accepted rows: feed failed ({e:#}); drafter unchanged"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-11: K=1 accept. The drafter wrote key N-1 at the last propose (rows N), the verify
    /// of `[x_N, d]` accepted `d`, so `position` is N+2 and the target hidden is verify row 1. Key
    /// N, `(hidden_N, d)`, is verify row 0 and token N+1.
    #[test]
    fn a_k1_accept_writes_the_accepted_position_from_verify_row_0() {
        let n = 100;
        assert_eq!(
            plan_accepted_rows(n, Some(n - 1), n + 2, 1, true, n + 2),
            Some(AcceptedRows {
                first_key: n,
                rows: 1,
                hidden_row0: 0,
                row_base: n,
            })
        );
    }

    /// 2026-10-11: K=3 with two accepted: keys N and N+1 from verify rows 0 and 1, the saved row
    /// being 2.
    #[test]
    fn two_accepted_drafts_write_two_rows_below_the_saved_one() {
        let n = 40;
        assert_eq!(
            plan_accepted_rows(n, Some(n - 1), n + 3, 2, true, n + 3),
            Some(AcceptedRows {
                first_key: n,
                rows: 2,
                hidden_row0: 0,
                row_base: n,
            })
        );
    }

    /// 2026-10-11: A reject (or the first propose after the drafter prefill) leaves no gap.
    #[test]
    fn no_gap_writes_nothing() {
        assert_eq!(plan_accepted_rows(50, Some(49), 51, 0, true, 51), None);
        assert_eq!(
            plan_accepted_rows(50, Some(49), 50, 0, true, 50),
            None,
            "drafter ahead"
        );
    }

    /// 2026-10-11: The rows the gap would need are not this forward's: a serial decode step saved
    /// row 0 and a stash-sourced save is not live.
    #[test]
    fn rows_not_from_this_verify_are_never_read() {
        assert_eq!(
            plan_accepted_rows(50, Some(49), 52, 0, true, 52),
            None,
            "serial step"
        );
        assert_eq!(
            plan_accepted_rows(50, Some(49), 52, 1, false, 52),
            None,
            "stash save"
        );
    }

    /// 2026-10-12: The bootstrap shape: the drafter missed the key the bootstrap decode committed
    /// (one key behind), then a K=1 accept. The gap is 2 but the verify holds one row below the
    /// saved one, so only the accepted draft's key (want - 1) is written, at the drafter's next
    /// row, never a row before this verify.
    #[test]
    fn a_wider_gap_writes_only_this_verifys_newest_rows() {
        let rows = 77;
        let position = rows + 3;
        assert_eq!(
            plan_accepted_rows(rows, Some(rows - 1), position, 1, true, position),
            Some(AcceptedRows {
                first_key: position - 2,
                rows: 1,
                hidden_row0: 0,
                row_base: rows,
            })
        );
    }

    /// 2026-10-11: A drafter with no context rows (prefix-cache hit, capture lost) gets only the
    /// accepted draft's key from the verify, not the prompt it never saw.
    #[test]
    fn an_empty_drafter_gets_only_the_accepted_key() {
        assert_eq!(
            plan_accepted_rows(0, None, 300, 1, true, 300),
            Some(AcceptedRows {
                first_key: 298,
                rows: 1,
                hidden_row0: 0,
                row_base: 0,
            })
        );
    }

    /// 2026-10-11: Keys that are not row indices (a compacted or carried row space) have no
    /// contiguous append point, and a token list that does not reach the key is refused.
    #[test]
    fn non_dense_rows_and_short_token_lists_are_refused() {
        assert_eq!(
            plan_accepted_rows(50, Some(60), 63, 1, true, 63),
            None,
            "key != row"
        );
        assert_eq!(
            plan_accepted_rows(50, None, 52, 1, true, 52),
            None,
            "rows but no key"
        );
        assert_eq!(
            plan_accepted_rows(50, Some(49), 52, 1, true, 51),
            None,
            "tokens short"
        );
    }
}
