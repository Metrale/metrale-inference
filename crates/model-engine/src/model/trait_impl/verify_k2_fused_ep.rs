// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The fused multi-rank K=2 MTP step (`speculative::k2_wire`, wire shape there):
//! rank 0's announcement and verdict, the worker's side, and the worker-side K=2 commit and
//! MTP propose the fused and the unfused commands share.
//!
//! Owner: model-engine (EP worker protocol).
//! Invariants:
//! - The worker runs the 2-row verify forward rank 0 runs, so every layer collective has its
//!   partners, and reads the verdict whatever the forward returned, so a failed verify never
//!   leaves the verdict on the wire to be taken for the next command.
//! - The worker proposes exactly when the verdict says `num_drafts > 0`, which is exactly when
//!   rank 0 proposes (`verify_k2_step.rs`), so the propose's collectives pair up.

use anyhow::Result;
use metrale_model_layers::ep_wire::EpWire;
use metrale_model_layers::speculative::{
    K2_VERDICT_WORDS, K2Verdict, k2_announce_block, k2_announced_tokens, k2_fused_requested,
};

use super::super::types::TransformerModel;
use crate::traits::{ModelDraft, ModelSsmState, ModelVerify, SequenceState};

impl TransformerModel {
    /// 2026-10-11: Whether this serve runs the fused K=2 wire: the lever, the multi-rank
    /// protocol on the v3 block, and a proposer every rank runs through `EP_CMD_MTP_PROPOSE`
    /// (sharded, and not the collective DFlash propose, which sends its own words).
    pub(in crate::model) fn k2_fused_wire_impl(&self) -> bool {
        k2_fused_requested()
            && self.multi_rank_protocol_active()
            && EpWire::get().one_block()
            && self
                .proposer
                .as_ref()
                .is_some_and(|p| p.needs_comm() && !p.collective_propose())
    }

    /// 2026-10-11: Rank 0: the announcement block.
    pub(in crate::model) fn ep_announce_k2_fused_impl(
        &self,
        seq_id: u32,
        tokens: [u32; 2],
    ) -> Result<()> {
        self.ep_broadcast_block(&k2_announce_block(seq_id, tokens))
    }

    /// 2026-10-11: Rank 0: the verdict, in one broadcast.
    pub(in crate::model) fn ep_send_k2_verdict_impl(&self, verdict: K2Verdict) -> Result<()> {
        self.ep_broadcast_tokens(&verdict.to_words()).map(|_| ())
    }

    /// 2026-10-11: Worker side of `EP_CMD_VERIFY_K2_FUSED`.
    pub(in crate::model) fn ep_worker_verify_k2_fused(
        &self,
        block: &metrale_model_layers::ep_wire::CmdBlock,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<()> {
        let tokens = k2_announced_tokens(block)?;
        self.sync_secondary()?;
        let verified = self.decode_verify_graphed(&tokens, seq, stream);
        let words = self.ep_broadcast_tokens(&[0u32; K2_VERDICT_WORDS])?;
        verified?;
        let verdict = K2Verdict::from_words(&words)?;
        self.ep_worker_k2_commit(seq, verdict.accepted)?;
        if verdict.num_drafts > 0 {
            let position = verdict.worker_position(seq.seq_len)?;
            self.ep_worker_mtp_propose(
                seq,
                verdict.next_token,
                position,
                verdict.num_drafts as usize,
                verdict.hidden_idx(),
            );
        }
        Ok(())
    }

    /// 2026-10-11: The worker's commit after a K=2 verdict, for the fused and the unfused
    /// command: keep both rows after an accept, drop the draft row after a reject.
    pub(in crate::model) fn ep_worker_k2_commit(
        &self,
        seq: &mut SequenceState,
        accepted: bool,
    ) -> Result<()> {
        if accepted {
            self.start_checkpoint_async(seq)?;
            self.trim_proposer_state(seq, 1, 0)
        } else {
            seq.seq_len -= 1;
            seq.tokens.pop();
            self.trim_proposer_state(seq, 0, 0)?;
            self.start_rollback_and_checkpoint_async(seq, 1)
        }
    }

    /// 2026-10-11: The worker's MTP propose (from `EP_CMD_MTP_PROPOSE`, or a fused K=2
    /// verdict): save the hidden row rank 0 saved, then run the same drafter forward, so its
    /// collectives have a partner. The drafts are discarded (rank 0 broadcasts the tokens it
    /// verifies); the drafter KV this writes stays in step because both ranks use the same
    /// `(last_token, position)` and the same hidden row. Errors are logged, not returned: rank
    /// 0 decides what is verified.
    pub(in crate::model) fn ep_worker_mtp_propose(
        &self,
        seq: &mut SequenceState,
        last_token: u32,
        position: usize,
        num_drafts: usize,
        hidden_idx: usize,
    ) {
        let stream = self.gpu.default_stream();
        if let Err(e) = self.save_hidden_for_mtp(hidden_idx, stream) {
            tracing::warn!("EP worker save_hidden_for_mtp({hidden_idx}) failed: {e:#}");
        }
        if let Err(e) = self.run_mtp_propose_inner(last_token, position, num_drafts, seq, None) {
            tracing::warn!("EP worker MTP propose failed (continuing): {e:#}");
        }
    }
}
