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
    K2Flags, K2Verdict, k2_announce_block, k2_announced_tokens, k2_fused_requested,
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
        flags: K2Flags,
    ) -> Result<()> {
        // 2026-10-21: The screened head only where this rank built it (`screen_head.rs`); the
        // block tells the workers.
        let flags = K2Flags {
            screen_head: self.screen_arm(flags.screen_head),
            ..flags
        };
        let block = k2_announce_block(seq_id, tokens, flags);
        // 2026-10-21: The control channel never waits behind the propose in flight.
        if self.ctrl_send_block(&block, None)? {
            return Ok(());
        }
        // 2026-10-11: A device-step verify (its draft on the device) is announced without a
        // host wait, so the host queues the verify behind the propose in flight.
        if tokens[1] == metrale_model_layers::speculative::DEVICE_DRAFT {
            return self.ep_announce_device_step(&block);
        }
        self.ep_broadcast_block(&block)
    }

    /// 2026-10-11: Rank 0: the verdict, in one broadcast.
    pub(in crate::model) fn ep_send_k2_verdict_impl(&self, verdict: K2Verdict) -> Result<()> {
        if self.ctrl_send_verdict(verdict)? {
            return Ok(());
        }
        self.ep_broadcast_tokens(&verdict.to_words()).map(|_| ())
    }

    /// 2026-10-11: Worker side of `EP_CMD_VERIFY_K2_FUSED`.
    pub(in crate::model) fn ep_worker_verify_k2_fused(
        &self,
        block: &metrale_model_layers::ep_wire::CmdBlock,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<()> {
        let (tokens, flags) = k2_announced_tokens(block)?;
        let device_verdict = flags.device_verdict;
        self.screen_arm_from_block(flags.screen_head)?;
        self.sync_secondary()?;
        let verified = self.decode_verify_graphed(&tokens, seq, stream);
        // 2026-10-11: `METRALE_GLM_MTP_DEVICE_STEP`: commit and propose on this rank's own verdict
        // at once; rank 0's verdict follows the propose (`device_step.rs`).
        if device_verdict {
            return self.ep_worker_k2_device_verdict(seq, verified?);
        }
        let words = self.ep_recv_k2_verdict_words()?;
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
    ///
    /// 2026-10-11: In snapshot mode the worker commits as rank 0 does (`commit_accepted_prefix`,
    /// nothing after an accept, intermediate 0 after a reject), as the K=γ worker already does.
    /// It used to checkpoint the live state after every accept (53.6 MB per step at TP=3) and
    /// roll back then checkpoint after every reject (107 MB), on the secondary stream beside
    /// the propose. In snapshot mode that checkpoint is never read: the only reads of
    /// `h_state_checkpoint` / `conv_state_checkpoint` are the `num_accepted == 0` rewinds
    /// (`start_rollback_and_checkpoint_async_dispatch`, `rollback_ssm_states`), and every
    /// caller passes 1 or more. The live state after the commit is the same copy of
    /// intermediate 0 either way. Replay mode reads the checkpoint (`replay_parts`), so it keeps
    /// the checkpointing path.
    pub(in crate::model) fn ep_worker_k2_commit(
        &self,
        seq: &mut SequenceState,
        accepted: bool,
    ) -> Result<()> {
        if let K2WorkerCommit::Mirror { rows } =
            k2_worker_commit(self.ssm_pool.rollback_mode, accepted)
        {
            if !accepted {
                seq.seq_len -= 1;
                seq.tokens.pop();
            }
            self.trim_proposer_state(seq, usize::from(accepted), 0)?;
            return self.commit_accepted_prefix(seq, rows, 2);
        }
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
        let flag = metrale_model_layers::speculative::MTP_HIDDEN_FROM_STASH as usize;
        // 2026-10-11: A flagged word names a batched-verify stash slot (rank 0 copied it with
        // `save_hidden_for_mtp_from_stash`); otherwise a `hidden_states` row.
        let saved = if hidden_idx & flag != 0 {
            self.save_hidden_for_mtp_from_stash(hidden_idx & !flag, stream)
        } else {
            self.save_hidden_for_mtp(hidden_idx, stream)
        };
        if let Err(e) = saved {
            tracing::warn!("EP worker save_hidden_for_mtp({hidden_idx:#x}) failed: {e:#}");
        }
        // 2026-10-21: With the control channel the lone draft may stay on the device
        // (`ep_ctrl::worker_proposes_on_device`).
        let device = super::super::ep_ctrl::worker_proposes_on_device(
            self.settings.runtime.glm_ep_ctrl_comm,
        );
        if let Err(e) =
            self.run_mtp_propose_inner(last_token, position, num_drafts, seq, None, device)
        {
            tracing::warn!("EP worker MTP propose failed (continuing): {e:#}");
        }
    }
}

/// 2026-10-11: How a worker commits its recurrent state after a K=2 verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::model) enum K2WorkerCommit {
    /// 2026-10-11: As rank 0: `commit_accepted_prefix(rows, 2)`, no checkpoint.
    Mirror { rows: usize },
    /// 2026-10-11: Checkpoint after an accept, roll back and checkpoint after a reject.
    Checkpoint,
}

/// 2026-10-11: The worker's K=2 commit for a rollback mode and verdict (see
/// `ep_worker_k2_commit`): rank 0's commit in snapshot mode, the checkpointing path in replay
/// mode.
pub(in crate::model) fn k2_worker_commit(
    mode: metrale_model_layers::ssm_reserve::SsmRollbackMode,
    accepted: bool,
) -> K2WorkerCommit {
    use metrale_model_layers::ssm_reserve::SsmRollbackMode;
    match mode {
        SsmRollbackMode::Snapshot => K2WorkerCommit::Mirror {
            rows: 1 + usize::from(accepted),
        },
        SsmRollbackMode::Replay => K2WorkerCommit::Checkpoint,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrale_model_layers::ssm_reserve::SsmRollbackMode;

    /// 2026-10-11: Snapshot mode commits what rank 0 commits: both rows after an accept (a
    /// no-op in `commit_accepted_prefix`), row 0 after a reject, and never 0 rows, which
    /// `commit_accepted_prefix` refuses.
    #[test]
    fn snapshot_mode_commits_what_rank_0_commits() {
        assert_eq!(
            k2_worker_commit(SsmRollbackMode::Snapshot, true),
            K2WorkerCommit::Mirror { rows: 2 }
        );
        assert_eq!(
            k2_worker_commit(SsmRollbackMode::Snapshot, false),
            K2WorkerCommit::Mirror { rows: 1 }
        );
    }

    /// 2026-10-21: A worker leaves its draft on the device only with the control channel, and
    /// then only where rank 0 may too (`wants_device_draft`: the device-step lever, one draft, no
    /// mask); without the channel it proposes and reads back as before. A device draft is safe on
    /// a worker even where rank 0 reads its own back: the next verify takes row 1 from the cell
    /// only when rank 0 announces `DEVICE_DRAFT`, and rank 0's token otherwise.
    #[test]
    fn a_worker_keeps_its_draft_on_the_device_only_with_the_control_channel() {
        use super::super::device_step::wants_device_draft;
        use crate::model::ep_ctrl::worker_proposes_on_device;
        for ctrl in [false, true] {
            for lever in [false, true] {
                for drafts in [1usize, 2] {
                    let got =
                        wants_device_draft(worker_proposes_on_device(ctrl), lever, drafts, false);
                    assert_eq!(
                        got,
                        ctrl && lever && drafts == 1,
                        "ctrl {ctrl} lever {lever} drafts {drafts}"
                    );
                }
            }
        }
    }

    /// 2026-10-11: Replay mode keeps the checkpointing path, since its verify reads the
    /// checkpoint.
    #[test]
    fn replay_mode_keeps_checkpointing() {
        for accepted in [true, false] {
            assert_eq!(
                k2_worker_commit(SsmRollbackMode::Replay, accepted),
                K2WorkerCommit::Checkpoint
            );
        }
    }
}
