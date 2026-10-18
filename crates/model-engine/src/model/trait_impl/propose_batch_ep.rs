// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The batched MTP propose on every rank: rank 0's announcement
//! (`speculative::EP_CMD_MTP_PROPOSE_BATCH`, wire shape in `speculative/propose_batch_wire.rs`),
//! the worker's side, and the forward both run.
//!
//! Owner: model-engine (EP worker protocol).
//! Invariants:
//! - Rank 0 announces a batch only after its proposer said it takes that many sequences
//!   (`propose_batch_max`), and nothing when it falls back to the per-sequence propose; so a
//!   worker that reads the command always runs the batched forward rank 0 runs.
//! - Both ranks read sequence `i`'s target hidden from stash slot `stash[i]`, which the batched
//!   verify filled on each rank in the same batch order (`ep_worker_verify_batch`).

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layer::ForwardContext;
use metrale_model_layers::speculative::{
    DraftProposer, EP_CMD_MTP_PROPOSE_BATCH, PROPOSE_BATCH_MAX_DRAFTS, PROPOSE_BATCH_MAX_SEQS,
    ProposerState, propose_batch_columns, propose_batch_header, propose_batch_payload,
    propose_batch_shape,
};

use super::super::types::TransformerModel;
use super::verify_batch_ep::ordered_slot_refs;
use crate::traits::{ModelEp, SequenceState};

impl TransformerModel {
    /// 2026-10-11: Whether a batched propose runs on every rank: a multi-rank serve whose
    /// proposer shards its block across ranks and proposes through the per-propose commands
    /// (not a collective drafter, which sends its own).
    fn propose_batch_is_collective(&self, proposer: &dyn DraftProposer) -> bool {
        self.multi_rank_protocol_active() && proposer.needs_comm() && !proposer.collective_propose()
    }

    /// 2026-10-11: Rank 0: the batched propose of `seqs`, announced to the workers first when it
    /// runs on every rank. `Ok(None)`, with nothing sent, when it does not apply: no proposer,
    /// no stash, a batch the proposer does not take, or (multi-rank) a shape the wire refuses.
    pub(super) fn run_mtp_propose_batched_dispatch(
        &self,
        tokens: &[u32],
        positions: &[usize],
        stash_idx: &[usize],
        num_drafts: usize,
        seqs: &mut [&mut SequenceState],
        out_conf: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        let Some(proposer) = self.proposer.as_deref() else {
            return Ok(None);
        };
        if self.verify_hidden_stash.is_null() {
            return Ok(None);
        }
        let n = seqs.len();
        let collective = self.propose_batch_is_collective(proposer);
        if collective {
            if proposer.propose_batch_max(&self.buffers, &self.config) < n
                || !(2..=PROPOSE_BATCH_MAX_SEQS).contains(&n)
                || !(1..=PROPOSE_BATCH_MAX_DRAFTS).contains(&num_drafts)
            {
                return Ok(None);
            }
            let slots: Vec<u32> = seqs.iter().map(|s| s.slot_idx as u32).collect();
            let pos: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
            let stash: Vec<u32> = stash_idx.iter().map(|&s| s as u32).collect();
            let payload = propose_batch_payload(&slots, tokens, &pos, &stash)?;
            self.ep_broadcast_cmd_for_seq(0, EP_CMD_MTP_PROPOSE_BATCH)?;
            self.ep_broadcast_tokens(&propose_batch_header(n, num_drafts)?)?;
            self.ep_broadcast_tokens(&payload)?;
        }
        let drafts = self.mtp_propose_batch_forward(
            proposer, tokens, positions, stash_idx, num_drafts, seqs, out_conf, collective,
        )?;
        if collective && drafts.is_none() {
            bail!("batched MTP propose: announced to the workers, then declined on rank 0");
        }
        Ok(drafts)
    }

    /// 2026-10-11: Worker side of `EP_CMD_MTP_PROPOSE_BATCH`: the batch, then the forward rank 0
    /// runs. The drafts are discarded (rank 0 sends the tokens it verifies); the drafter rows
    /// this writes stay in step because both ranks use the same tokens, positions and stash
    /// rows.
    pub(in crate::model) fn ep_worker_mtp_propose_batch(
        &self,
        slots: &mut [Option<SequenceState>],
    ) -> Result<bool> {
        let header = self.ep_broadcast_tokens(&[0u32; 2])?;
        let (n, num_drafts) = propose_batch_shape(&header)?;
        let words = self.ep_broadcast_tokens(&vec![0u32; 4 * n])?;
        let cols = propose_batch_columns(&words, n)?;
        let Some(proposer) = self.proposer.as_deref() else {
            bail!("EP_CMD_MTP_PROPOSE_BATCH on a rank without a proposer");
        };
        let mut refs = ordered_slot_refs(slots, &cols.slots, "ep_worker_mtp_propose_batch")?;
        let drafts = self.mtp_propose_batch_forward(
            proposer,
            &cols.tokens,
            &cols.positions,
            &cols.stash,
            num_drafts,
            &mut refs,
            None,
            true,
        )?;
        if drafts.is_none() {
            bail!("batched MTP propose: rank 0 announced a batch this rank's proposer declines");
        }
        Ok(true)
    }

    /// 2026-10-11: The batched drafter forward both ranks run: each sequence's drafter context
    /// on its first propose, then `propose_batch` over the stash rows, with the communicator
    /// when `collective`.
    #[allow(clippy::too_many_arguments)]
    fn mtp_propose_batch_forward(
        &self,
        proposer: &dyn DraftProposer,
        tokens: &[u32],
        positions: &[usize],
        stash_idx: &[usize],
        num_drafts: usize,
        seqs: &mut [&mut SequenceState],
        out_conf: Option<&mut Vec<Vec<f32>>>,
        collective: bool,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        let stream = self.gpu.default_stream();
        let ctx = ForwardContext {
            comm: if collective { self.comm_ref() } else { None },
            ..self.mtp_propose_ctx()
        };
        // 2026-09-25: Drafter context on each sequence's first propose; returns without
        // work on later calls.
        for seq in seqs.iter_mut() {
            self.ensure_drafter_context(proposer, seq, &ctx, stream);
        }
        let h = self.config.hidden_size;
        let hiddens: Vec<DevicePtr> = stash_idx
            .iter()
            .map(|&i| self.verify_hidden_stash.offset(i * h * 2))
            .collect();
        let mut states: Vec<&mut dyn ProposerState> = Vec::with_capacity(seqs.len());
        for seq in seqs.iter_mut() {
            match seq.proposer_state.as_mut() {
                Some(s) => states.push(s.as_mut()),
                None => return Ok(None),
            }
        }
        proposer.propose_batch(
            tokens,
            &hiddens,
            positions,
            num_drafts,
            &mut states,
            &ctx,
            stream,
            out_conf,
        )
    }
}
