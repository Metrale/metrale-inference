// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The EP/TP worker's side of the batched DFlash verify command
//! (`speculative::EP_CMD_VERIFY_BATCH`; the wire shape is in `speculative/verify_batch_wire.rs`),
//! and the slot lookup it shares with the batched decode.
//!
//! Owner: model-engine (EP worker protocol).
//! Invariants:
//! - The worker runs the batched verify forward rank 0 runs (`decode_verify_batched`, same
//!   tokens, row counts, sequence order and options), so every layer collective has its
//!   partners.
//! - The verdict is read whatever the forward returned, so a failed verify never leaves it on
//!   the wire to be taken for the next command.
//! - After a verdict of counts the worker keeps the positions rank 0 keeps, then folds and
//!   commits the recurrent state in rank 0's order (`gdn_fold_accepted`, then
//!   `commit_accepted_prefix` per sequence).

use anyhow::{Result, bail};
use metrale_model_layers::speculative::{
    kgamma_committed_len, verify_batch_shape, verify_batch_verdict,
};

use super::super::types::TransformerModel;
use crate::traits::{ModelDraft, ModelSsmState, ModelVerify, SequenceState, VerifyBatchedOpts};

/// 2026-10-09: The sequences of `slots` named by `seq_ids`, in that order. Errors, before any
/// sequence is touched, on an id past the slots, a repeated id or an empty slot; `what` names
/// the command in the error.
pub(in crate::model) fn ordered_slot_refs<'a>(
    slots: &'a mut [Option<SequenceState>],
    seq_ids: &[u32],
    what: &str,
) -> Result<Vec<&'a mut SequenceState>> {
    let mut seen = std::collections::HashSet::new();
    for &id in seq_ids {
        let idx = id as usize;
        if idx >= slots.len() {
            bail!("{what}: seq_id {id} exceeds slot capacity {}", slots.len());
        }
        if !seen.insert(id) {
            bail!("{what}: duplicate seq_id {id} in batch");
        }
    }
    // 2026-09-25: Collect `(idx, &mut)` for the populated slots and take them out with
    // `swap_remove`: indexing `slots[seq_ids[i]]` mutably in a loop does not borrow-check,
    // since the compiler cannot prove the indices distinct.
    let mut slot_refs: Vec<(usize, &mut SequenceState)> = slots
        .iter_mut()
        .enumerate()
        .filter_map(|(i, opt)| opt.as_mut().map(|s| (i, s)))
        .collect();
    let mut refs: Vec<&mut SequenceState> = Vec::with_capacity(seq_ids.len());
    for &id in seq_ids {
        let idx = id as usize;
        let pos = slot_refs
            .iter()
            .position(|(i, _)| *i == idx)
            .ok_or_else(|| anyhow::anyhow!("{what}: slot {idx} not allocated"))?;
        let (_, seq) = slot_refs.swap_remove(pos);
        refs.push(seq);
    }
    Ok(refs)
}

impl TransformerModel {
    /// 2026-10-09: Worker side of `EP_CMD_VERIFY_BATCH`: the batch, the forward, the verdict,
    /// then the trim and the recurrent-state commit rank 0 applies.
    pub(in crate::model) fn ep_worker_verify_batch(
        &self,
        slots: &mut [Option<SequenceState>],
    ) -> Result<bool> {
        let n = self.ep_broadcast_u32(0)?;
        let k = self.ep_broadcast_u32(0)?;
        let (n, k) = verify_batch_shape(n, k)?;
        let seq_ids = self.ep_broadcast_tokens(&vec![0u32; n])?;
        let tokens = self.ep_broadcast_tokens(&vec![0u32; n * k])?;
        let mut refs = ordered_slot_refs(slots, &seq_ids, "ep_worker_verify_batch")?;
        self.sync_secondary()?;
        let ks = vec![k; n];
        let stream = self.gpu.default_stream();
        // 2026-10-09: Rank 0's scheduler asks for write-on-accept; the same request keeps the
        // two forwards' graph keys and staging identical.
        let verified = self.decode_verify_batched(
            &tokens,
            &ks,
            &mut refs,
            stream,
            VerifyBatchedOpts {
                write_on_accept: true,
            },
        );
        let words = self.ep_broadcast_tokens(&vec![0u32; n])?;
        let Some(committed) = verify_batch_verdict(&words, k)? else {
            // 2026-10-09: Rank 0's forward failed and it finishes every sequence of the
            // batch; undo this rank's positions so its host state is the pre-verify one.
            if verified.is_ok() {
                for seq in refs.iter_mut() {
                    seq.seq_len -= k;
                    for _ in 0..k.min(seq.tokens.len()) {
                        seq.tokens.pop();
                    }
                }
            }
            bail!("batched verify failed on rank 0; its {n} sequences are finished there");
        };
        verified?;
        for (seq, &c) in refs.iter_mut().zip(&committed) {
            let keep = kgamma_committed_len(seq.seq_len, k, c as u32)?;
            let drop = seq.seq_len - keep;
            seq.seq_len = keep;
            for _ in 0..drop.min(seq.tokens.len()) {
                seq.tokens.pop();
            }
        }
        let slot_idx: Vec<usize> = refs.iter().map(|s| s.slot_idx).collect();
        let rows: Vec<u32> = committed.iter().map(|&c| c as u32).collect();
        self.gdn_fold_accepted(&slot_idx, &rows, k)?;
        for (seq, &c) in refs.iter_mut().zip(&committed) {
            self.commit_accepted_prefix(seq, c, k)?;
        }
        // 2026-10-11: An MTP batch (the MTP proposer every rank runs; the collective DFlash
        // drafter keeps its own context): trim the drafter as rank 0's verdict does
        // (`k4_apply_verdict`, `num_accepted` = committed - 1), and stash each sequence's
        // accepted hidden row (`i * k + committed - 1`) in slot i, as rank 0's step does before
        // its per-sequence proposes read it (`EP_CMD_MTP_PROPOSE` with the stash flag).
        if self
            .proposer
            .as_ref()
            .is_some_and(|p| p.needs_comm() && !p.collective_propose())
        {
            for (seq, &c) in refs.iter_mut().zip(&committed) {
                self.trim_proposer_state(seq, c - 1, 0)?;
            }
            let stash: Vec<usize> = committed
                .iter()
                .enumerate()
                .map(|(i, &c)| i * k + c - 1)
                .collect();
            self.stash_verify_hidden_rows(&stash, 0)?;
            // 2026-10-11: The accepted drafts' drafter rows, as rank 0's batched step writes them
            // before its proposes (`run_mtp_catchup_batched`): a no-op unless
            // `METRALE_MTP_ACCEPTED_ROWS` or `_KV_EXACT` (`accepted_catchup_plan`).
            let plan = accepted_catchup_plan(&refs, &committed, k);
            self.stash_verify_catchup_rows_dispatch(&plan.slot_rows)?;
            self.run_mtp_catchup_batched_dispatch(
                &plan.tokens,
                &plan.first_slot,
                &plan.first_pos,
                &mut refs,
            )?;
        }
        Ok(true)
    }
}

/// 2026-10-11: A worker's catch-up after a batched MTP verify, the same rows rank 0's step
/// stashes and appends (`verify_k4_batch_step`): for sequence i with `committed[i]` rows, its
/// `committed - 1` accepted drafts (at most `MTP_CATCHUP_MAX`), verify rows `i * k + j` into
/// stash slots `i * MTP_CATCHUP_MAX + j`, tokens at positions `seq_len - na..seq_len`.
pub(in crate::model) struct AcceptedCatchup {
    pub slot_rows: Vec<(usize, usize)>,
    pub tokens: Vec<Vec<u32>>,
    pub first_slot: Vec<usize>,
    pub first_pos: Vec<usize>,
}

pub(in crate::model) fn accepted_catchup_plan(
    seqs: &[&mut SequenceState],
    committed: &[usize],
    k: usize,
) -> AcceptedCatchup {
    let cmax = metrale_model_layers::layer::MTP_CATCHUP_MAX;
    let mut p = AcceptedCatchup {
        slot_rows: Vec::new(),
        tokens: Vec::new(),
        first_slot: Vec::new(),
        first_pos: Vec::new(),
    };
    for (i, (seq, &c)) in seqs.iter().zip(committed).enumerate() {
        let na = c.saturating_sub(1).min(cmax);
        let na = if seq.tokens.len() >= seq.seq_len && seq.seq_len >= na {
            na
        } else {
            0
        };
        p.slot_rows
            .extend((0..na).map(|j| (i * cmax + j, i * k + j)));
        p.tokens.push(if na == 0 {
            Vec::new()
        } else {
            seq.tokens[seq.seq_len - na..seq.seq_len].to_vec()
        });
        p.first_slot.push(i * cmax);
        p.first_pos.push(seq.seq_len - na);
    }
    p
}

#[cfg(test)]
mod accepted_catchup_tests {
    use super::*;

    fn seq(tokens: &[u32], seq_len: usize) -> SequenceState {
        let mut s = SequenceState::host_only(0);
        s.tokens = tokens.to_vec();
        s.seq_len = seq_len;
        s
    }

    /// 2026-10-11: A worker's rows match rank 0's step: sequence 0 committed 3 rows of a k=4
    /// verify (two accepted drafts at positions 11 and 12), sequence 1 committed 1 (none). The
    /// drafts are the tokens at `seq_len - 2..seq_len`, their hiddens verify rows 0 and 1 of
    /// sequence 0's block, into stash slots 0 and 1; sequence 1 writes nothing.
    #[test]
    fn a_worker_stashes_and_appends_the_same_rows_as_rank_0() {
        let toks: Vec<u32> = (0..13).collect();
        let mut a = seq(&toks, 13);
        let mut b = seq(&toks[..9], 9);
        let refs: Vec<&mut SequenceState> = vec![&mut a, &mut b];
        let p = accepted_catchup_plan(&refs, &[3, 1], 4);
        let cmax = metrale_model_layers::layer::MTP_CATCHUP_MAX;
        assert_eq!(p.slot_rows, vec![(0, 0), (1, 1)]);
        assert_eq!(p.tokens, vec![vec![11, 12], vec![]]);
        assert_eq!(p.first_slot, vec![0, cmax]);
        assert_eq!(p.first_pos, vec![11, 9]);
    }

    /// 2026-10-11: Sequence 1's block starts at verify row k; more accepted drafts than the
    /// stash holds are clamped to `MTP_CATCHUP_MAX`, and a token list shorter than `seq_len`
    /// writes nothing rather than reading past it.
    #[test]
    fn rows_follow_the_block_and_are_clamped_and_guarded() {
        let cmax = metrale_model_layers::layer::MTP_CATCHUP_MAX;
        let toks: Vec<u32> = (0..64).collect();
        let mut a = seq(&toks[..5], 9);
        let mut b = seq(&toks, 40);
        let refs: Vec<&mut SequenceState> = vec![&mut a, &mut b];
        let p = accepted_catchup_plan(&refs, &[2, cmax + 3], cmax + 3);
        assert!(p.tokens[0].is_empty(), "tokens shorter than seq_len");
        assert_eq!(p.tokens[1].len(), cmax);
        assert_eq!(p.slot_rows.first(), Some(&(cmax, cmax + 3)));
        assert_eq!(p.first_pos[1], 40 - cmax);
    }
}
