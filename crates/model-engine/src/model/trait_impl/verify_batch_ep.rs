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
use crate::traits::{ModelSsmState, ModelVerify, SequenceState, VerifyBatchedOpts};

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
        Ok(true)
    }
}
