// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The EP/TP worker's side of the DFlash K=γ verify command
//! (`speculative::EP_CMD_VERIFY_KGAMMA`; the wire shape is in `speculative/kgamma_wire.rs`).
//!
//! Owner: model-engine (EP worker protocol).
//! Invariants:
//! - The worker runs the K-row verify forward rank 0 runs (`decode_verify_graphed_kgamma`), so
//!   every layer collective has its partners.
//! - The verdict word is read whatever the forward returned, so a failed verify never leaves it
//!   on the wire to be taken for the next command.
//! - After the verdict the worker keeps the positions rank 0 keeps and commits its recurrent
//!   state through `commit_accepted_prefix`, as rank 0 does.

use anyhow::Result;
use metrale_model_layers::speculative::{kgamma_committed_len, kgamma_width};

use super::super::types::TransformerModel;
use crate::traits::{Model, SequenceState};

impl TransformerModel {
    /// 2026-10-08: Worker side of `EP_CMD_VERIFY_KGAMMA` on the addressed slot's sequence.
    pub(in crate::model) fn ep_worker_verify_kgamma(
        &self,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<()> {
        let k = kgamma_width(self.ep_broadcast_u32(0)?)?;
        let tokens = self.ep_broadcast_tokens(&vec![0u32; k])?;
        self.sync_secondary()?;
        let verified = self.decode_verify_graphed_kgamma(&tokens, seq, stream);
        let committed = self.ep_broadcast_u32(0)?;
        verified?;
        let keep = kgamma_committed_len(seq.seq_len, k, committed)?;
        let drop = seq.seq_len - keep;
        seq.seq_len = keep;
        for _ in 0..drop.min(seq.tokens.len()) {
            seq.tokens.pop();
        }
        self.commit_accepted_prefix(seq, committed as usize, k)
    }
}
