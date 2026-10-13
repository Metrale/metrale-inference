// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: The per-step CUDA-graph decision of a batched decode step under
//! `METRALE_DECODE_GRAPH_PER_STEP`, and the host work around its replay or capture, as the
//! single-sequence decode does it (`decode_a.rs`): each layer answers for each live row
//! (`decode_graph_unsupported_at`) instead of the build-time `decode_graph_veto`; the
//! layers' host halves run for the whole batch before the replay or capture
//! (`decode_prestage_rows`); a replay checks each row's room first and syncs each row's
//! host bookkeeping after. Without the lever every function keeps the old behaviour: the
//! static veto, and no host work around a replay.
//!
//! Owner: model-engine (decode).
//! Invariants:
//! - Only the live rows (the batch's sequences) are staged, checked and synced; padding
//!   rows carry dummy states with no host bookkeeping.

use anyhow::Result;
use metrale_model_layers::layer::LayerState;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    /// 2026-10-13: Whether a layer vetoes graphs for this step. `host_tokens` is whether the
    /// batch's token ids are on the host.
    pub(super) fn batch_graph_veto(&self, seqs: &[&mut SequenceState], host_tokens: bool) -> bool {
        if !self.levers.decode_graph_per_step {
            return self.decode_graph_veto;
        }
        self.layers.iter().any(|l| {
            seqs.iter()
                .any(|s| l.decode_graph_unsupported_at(s.seq_len, host_tokens))
        })
    }

    /// 2026-10-13: Every layer's host half for the live rows, before the replay or capture.
    pub(super) fn batch_prestage(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<()> {
        if !self.levers.decode_graph_per_step {
            return Ok(());
        }
        for (li, layer) in self.layers.iter().enumerate() {
            let mut rows: Vec<&mut (dyn LayerState + 'static)> = seqs
                .iter_mut()
                .map(|s| s.layer_states[li].as_mut())
                .collect();
            layer.decode_prestage_rows(tokens, &mut rows, self.gpu.as_ref(), stream)?;
        }
        Ok(())
    }

    /// 2026-10-13: Errors unless every layer can serve every live row's step from a replay.
    pub(super) fn batch_check_replay_room(&self, seqs: &[&mut SequenceState]) -> Result<()> {
        if !self.levers.decode_graph_per_step {
            return Ok(());
        }
        for (li, layer) in self.layers.iter().enumerate() {
            for s in seqs.iter() {
                layer.check_replay_room(&*s.layer_states[li], s.seq_len, 1)?;
            }
        }
        Ok(())
    }

    /// 2026-10-13: Every layer's host bookkeeping of the step a replay served, per live row.
    pub(super) fn batch_sync_replayed(&self, seqs: &mut [&mut SequenceState]) -> Result<()> {
        if !self.levers.decode_graph_per_step {
            return Ok(());
        }
        for (li, layer) in self.layers.iter().enumerate() {
            for s in seqs.iter_mut() {
                let seq_len = s.seq_len;
                layer.sync_replayed_step(s.layer_states[li].as_mut(), seq_len, 1)?;
            }
        }
        Ok(())
    }
}
