// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The per-layer host hooks of a batched decode step: the room check and the
//! bookkeeping around a replayed batched-decode graph, and the layer vote that lifts the mHC +
//! sparse-index per-sequence rule. 2026-10-09: also the vote that lets the single-sequence
//! decode capture with a communicator.
//!
//! Owner: model-engine (decode).
//! Invariants:
//! - `batch_replay_check_room` runs before `launch_graph` and `batch_replay_sync` after it,
//!   as `decode_a.rs` orders the single-sequence replay.

use anyhow::Result;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    /// 2026-10-08: True when the model has layers and every one answers true to
    /// `decode_multi_seq_selects_index_per_row`. `decode_a2`'s `hc_perseq` and `decode_b`'s
    /// `hc_qsa_perseq` then skip their sparse-index term; a model with any layer answering
    /// false (every model but GLM-5.3 today) keeps it unchanged.
    pub(crate) fn layers_select_index_per_row(&self) -> bool {
        !self.layers.is_empty()
            && self
                .layers
                .iter()
                .all(|l| l.decode_multi_seq_selects_index_per_row())
    }

    /// 2026-10-09: True when the model has layers, every one answers true to
    /// `decode_graph_with_comm`, and `METRALE_COMM_DECODE_GRAPHS` is not `0` (read once per
    /// process). `decode_a.rs` then captures the single-sequence decode with a communicator.
    pub(crate) fn layers_capture_with_comm(&self) -> bool {
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let off =
            *OFF.get_or_init(|| std::env::var("METRALE_COMM_DECODE_GRAPHS").as_deref() == Ok("0"));
        !off && !self.layers.is_empty() && self.layers.iter().all(|l| l.decode_graph_with_comm())
    }

    /// 2026-10-08: Every real row's `check_replay_room` for one step, before a batched graph
    /// replays. The graph writes GLM-5.3's DSA indexer rows from device positions, so a row
    /// past its cache would be written before any host code could refuse it. Padding rows
    /// are not in `seqs` and write the layer's own padding buffers. The default hook does
    /// nothing.
    pub(super) fn batch_replay_check_room(&self, seqs: &[&mut SequenceState]) -> Result<()> {
        for seq in seqs.iter() {
            for (i, layer) in self.layers.iter().enumerate() {
                layer.check_replay_room(&*seq.layer_states[i], seq.seq_len, 1)?;
            }
        }
        Ok(())
    }

    /// 2026-10-08: Every real row's `sync_replayed_step` for one step, after a batched graph
    /// replayed: a replay runs kernels only, so per-sequence host bookkeeping (GLM-5.3's DSA
    /// indexer length) advances here. `seq_len` is still the length before the step. The
    /// default hook does nothing.
    pub(super) fn batch_replay_sync(&self, seqs: &mut [&mut SequenceState]) -> Result<()> {
        for seq in seqs.iter_mut() {
            let seq_len = seq.seq_len;
            for (i, layer) in self.layers.iter().enumerate() {
                layer.sync_replayed_step(seq.layer_states[i].as_mut(), seq_len, 1)?;
            }
        }
        Ok(())
    }
}
