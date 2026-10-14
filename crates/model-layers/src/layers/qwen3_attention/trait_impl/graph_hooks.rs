// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: `LayerGraphHooks` for [`Qwen3AttentionLayer`]: the QSA indexer's host
//! bookkeeping around a CUDA-graph decode step. A replayed step ingests and pools on the
//! device at the device position (`qsa_ingest_step`), so the host counters advance here;
//! a capture that recorded the step without running it is rewound so it re-runs eagerly.
//! A layer without an indexer does nothing.
//!
//! Owner: model-layers, Qwen3 attention layer.
//! Invariants: none beyond `qsa_decode.rs`'s.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

use super::super::Qwen3AttentionLayer;
use crate::layer::{AttnLayerState, LayerGraphHooks, LayerState};
use crate::layers::qsa::{QsaIndexer, QsaSeqState};

impl Qwen3AttentionLayer {
    /// 2026-10-13: The indexer and this sequence's indexer state, when both exist.
    fn qsa_parts<'a>(
        &'a self,
        state: &'a mut dyn LayerState,
    ) -> Option<(&'a QsaIndexer, &'a mut QsaSeqState)> {
        let qsa = self.qsa.as_ref()?;
        let attn = state.as_any_mut().downcast_mut::<AttnLayerState>()?;
        Some((qsa, attn.qsa.as_mut()?))
    }
}

impl LayerGraphHooks for Qwen3AttentionLayer {
    fn decode_prestage(
        &self,
        _token: u32,
        state: &mut dyn LayerState,
        _gpu: &dyn GpuBackend,
        _stream: u64,
    ) -> Result<()> {
        if let Some((qsa, st)) = self.qsa_parts(state) {
            qsa.begin_step(st);
        }
        Ok(())
    }

    fn decode_prestage_rearm(&self, state: &mut dyn LayerState) {
        if let Some((qsa, st)) = self.qsa_parts(state) {
            qsa.rearm_step(st);
        }
    }

    fn check_replay_room(&self, state: &dyn LayerState, seq_len: usize, k: usize) -> Result<()> {
        let Some(qsa) = self.qsa.as_ref() else {
            return Ok(());
        };
        let st = state
            .as_any()
            .downcast_ref::<AttnLayerState>()
            .and_then(|attn| attn.qsa.as_ref())
            .ok_or_else(|| anyhow::anyhow!("QSA: a graph replay for a sequence without state"))?;
        qsa.check_replay(st, seq_len, k)
    }

    fn sync_replayed_step(
        &self,
        state: &mut dyn LayerState,
        seq_len: usize,
        k: usize,
    ) -> Result<()> {
        if self.qsa.is_none() {
            return Ok(());
        }
        let (qsa, st) = self
            .qsa_parts(state)
            .ok_or_else(|| anyhow::anyhow!("QSA: a graph replay for a sequence without state"))?;
        qsa.replayed_steps(st, seq_len, k)
    }
}
