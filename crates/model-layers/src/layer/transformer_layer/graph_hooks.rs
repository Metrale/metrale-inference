// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `LayerGraphHooks`, the host-side work a layer does around a CUDA-graph decode
//! step: the per-step prestage and its re-arm, and the room check and bookkeeping of a
//! replayed step.
//!
//! Owner: model-layers.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

use crate::layer::LayerState;

/// 2026-09-26: A supertrait of `TransformerLayer`; see the module header.
pub trait LayerGraphHooks {
    /// 2026-09-25: Per-step host work of a layer that computes on the host at decode (the
    /// PLE hash and slot upload). The model's single-token decode calls it before any
    /// CUDA graph replay or capture, when the token id is on the host (`decode_a.rs`), so
    /// a graph reads only device buffers refreshed here. Default: nothing.
    fn decode_prestage(
        &self,
        _token: u32,
        _state: &mut dyn LayerState,
        _gpu: &dyn GpuBackend,
        _stream: u64,
    ) -> Result<()> {
        Ok(())
    }

    /// 2026-10-13: `decode_prestage` for the live rows of a batched decode step: row `i`'s
    /// token is `tokens[i]` and its state `states[i]`. The batched decode calls it before a
    /// replay or capture under `METRALE_DECODE_GRAPH_PER_STEP` (`decode_a2.rs`). Default:
    /// `decode_prestage` per row. PLE overrides it with one hash and one slot upload for
    /// the batch, which its multi-sequence forward then consumes.
    fn decode_prestage_rows(
        &self,
        tokens: &[u32],
        states: &mut [&mut (dyn LayerState + 'static)],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            tokens.len() == states.len(),
            "decode_prestage_rows: {} tokens for {} rows",
            tokens.len(),
            states.len()
        );
        for (&token, state) in tokens.iter().zip(states.iter_mut()) {
            self.decode_prestage(token, &mut **state, gpu, stream)?;
        }
        Ok(())
    }

    /// 2026-09-25: Re-arm prestaged state that a failed CUDA-graph capture consumed, so the
    /// same step can re-run eagerly (`decode_a.rs`). It must not recompute the prestage
    /// work: PLE's history already advanced in `decode_prestage`.
    fn decode_prestage_rearm(&self, _state: &mut dyn LayerState) {}

    /// 2026-09-25: Update host-side per-sequence bookkeeping for a step served by a
    /// replayed CUDA graph, in which the layer's `decode` did not run. `seq_len` is the
    /// length before the step's `k` rows. GLM-5.3's DSA layer rewinds its indexer-cache
    /// length to `seq_len` when it is ahead, errors when it is behind, and then advances it
    /// by `k` (`Glm5NextDsaState::sync_to`). The model calls it after a replay in
    /// `decode_a.rs`, `verify_b.rs`, `verify_c.rs` and `verify_c2.rs`. Default: nothing.
    fn sync_replayed_step(
        &self,
        _state: &mut dyn LayerState,
        _seq_len: usize,
        _k: usize,
    ) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Return an error for a replayed step whose writes would land past the end
    /// of a host-tracked cache. The model calls it before launching the graph;
    /// `sync_replayed_step` runs only after the launch. The step ends at `seq_len + k`.
    /// Default: nothing.
    fn check_replay_room(&self, _state: &dyn LayerState, _seq_len: usize, _k: usize) -> Result<()> {
        Ok(())
    }
}
