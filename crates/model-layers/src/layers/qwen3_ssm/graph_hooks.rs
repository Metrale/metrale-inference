// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: `LayerGraphHooks` for [`Qwen3SsmLayer`]: PLE's host half around a
//! CUDA-graph decode step (moved from `trait_layer.rs`), for one sequence and for the rows
//! of a batched step. A layer without PLE does nothing.
//!
//! Owner: model-layers, GDN/SSM layer (`qwen3_ssm`).
//! Invariants:
//! - A stage is spent exactly once: by the forward that consumes it, or by
//!   `sync_replayed_step` after a replayed graph ran it.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

use super::Qwen3SsmLayer;
use super::ple_seq::ple_seq_state;
use crate::layer::{LayerGraphHooks, LayerState, SsmLayerState};
use crate::layers::ple::PleSeqState;

/// 2026-10-13: This sequence's PLE carry, when it has one.
fn carry(state: &mut dyn LayerState) -> Option<&mut PleSeqState> {
    state
        .as_any_mut()
        .downcast_mut::<SsmLayerState>()?
        .ple
        .as_mut()
}

impl LayerGraphHooks for Qwen3SsmLayer {
    /// 2026-09-25: PLE's host half (id hash, NVMe fault-in, slot upload), run
    /// before graph replay or capture. A no-op on a layer without PLE.
    fn decode_prestage(
        &self,
        token: u32,
        state: &mut dyn LayerState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if let Some(ple) = self.ple.as_ref() {
            let st = ple_seq_state(ple, state, gpu)?;
            ple.prestage(st, &[token], gpu, stream)?;
        }
        Ok(())
    }

    /// 2026-10-13: PLE's host half for every row of a batched step at once
    /// (`PleLayer::prestage_rows`), which the multi-sequence forward consumes.
    fn decode_prestage_rows(
        &self,
        tokens: &[u32],
        states: &mut [&mut (dyn LayerState + 'static)],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let Some(ple) = self.ple.as_ref() else {
            return Ok(());
        };
        let mut carries = Vec::with_capacity(states.len());
        for state in states.iter_mut() {
            carries.push(ple_seq_state(ple, &mut **state, gpu)?);
        }
        ple.prestage_rows(&mut carries, tokens, gpu, stream)
    }

    fn decode_prestage_rearm(&self, state: &mut dyn LayerState) {
        if let Some(ple) = self.ple.as_ref()
            && let Some(st) = carry(state)
        {
            ple.rearm(st);
        }
    }

    /// 2026-10-13: The replayed graph ran the staged step's kernels: the stage is spent.
    fn sync_replayed_step(
        &self,
        state: &mut dyn LayerState,
        _seq_len: usize,
        _k: usize,
    ) -> Result<()> {
        if let Some(ple) = self.ple.as_ref()
            && let Some(st) = carry(state)
        {
            ple.replayed(st);
        }
        Ok(())
    }
}
