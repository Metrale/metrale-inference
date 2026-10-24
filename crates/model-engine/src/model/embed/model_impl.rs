// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: `impl Model for EmbedGpuModel`. Only `embed_batch` computes; every generation
//! entry point refuses with `GenerateNotSupported`, and the speculative, SSM and MTP methods
//! either refuse or do nothing. The model holds no paged KV: a sequence is host-only.
//!
//! Owner: model-engine (embedding model).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_error::{ErrorCode, error};
use metrale_gpu_runtime::gpu::DevicePtr;

use super::EmbedGpuModel;
use crate::traits::{
    Model, ModelAdapters, ModelDeviceFeed, ModelDraft, ModelEp, ModelForward, ModelLifecycle,
    ModelLogits, ModelSsmState, ModelStreams, ModelVerify, ModelVision, SequenceState,
};

/// 2026-10-24: The refusal of every generation entry point.
fn no_generate<T>() -> Result<T> {
    Err(error!(ErrorCode::GenerateNotSupported, "this checkpoint").into())
}

impl Model for EmbedGpuModel {}

impl ModelLifecycle for EmbedGpuModel {
    fn bind_gpu_to_thread(&self) -> Result<()> {
        self.gpu.bind_to_thread()
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        no_generate()
    }
    fn free_sequence(&self, _seq: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn compact_sequence(&self, seq: &mut SequenceState, new_slot: usize) -> Result<()> {
        seq.slot_idx = new_slot;
        Ok(())
    }
    fn detach_slot_for_reuse(&self, seq: &mut SequenceState) {
        seq.slot_idx = usize::MAX;
    }
    fn cache_sequence(&self, _seq: &SequenceState) {}
    fn num_free_blocks(&self) -> usize {
        // 2026-10-24: No paged KV cache exists, as for NLLB; a non-zero count keeps admission
        // from waiting on blocks that will never be needed.
        1 << 20
    }
    fn kv_block_bytes(&self) -> usize {
        0
    }
}

impl ModelForward for EmbedGpuModel {
    fn embedding_dim(&self) -> Option<usize> {
        Some(self.s.hidden)
    }
    fn embed_batch(&self, inputs: &[Vec<u32>]) -> Result<Vec<f32>> {
        self.embed_all(inputs)
    }
    fn prefill(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        no_generate()
    }
    fn prefill_chunk(
        &self,
        _t: &[u32],
        _s: &mut SequenceState,
        _start: usize,
        _len: usize,
        _last: bool,
        _st: u64,
    ) -> Result<DevicePtr> {
        no_generate()
    }
    fn decode(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        no_generate()
    }
    fn decode_batch(
        &self,
        _t: &[u32],
        _s: &mut [&mut SequenceState],
        _st: u64,
    ) -> Result<DevicePtr> {
        no_generate()
    }
}

impl ModelLogits for EmbedGpuModel {
    fn vocab_size(&self) -> usize {
        self.w.vocab_rows
    }
    fn copy_logits_to_host(&self, _p: DevicePtr, _dst: &mut [u8]) -> Result<()> {
        no_generate()
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn argmax_on_device(&self, _p: DevicePtr, _st: u64) -> Result<u32> {
        no_generate()
    }
    fn argmax_batch(&self, _p: DevicePtr, _n: usize, _st: u64) -> Result<Vec<u32>> {
        no_generate()
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        DevicePtr::NULL
    }
}

impl ModelAdapters for EmbedGpuModel {}

impl ModelSsmState for EmbedGpuModel {
    fn checkpoint_ssm_states(&self, _seq: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn rollback_ssm_states(&self, _seq: &mut SequenceState, _n: usize) -> Result<()> {
        Ok(())
    }
}

impl ModelVerify for EmbedGpuModel {
    fn decode_verify(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<Vec<u32>> {
        no_generate()
    }
    fn decode_verify_graphed(
        &self,
        _t: &[u32; 2],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 2]> {
        no_generate()
    }
    fn decode_verify_graphed_k3(
        &self,
        _t: &[u32; 3],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 3]> {
        no_generate()
    }
    fn decode_verify_graphed_k4(
        &self,
        _t: &[u32; 4],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 4]> {
        no_generate()
    }
}

impl ModelDraft for EmbedGpuModel {
    fn generate_speculative(
        &self,
        _tokens: &[u32],
        _params: &metrale_sampling::SamplingParams,
        _num_drafts: usize,
    ) -> Result<crate::engine::GenerateResult> {
        no_generate()
    }
    fn has_proposer(&self) -> bool {
        false
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn decode_draft(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        no_generate()
    }
    fn save_hidden_for_mtp(&self, _token_idx: usize, _stream: u64) -> Result<()> {
        Ok(())
    }
    fn run_mtp_propose(
        &self,
        _token: u32,
        _position: usize,
        _seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Option<u32>> {
        Ok(None)
    }
    fn run_mtp_propose_multi(
        &self,
        _token: u32,
        _position: usize,
        _num_drafts: usize,
        _seq: &mut SequenceState,
        _stream: u64,
        _grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        Ok(Vec::new())
    }
    fn trim_proposer_state(
        &self,
        _seq: &mut SequenceState,
        _num_accepted: usize,
        _stream: u64,
    ) -> Result<()> {
        Ok(())
    }
}

impl ModelVision for EmbedGpuModel {}
impl ModelEp for EmbedGpuModel {}
impl crate::traits::ModelCircuit for EmbedGpuModel {}
impl ModelStreams for EmbedGpuModel {}
impl ModelDeviceFeed for EmbedGpuModel {}
