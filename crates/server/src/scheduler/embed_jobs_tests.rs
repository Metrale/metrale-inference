// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: Tests for `run_embed_jobs`: one packed call per tick, each job its own slice in
//! order, a bad job isolated from the good ones, and a generative model refusing.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.
//!
//! 2026-10-24: Mock justification: the scheduler half is tested against a host-only fake model
//! (`FakeEmbedder`) because the real one needs a GPU; the fake keeps the contract that matters
//! here (vectors in input order, a coded refusal for a bad input) and counts its calls.

use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use metrale_error::{ErrorCode, error};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_engine::traits::{
    Model, ModelAdapters, ModelDeviceFeed, ModelDraft, ModelEp, ModelForward, ModelLifecycle,
    ModelLogits, ModelSsmState, ModelStreams, ModelVerify, ModelVision, SequenceState,
};

use super::*;

/// 2026-10-24: Width 2: input `ids` embeds as `[sum(ids), len(ids)]`; id 999 is refused.
struct FakeEmbedder {
    dim: Option<usize>,
    calls: AtomicUsize,
}

impl FakeEmbedder {
    fn new(dim: Option<usize>) -> Self {
        Self {
            dim,
            calls: AtomicUsize::new(0),
        }
    }
}

const UNUSED: &str = "not used by embedding jobs";

impl Model for FakeEmbedder {}
impl ModelLifecycle for FakeEmbedder {
    fn bind_gpu_to_thread(&self) -> Result<()> {
        Ok(())
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        anyhow::bail!(UNUSED)
    }
    fn cache_sequence(&self, _s: &SequenceState) {}
    fn free_sequence(&self, _s: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn compact_sequence(&self, _s: &mut SequenceState, _slot: usize) -> Result<()> {
        Ok(())
    }
    fn detach_slot_for_reuse(&self, _s: &mut SequenceState) {}
    fn kv_block_bytes(&self) -> usize {
        0
    }
}
impl ModelForward for FakeEmbedder {
    fn embedding_dim(&self) -> Option<usize> {
        self.dim
    }
    fn embed_batch(&self, inputs: &[Vec<u32>]) -> Result<Vec<f32>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(i) = inputs.iter().position(|v| v.contains(&999)) {
            return Err(error!(ErrorCode::EmbedInputInvalid, i, "id 999").into());
        }
        Ok(inputs
            .iter()
            .flat_map(|v| [v.iter().sum::<u32>() as f32, v.len() as f32])
            .collect())
    }
    fn prefill(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        anyhow::bail!(UNUSED)
    }
    fn prefill_chunk(
        &self,
        _t: &[u32],
        _s: &mut SequenceState,
        _a: usize,
        _b: usize,
        _c: bool,
        _st: u64,
    ) -> Result<DevicePtr> {
        anyhow::bail!(UNUSED)
    }
    fn decode(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        anyhow::bail!(UNUSED)
    }
    fn decode_batch(
        &self,
        _t: &[u32],
        _s: &mut [&mut SequenceState],
        _st: u64,
    ) -> Result<DevicePtr> {
        anyhow::bail!(UNUSED)
    }
}
impl ModelLogits for FakeEmbedder {
    fn vocab_size(&self) -> usize {
        1000
    }
    fn copy_logits_to_host(&self, _p: DevicePtr, _d: &mut [u8]) -> Result<()> {
        anyhow::bail!(UNUSED)
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn argmax_on_device(&self, _p: DevicePtr, _st: u64) -> Result<u32> {
        anyhow::bail!(UNUSED)
    }
    fn argmax_batch(&self, _p: DevicePtr, _n: usize, _st: u64) -> Result<Vec<u32>> {
        anyhow::bail!(UNUSED)
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        DevicePtr::NULL
    }
}
impl ModelSsmState for FakeEmbedder {
    fn checkpoint_ssm_states(&self, _s: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn rollback_ssm_states(&self, _s: &mut SequenceState, _n: usize) -> Result<()> {
        Ok(())
    }
}
impl ModelDraft for FakeEmbedder {
    fn generate_speculative(
        &self,
        _t: &[u32],
        _p: &metrale_sampling::SamplingParams,
        _n: usize,
    ) -> Result<metrale_model_engine::engine::GenerateResult> {
        anyhow::bail!(UNUSED)
    }
    fn run_mtp_propose(
        &self,
        _t: u32,
        _p: usize,
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<Option<u32>> {
        Ok(None)
    }
    fn run_mtp_propose_multi(
        &self,
        _t: u32,
        _p: usize,
        _n: usize,
        _s: &mut SequenceState,
        _st: u64,
        _g: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        Ok(Vec::new())
    }
    fn trim_proposer_state(&self, _s: &mut SequenceState, _n: usize, _st: u64) -> Result<()> {
        Ok(())
    }
    fn has_proposer(&self) -> bool {
        false
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn decode_draft(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        anyhow::bail!(UNUSED)
    }
    fn save_hidden_for_mtp(&self, _i: usize, _st: u64) -> Result<()> {
        Ok(())
    }
}
impl ModelAdapters for FakeEmbedder {}
impl ModelVerify for FakeEmbedder {
    fn decode_verify(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<Vec<u32>> {
        anyhow::bail!(UNUSED)
    }
    fn decode_verify_graphed(
        &self,
        _t: &[u32; 2],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 2]> {
        anyhow::bail!(UNUSED)
    }
    fn decode_verify_graphed_k3(
        &self,
        _t: &[u32; 3],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 3]> {
        anyhow::bail!(UNUSED)
    }
    fn decode_verify_graphed_k4(
        &self,
        _t: &[u32; 4],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 4]> {
        anyhow::bail!(UNUSED)
    }
}
impl ModelVision for FakeEmbedder {}
impl ModelEp for FakeEmbedder {}
impl metrale_model_engine::traits::ModelCircuit for FakeEmbedder {}
impl ModelStreams for FakeEmbedder {}
impl ModelDeviceFeed for FakeEmbedder {}

type Rx = tokio::sync::oneshot::Receiver<Result<Vec<f32>, EmbedFailure>>;

fn job(inputs: Vec<Vec<u32>>) -> (EmbedJob, Rx) {
    let (reply, rx) = tokio::sync::oneshot::channel();
    (EmbedJob { inputs, reply }, rx)
}

/// 2026-10-24: Two jobs, one call, and each job gets exactly its own inputs' vectors.
#[test]
fn one_call_per_tick_and_each_job_its_slice() {
    let m = FakeEmbedder::new(Some(2));
    let (a, ra) = job(vec![vec![1, 2], vec![5]]);
    let (b, rb) = job(vec![vec![10, 20, 30]]);
    run_embed_jobs(&m, vec![a, b]);
    assert_eq!(
        m.calls.load(Ordering::SeqCst),
        1,
        "the tick's jobs must share one forward"
    );
    assert_eq!(
        ra.blocking_recv().unwrap().unwrap(),
        vec![3.0, 2.0, 5.0, 1.0]
    );
    assert_eq!(rb.blocking_recv().unwrap().unwrap(), vec![60.0, 3.0]);
}

/// 2026-10-24: A refused input fails its own job with its code; the other job still gets its
/// vectors.
#[test]
fn a_bad_job_does_not_fail_the_tick() {
    let m = FakeEmbedder::new(Some(2));
    let (a, ra) = job(vec![vec![1], vec![999]]);
    let (b, rb) = job(vec![vec![4, 4]]);
    run_embed_jobs(&m, vec![a, b]);
    let fa = ra.blocking_recv().unwrap().unwrap_err();
    assert_eq!(fa.code, Some(ErrorCode::EmbedInputInvalid));
    assert!(
        fa.text.starts_with("[0x0203 EmbedInputInvalid]"),
        "{}",
        fa.text
    );
    assert_eq!(rb.blocking_recv().unwrap().unwrap(), vec![8.0, 2.0]);
}

/// 2026-10-24: A generative model answers every job with a refusal and runs nothing.
#[test]
fn a_generative_model_refuses() {
    let m = FakeEmbedder::new(None);
    let (a, ra) = job(vec![vec![1]]);
    run_embed_jobs(&m, vec![a]);
    assert!(ra.blocking_recv().unwrap().is_err());
    assert_eq!(m.calls.load(Ordering::SeqCst), 0);
}
