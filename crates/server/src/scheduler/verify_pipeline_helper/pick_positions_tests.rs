// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: tests of `pick_positions_from_host` on verify rows that do
//! and do not cross `</think>`, over synthetic BF16 rows and a real
//! `GrammarState` compiled from a `required` tool grammar.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::pick_positions::pick_positions_from_host;
use super::verify_pick_all_with_pipeline;
use crate::grammar::tests::{test_tool_defs, test_vocab};
use crate::grammar::{GrammarEngine, GrammarState};
use crate::scheduler::logit_processors::{LogitsContext, SamplingLevers};
use crate::scheduler::test_support::test_seq;
use crate::scheduler::types::ActiveSeq;
use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_engine::traits::{
    Model, ModelAdapters, ModelDeviceFeed, ModelDraft, ModelEp, ModelForward, ModelLifecycle,
    ModelLogits, ModelSsmState, ModelStreams, ModelVerify, ModelVision, SequenceState,
};

const VOCAB: usize = 131;
const TOOL_CALL_OPEN: u32 = 128;
const TOOL_CALL_CLOSE: u32 = 129;
const EOS: u32 = 130;
/// 2026-09-25: a prose token the tool grammar refuses as the first content
/// token.
const HELLO: u32 = b'h' as u32;
/// 2026-09-25: in-vocab ids standing in for `</think>` and `<think>`.
const THINK_END: u32 = 127;
const THINK_START: u32 = 126;

fn required_tool_grammar() -> GrammarState {
    let vocab = test_vocab();
    let mut engine = GrammarEngine::new(&vocab, &[EOS as i32]).unwrap();
    let compiled = engine
        .compile_hermes_tool_grammar(&test_tool_defs(), false)
        .unwrap();
    GrammarState::new(&compiled, engine.vocab_size())
        .unwrap()
        .with_stop_tokens(&[EOS])
}

/// 2026-09-25: a sequence inside thinking with the tool grammar attached.
fn thinking_seq() -> ActiveSeq {
    let (mut a, _rx) = test_seq(Vec::new(), 5000, None, 10);
    a.finished = false;
    a.inside_thinking = true;
    a.enable_thinking = true;
    a.think_end_token = Some(THINK_END);
    a.think_start_token = Some(THINK_START);
    a.tool_call_start_token = Some(TOOL_CALL_OPEN);
    a.grammar_state = Some(required_tool_grammar());
    a
}

fn row(hot: &[(u32, f32)]) -> Vec<f32> {
    let mut r = vec![0.0f32; VOCAB];
    for &(id, v) in hot {
        r[id as usize] = v;
    }
    r
}

/// 2026-09-25: rows as the little-endian BF16 `[K, vocab]` buffer
/// `pick_positions_from_host` reads.
fn bf16_rows(rows: &[Vec<f32>]) -> Vec<u8> {
    rows.iter()
        .flat_map(|r| {
            r.iter().flat_map(|&v| {
                let b = v.to_bits();
                [(b >> 16) as u8, (b >> 24) as u8]
            })
        })
        .collect()
}

fn with_ctx<R>(f: impl FnOnce(&LogitsContext) -> R) -> R {
    let scratch = crate::scheduler::sched_ctx::DecodeScratch::default();
    let io = crate::scheduler::io::SchedIo::for_test();
    let ctx = LogitsContext {
        scratch: &scratch,
        tel: &*io.tel,
        clock: &*io.clock,
        watchdog: crate::scheduler::helpers::WatchdogParams::default(),
        boundary_mask: None,
        mid_word_mask: None,
        sampling: SamplingLevers::default(),
        think_end_token: Some(THINK_END),
        think_start_token: Some(THINK_START),
        tool_call_start_token: Some(TOOL_CALL_OPEN),
        tool_call_end_token: Some(TOOL_CALL_CLOSE),
        // 2026-09-25: unused: `verify_pick_with_pipeline` sets `verify_pos`
        // per position on its own copy of the context.
        verify_pos: 0,
    };
    f(&ctx)
}

#[test]
fn verify_row_crossing_think_end_masks_the_first_post_think_position() {
    let mut a = thinking_seq();
    // 2026-09-25: row 0 picks `</think>`. Row 1's argmax is prose, with
    // `<tool_call>` second; the grammar must win.
    let buf = bf16_rows(&[
        row(&[(THINK_END, 10.0)]),
        row(&[(HELLO, 10.0), (TOOL_CALL_OPEN, 5.0)]),
    ]);
    let picks = with_ctx(|ctx| pick_positions_from_host(&buf, VOCAB, 2, 2, &mut a, ctx));
    assert_eq!(picks[0], THINK_END, "position 0 closes the reasoning span");
    assert_eq!(
        picks[1], TOOL_CALL_OPEN,
        "the first post-think position must be picked under the pristine grammar, not free-run"
    );
    // 2026-09-25: the loop restores the flags and the matcher; it only
    // picks.
    assert!(
        a.inside_thinking && !a.think_ended,
        "sequence state restored after the loop"
    );
    let gs = a
        .grammar_state
        .as_mut()
        .expect("grammar untouched by the loop");
    assert_eq!(
        gs.num_history_steps(),
        0,
        "</think> never fed; speculative advances rolled back"
    );
}

#[test]
fn verify_row_that_stays_inside_thinking_is_not_masked() {
    // 2026-09-25: control: without `</think>` every position stays inside
    // thinking, unmasked by the grammar, and the matcher is not advanced.
    let mut a = thinking_seq();
    let buf = bf16_rows(&[
        row(&[(HELLO, 10.0)]),
        row(&[(HELLO, 10.0), (TOOL_CALL_OPEN, 5.0)]),
    ]);
    let picks = with_ctx(|ctx| pick_positions_from_host(&buf, VOCAB, 2, 2, &mut a, ctx));
    assert_eq!(picks, vec![HELLO, HELLO]);
    assert!(a.inside_thinking);
    assert_eq!(a.grammar_state.as_ref().unwrap().num_history_steps(), 0);
}

// 2026-10-10: the masked chat fast path (`fast_masked.rs`) returned
// the GPU argmax at any temperature, so speculative content tokens above 0
// were greedy where decode and the host verify path sample. It now declines a
// row with temperature > 0 when `mtp_verify_sample` is on (and
// `force_temp_zero` off), as the other two fast paths already do.

/// 2026-10-10: like [`with_ctx`], with only the masked chat fast path's
/// levers on (`dflash_masked_verify`, `fast_masked`) and `mtp_verify_sample`
/// as given.
fn with_ctx_masked_chat<R>(mtp_verify_sample: bool, f: impl FnOnce(&LogitsContext) -> R) -> R {
    let scratch = crate::scheduler::sched_ctx::DecodeScratch::default();
    let io = crate::scheduler::io::SchedIo::for_test();
    let ctx = LogitsContext {
        scratch: &scratch,
        tel: &*io.tel,
        clock: &*io.clock,
        watchdog: crate::scheduler::helpers::WatchdogParams::default(),
        boundary_mask: None,
        mid_word_mask: None,
        sampling: SamplingLevers {
            dflash_masked_verify: true,
            fast_masked: true,
            mtp_verify_sample,
            ..SamplingLevers::default()
        },
        think_end_token: Some(THINK_END),
        think_start_token: Some(THINK_START),
        tool_call_start_token: Some(TOOL_CALL_OPEN),
        tool_call_end_token: Some(TOOL_CALL_CLOSE),
        verify_pos: 0,
    };
    f(&ctx)
}

/// 2026-10-10: a content token the host rows below make certain.
const WORLD: u32 = b'w' as u32;

#[test]
fn masked_chat_fast_path_at_temperature_zero_returns_the_gpu_argmax() {
    // 2026-10-10: the stub holds no rows, so a D2H would panic: passing proves
    // the masked chat fast path answered.
    let mut a = post_think_grammarless_seq();
    a.temperature = 0.0;
    let model = FastPathStubModel::new(VOCAB, &[]);
    let picks = with_ctx_masked_chat(true, |ctx| {
        verify_pick_all_with_pipeline(&model, &[HELLO], &mut a, ctx, 0)
    });
    assert_eq!(picks, vec![HELLO]);
}

#[test]
fn masked_chat_fast_path_declines_above_temperature_zero() {
    // 2026-10-10: the passed GPU argmax (HELLO) disagrees with the host row,
    // where WORLD holds all the mass. Only the host path, which samples, can
    // return WORLD.
    let mut a = post_think_grammarless_seq();
    a.temperature = 0.6;
    let model = FastPathStubModel::new(VOCAB, &[row(&[(WORLD, 100.0)])]);
    let picks = with_ctx_masked_chat(true, |ctx| {
        verify_pick_all_with_pipeline(&model, &[HELLO], &mut a, ctx, 0)
    });
    assert_eq!(
        picks,
        vec![WORLD],
        "a row above temperature 0 must be sampled on the host, not given the GPU argmax"
    );
}

#[test]
fn masked_chat_fast_path_above_zero_without_verify_sampling_is_unchanged() {
    // 2026-10-10: with `mtp_verify_sample` off the host path would itself take
    // the argmax, so the fast path still answers (no rows: a D2H would panic).
    let mut a = post_think_grammarless_seq();
    a.temperature = 0.6;
    let model = FastPathStubModel::new(VOCAB, &[]);
    let picks = with_ctx_masked_chat(false, |ctx| {
        verify_pick_all_with_pipeline(&model, &[HELLO], &mut a, ctx, 0)
    });
    assert_eq!(picks, vec![HELLO]);
}

/// 2026-09-29: a sequence past `</think>` (`think_ended`) with no grammar:
/// the grammarless fast path's eligibility regime.
fn post_think_grammarless_seq() -> ActiveSeq {
    let (mut a, _rx) = test_seq(Vec::new(), 5000, None, 10);
    a.finished = false;
    a.inside_thinking = false;
    a.enable_thinking = true;
    a.think_end_token = Some(THINK_END);
    a.think_start_token = Some(THINK_START);
    a.think_ended = true;
    a.grammar_state = None;
    a
}

/// 2026-09-29: minimal `Model`: only `vocab_size`, `logits_buffer_ptr` and
/// `copy_logits_to_host` work (the host-path D2H the guard routes to); the
/// rest is unreachable for a grammarless, temperature-0 verify call. Same
/// pattern as `PrefillStubModel` (`prefill_fifo_tests.rs`).
struct FastPathStubModel {
    vocab: usize,
    /// 2026-09-29: `[K, vocab]` BF16 bytes, as `bf16_rows` lays them out.
    buf: Vec<u8>,
}

impl FastPathStubModel {
    fn new(vocab: usize, rows: &[Vec<f32>]) -> Self {
        Self {
            vocab,
            buf: bf16_rows(rows),
        }
    }
}

impl Model for FastPathStubModel {}

impl ModelLifecycle for FastPathStubModel {
    fn free_sequence(&self, _seq: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn cache_sequence(&self, _seq: &SequenceState) {}
    fn detach_slot_for_reuse(&self, _seq: &mut SequenceState) {}
    fn bind_gpu_to_thread(&self) -> Result<()> {
        Ok(())
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        Ok(SequenceState::host_only(0))
    }
    fn compact_sequence(&self, _s: &mut SequenceState, _new_slot: usize) -> Result<()> {
        unreachable!("no compaction in this harness")
    }
    fn kv_block_bytes(&self) -> usize {
        0
    }
}

impl ModelForward for FastPathStubModel {
    fn prefill_chunk(
        &self,
        _t: &[u32],
        _s: &mut SequenceState,
        _chunk_start: usize,
        _chunk_len: usize,
        _is_last: bool,
        _st: u64,
    ) -> Result<DevicePtr> {
        unreachable!("no prefill in this harness")
    }
    fn prefill(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        unreachable!("no prefill in this harness")
    }
    fn decode(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        unreachable!("no decode in this harness")
    }
    fn decode_batch(
        &self,
        _t: &[u32],
        _s: &mut [&mut SequenceState],
        _st: u64,
    ) -> Result<DevicePtr> {
        unreachable!("no decode in this harness")
    }
}

impl ModelLogits for FastPathStubModel {
    fn argmax_on_device(&self, _logits_ptr: DevicePtr, _stream: u64) -> Result<u32> {
        unreachable!("no on-device argmax in this harness")
    }
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        unreachable!("no MTP in this harness")
    }
    fn copy_logits_to_host(&self, logits_ptr: DevicePtr, dst: &mut [u8]) -> Result<()> {
        let off = logits_ptr.0 as usize;
        dst.copy_from_slice(&self.buf[off..off + dst.len()]);
        Ok(())
    }
    fn argmax_batch(&self, _l: DevicePtr, _n: usize, _st: u64) -> Result<Vec<u32>> {
        unreachable!("not the decode fast path")
    }
}

impl ModelAdapters for FastPathStubModel {}

impl ModelSsmState for FastPathStubModel {
    fn checkpoint_ssm_states(&self, _s: &mut SequenceState) -> Result<()> {
        unreachable!("no SSM in this harness")
    }
    fn rollback_ssm_states(&self, _s: &mut SequenceState, _n: usize) -> Result<()> {
        unreachable!("no SSM in this harness")
    }
}

impl ModelVerify for FastPathStubModel {
    fn decode_verify(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<Vec<u32>> {
        unreachable!("no speculation in this harness")
    }
    fn decode_verify_graphed(
        &self,
        _t: &[u32; 2],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 2]> {
        unreachable!("no speculation in this harness")
    }
    fn decode_verify_graphed_k3(
        &self,
        _t: &[u32; 3],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 3]> {
        unreachable!("no speculation in this harness")
    }
    fn decode_verify_graphed_k4(
        &self,
        _t: &[u32; 4],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 4]> {
        unreachable!("no speculation in this harness")
    }
}

impl ModelDraft for FastPathStubModel {
    fn has_proposer(&self) -> bool {
        false
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn decode_draft(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        unreachable!("no speculation in this harness")
    }
    fn save_hidden_for_mtp(&self, _token_idx: usize, _st: u64) -> Result<()> {
        unreachable!("no MTP in this harness")
    }
    fn run_mtp_propose(
        &self,
        _t: u32,
        _p: usize,
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<Option<u32>> {
        unreachable!("no MTP in this harness")
    }
    fn run_mtp_propose_multi(
        &self,
        _t: u32,
        _p: usize,
        _n: usize,
        _s: &mut SequenceState,
        _st: u64,
        _mask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        unreachable!("no MTP in this harness")
    }
    fn trim_proposer_state(&self, _s: &mut SequenceState, _n: usize, _st: u64) -> Result<()> {
        unreachable!("no MTP in this harness")
    }
    fn generate_speculative(
        &self,
        _p: &[u32],
        _params: &metrale_sampling::SamplingParams,
        _n: usize,
    ) -> Result<metrale_model_engine::engine::GenerateResult> {
        unreachable!("no speculation in this harness")
    }
}

impl ModelVision for FastPathStubModel {}

impl ModelEp for FastPathStubModel {}

impl metrale_model_engine::traits::ModelCircuit for FastPathStubModel {}

impl ModelStreams for FastPathStubModel {}

impl ModelDeviceFeed for FastPathStubModel {}
