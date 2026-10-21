// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Shared fixture for the tests that hold the speculative picks
//! (the verify window, the MTP bootstrap) to the serial decode path: the
//! grammar tests' 131-token vocabulary, host logits rows behind
//! `PreemptStubModel` (the only mock: there is no GPU in a unit test), and the
//! `LogitsContext` a step builds from its `SchedCtx` (`core/lane_decode.rs`).
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::logit_processors::LogitsContext;
use super::sched_ctx::SchedCtx;
use super::test_support::{HostLogits, PreemptStubModel, test_seq};
use super::types::ActiveSeq;
use crate::grammar::tests::{test_tool_defs, test_vocab};
use crate::grammar::{GrammarEngine, GrammarState};
use std::sync::Arc;

pub(super) const VOCAB: usize = 131;
pub(super) const TOOL_CALL: u32 = 128;
pub(super) const TOOL_CALL_END: u32 = 129;
pub(super) const TOY_EOS: u32 = 130;
/// 2026-10-21: in-vocab ids standing in for `</think>` and `<think>`.
pub(super) const THINK_END: u32 = 127;
pub(super) const THINK_START: u32 = 126;
/// 2026-10-21: a prose token the tool grammar refuses as the first content
/// token and after `<tool_call>`.
pub(super) const HELLO: u32 = b'h' as u32;

/// 2026-10-21: the `required` Hermes tool grammar over the toy vocab: the
/// output must open with `<tool_call>`.
pub(super) fn required_tool_grammar() -> GrammarState {
    let vocab = test_vocab();
    let mut engine = GrammarEngine::new(&vocab, &[TOY_EOS as i32]).unwrap();
    let compiled = engine
        .compile_hermes_tool_grammar(&test_tool_defs(), false)
        .unwrap();
    GrammarState::new(&compiled, engine.vocab_size())
        .unwrap()
        .with_stop_tokens(&[TOY_EOS])
}

/// 2026-10-21: the first id the grammar allows after `<tool_call>`.
pub(super) fn first_id_after_tool_call() -> u32 {
    let mut gs = required_tool_grammar();
    assert!(gs.accept_token(TOOL_CALL));
    gs.fill_bitmask();
    (0..VOCAB as u32)
        .find(|&t| gs.is_token_allowed(t))
        .expect("the grammar continues after <tool_call>")
}

/// 2026-10-21: a row of zeros with the given logits set.
pub(super) fn row(hot: &[(u32, f32)]) -> Vec<f32> {
    let mut r = vec![0.0f32; VOCAB];
    for &(id, v) in hot {
        r[id as usize] = v;
    }
    r
}

pub(super) fn model(rows: &[Vec<f32>]) -> Arc<PreemptStubModel> {
    Arc::new(PreemptStubModel {
        host_logits: Some(HostLogits::new(VOCAB, rows)),
        ..Default::default()
    })
}

/// 2026-10-21: the row copies (host fallbacks) the model has served.
pub(super) fn row_copies(m: &PreemptStubModel) -> usize {
    m.host_logits
        .as_ref()
        .unwrap()
        .row_copies
        .load(std::sync::atomic::Ordering::SeqCst)
}

/// 2026-10-21: a GLM-like serve over `m`: `thinking_steering` off (MODEL.toml
/// `[behavior]`), so a forced `</think>` is injected at once, and
/// `closed_by` as `SchedLimits::thinking_closed_by`.
pub(super) fn sched(m: Arc<PreemptStubModel>, closed_by: Option<u32>) -> SchedCtx {
    let mut s = SchedCtx::for_test_with(m);
    s.limits.thinking_closed_by = closed_by;
    s.watchdog.thinking_steering = false;
    s
}

/// 2026-10-21: the context `core/lane_decode.rs` builds for a step.
pub(super) fn ctx(s: &SchedCtx) -> LogitsContext<'_> {
    LogitsContext {
        watchdog: s.watchdog,
        scratch: &s.scratch,
        tel: &*s.io.tel,
        clock: &*s.io.clock,
        think_end_token: Some(THINK_END),
        think_start_token: Some(THINK_START),
        tool_call_start_token: Some(TOOL_CALL),
        tool_call_end_token: Some(TOOL_CALL_END),
        thinking_closed_by: s.limits.thinking_closed_by,
        verify_pos: 0,
        boundary_mask: None,
        mid_word_mask: None,
        sampling: s.levers.sampling(),
    }
}

/// 2026-10-21: an unfinished greedy sequence with the toy ids, `min_tokens`
/// met and no LZ penalty (the test fixture's default would rule every fast
/// path out; both paths apply the same penalties either way).
pub(super) fn seq() -> ActiveSeq {
    let (mut a, _rx) = test_seq(Vec::new(), 5000, None, 10);
    a.finished = false;
    a.min_tokens = 0;
    a.eos_tokens = vec![TOY_EOS];
    a.lz_penalty = 0.0;
    a.think_end_token = Some(THINK_END);
    a.think_start_token = Some(THINK_START);
    a.tool_call_start_token = Some(TOOL_CALL);
    a.tool_call_end_token = Some(TOOL_CALL_END);
    a
}

/// 2026-10-21: [`seq`] inside a thinking block the prompt opened.
pub(super) fn thinking_seq() -> ActiveSeq {
    let mut a = seq();
    a.inside_thinking = true;
    a.enable_thinking = true;
    a
}
