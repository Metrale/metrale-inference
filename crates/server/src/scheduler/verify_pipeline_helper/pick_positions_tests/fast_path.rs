// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: A143, A144 and A144b tests of `verify_pick_all_with_pipeline`
//! and `pick_positions_from_host`: the post-think guard on the verify fast
//! paths, `logit_bias` parity with decode, and decode's last-wins tie break.
//! Moved out of `pick_positions_tests.rs` unchanged to keep it under the
//! file-size cap; the fixtures and `FastPathStubModel` stay there.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

// 2026-10-04: post-think structural guard on the fast paths (A143 sibling).
// `verify_pick_all_with_pipeline`'s grammar and grammarless GPU-argmax fast
// paths used to return `argmax_ids` with no post-think check, unlike the
// host path (`pick_positions_from_host`, tested in `pick_positions_tests.rs`),
// which always runs `PostCloseThinkMask`. These tests drive the real
// `verify_pick_all_with_pipeline` entry point, so a regression in the
// guard's wiring, not only in the guard function, fails a test.

#[test]
fn verify_fast_path_bails_on_reopened_think_end_after_think_ended() {
    // 2026-09-29: the single-position GPU argmax reopens `</think>`, with
    // `HELLO` the runner-up. With `think_ended` and `fast_greedy_chat` on,
    // the grammarless fast path would, without the guard, return
    // `THINK_END` with no D2H at all.
    let mut a = post_think_grammarless_seq();
    let model = FastPathStubModel::new(VOCAB, &[row(&[(THINK_END, 10.0), (HELLO, 5.0)])]);
    let argmax_ids = [THINK_END];
    let picks = with_ctx_fast_greedy_chat(|ctx| {
        verify_pick_all_with_pipeline(&model, &argmax_ids, &mut a, ctx, 0)
    });
    assert_eq!(
        picks,
        vec![HELLO],
        "the post-think structural guard must force the host path so \
         PostCloseThinkMask masks the reopened </think> and the runner-up \
         wins, instead of the fast path returning the raw GPU argmax"
    );
}

#[test]
fn verify_fast_path_takes_the_fast_path_when_no_structural_hit() {
    // 2026-09-29: control: an ordinary content argmax must not force the
    // host path. The stub holds no rows, so a D2H would panic; passing
    // proves the fast path is reachable in this fixture and the guarded
    // test above is not passing through some other ineligibility.
    let mut a = post_think_grammarless_seq();
    let model = FastPathStubModel::new(VOCAB, &[]);
    let argmax_ids = [HELLO];
    let picks = with_ctx_fast_greedy_chat(|ctx| {
        verify_pick_all_with_pipeline(&model, &argmax_ids, &mut a, ctx, 0)
    });
    assert_eq!(picks, vec![HELLO]);
}

// 2026-09-29: A144: the speculative paths apply the same `logit_bias` as
// decode. A tools-active request carries `(<tool_call>, +3.0)` in
// `ActiveSeq.logit_bias` (`sampling_setup`). Decode applied it; verify passed
// an empty bias, so spec-on picked the raw prose token where spec-off opened
// a call.

/// 2026-09-29: post-think, grammarless, greedy, tools-active.
/// `repetition_penalty` 1.05 keeps decode on the host pipeline for this row
/// (the `think_ended` device-argmax admission needs neutral penalties), so
/// decode applies the bias.
fn tools_present_seq() -> ActiveSeq {
    let mut a = post_think_grammarless_seq();
    a.min_tokens = 0;
    a.repetition_penalty = 1.05;
    a.tool_call_start_token = Some(TOOL_CALL_OPEN);
    a.tool_call_end_token = Some(TOOL_CALL_CLOSE);
    a.logit_bias = vec![(TOOL_CALL_OPEN, 3.0)];
    a
}

/// 2026-09-29: `<tool_call>` 2.0 below the prose argmax: +3.0 flips it, 0.0
/// does not.
fn opener_near_miss_row() -> Vec<f32> {
    row(&[(HELLO, 10.0), (TOOL_CALL_OPEN, 8.0)])
}

#[test]
fn a144_verify_applies_the_same_bias_as_decode_at_a_tools_present_position() {
    use crate::scheduler::sample_step::{
        PositionKind, penalty_params_for, speculative_base_logit_bias,
    };
    let floor = crate::scheduler::helpers::WatchdogParams::default().min_reasoning_floor;
    let mut a = tools_present_seq();
    // 2026-09-29: parameter-level parity: the Verify params carry exactly
    // decode's bias.
    let decode = penalty_params_for(
        &a,
        PositionKind::FinalDecode,
        0.0,
        None,
        a.logit_bias.clone(),
        floor,
    );
    let verify_bias = speculative_base_logit_bias(&a, 0, Some(THINK_END), true, || {
        unreachable!("a host-regime row never probes the raw argmax")
    });
    let verify = penalty_params_for(&a, PositionKind::Verify, 0.0, None, verify_bias, floor);
    assert_eq!(verify.logit_bias, decode.logit_bias);
    assert_eq!(verify.logit_bias, vec![(TOOL_CALL_OPEN, 3.0)]);

    // 2026-09-29: end to end through the host path (fast paths off in
    // `with_ctx`).
    let model = FastPathStubModel::new(VOCAB, &[opener_near_miss_row()]);
    let picks = with_ctx(|ctx| verify_pick_all_with_pipeline(&model, &[HELLO], &mut a, ctx, 0));
    assert_eq!(
        picks,
        vec![TOOL_CALL_OPEN],
        "verify must pick what decode picks: HELLO 10.0 < <tool_call> 8.0 + 3.0"
    );
}

#[test]
fn a144_opener_bias_is_stripped_per_position_inside_a_tool_body() {
    // 2026-09-29: one window opens a call, stays in its body, closes it, then
    // sits at a fresh opener decision. The +3.0 must be off at position 1
    // (inside the body position 0 opened, else a spurious mid-body re-open)
    // and on again at position 3 (after position 2's `</tool_call>`). The
    // step-start state (outside a body) is wrong for positions 1 and 2.
    let mut a = tools_present_seq();
    let buf = bf16_rows(&[
        row(&[(TOOL_CALL_OPEN, 10.0)]),
        opener_near_miss_row(),
        row(&[(TOOL_CALL_CLOSE, 10.0)]),
        opener_near_miss_row(),
    ]);
    let picks = with_ctx(|ctx| pick_positions_from_host(&buf, VOCAB, 2, 4, &mut a, ctx));
    assert_eq!(
        picks,
        vec![TOOL_CALL_OPEN, HELLO, TOOL_CALL_CLOSE, TOOL_CALL_OPEN]
    );
    assert!(
        !a.inside_tool_body,
        "tool-body flag restored after the loop"
    );

    // 2026-09-29: starting inside a body: position 0 is stripped; after the
    // close the nudge returns.
    let mut a = tools_present_seq();
    a.inside_tool_body = true;
    let buf = bf16_rows(&[
        opener_near_miss_row(),
        row(&[(TOOL_CALL_CLOSE, 10.0)]),
        opener_near_miss_row(),
    ]);
    let picks = with_ctx(|ctx| pick_positions_from_host(&buf, VOCAB, 2, 3, &mut a, ctx));
    assert_eq!(picks, vec![HELLO, TOOL_CALL_CLOSE, TOOL_CALL_OPEN]);
    assert!(a.inside_tool_body, "tool-body flag restored after the loop");
}

#[test]
fn a144_fast_greedy_falls_back_to_host_when_bias_present() {
    // 2026-09-29: `fast_greedy_chat` on and the penalties reduce-only:
    // without the A144 guard the grammarless fast path returns the raw GPU
    // argmax (HELLO) with no D2H, never seeing the bias.
    let mut a = tools_present_seq();
    assert!(crate::scheduler::sample_step::speculative_bias_forces_host(
        &a, true
    ));
    let model = FastPathStubModel::new(VOCAB, &[opener_near_miss_row()]);
    let picks = with_ctx_fast_greedy_chat(|ctx| {
        verify_pick_all_with_pipeline(&model, &[HELLO], &mut a, ctx, 0)
    });
    assert_eq!(picks, vec![TOOL_CALL_OPEN]);

    // 2026-09-29: control: without a bias the same fixture takes the fast
    // path.
    let mut a = tools_present_seq();
    a.logit_bias.clear();
    assert!(!crate::scheduler::sample_step::speculative_bias_forces_host(&a, true));
    let picks = with_ctx_fast_greedy_chat(|ctx| {
        verify_pick_all_with_pipeline(&model, &[HELLO], &mut a, ctx, 0)
    });
    assert_eq!(picks, vec![HELLO]);
}

#[test]
fn a144_bias_skipped_exactly_where_decode_gpu_argmax_skips_it() {
    use crate::scheduler::sample_step::{
        speculative_base_logit_bias, speculative_bias_forces_host,
    };
    // 2026-09-29: neutral penalties, `think_ended`, greedy, no grammar:
    // decode admits the row to its device argmax and never applies
    // `logit_bias`. Parity with decode means verify must not apply it either.
    let mut a = tools_present_seq();
    a.repetition_penalty = 1.0;
    assert!(!speculative_bias_forces_host(&a, true));
    assert!(speculative_base_logit_bias(&a, 0, Some(THINK_END), true, || HELLO).is_empty());
    let model = FastPathStubModel::new(VOCAB, &[opener_near_miss_row()]);
    let picks = with_ctx(|ctx| verify_pick_all_with_pipeline(&model, &[HELLO], &mut a, ctx, 0));
    assert_eq!(
        picks,
        vec![HELLO],
        "decode's GPU argmax emits HELLO; so must verify"
    );

    // 2026-09-29: except when that device argmax lands on a post-think
    // `</think>`/`<think>`: decode then redoes the step on the host, bias
    // included.
    assert_eq!(
        speculative_base_logit_bias(&a, 0, Some(THINK_END), true, || THINK_END),
        vec![(TOOL_CALL_OPEN, 3.0)]
    );
    assert_eq!(
        speculative_base_logit_bias(&a, 0, Some(THINK_END), true, || THINK_START),
        vec![(TOOL_CALL_OPEN, 3.0)]
    );

    // 2026-09-29: a `min_tokens` floor keeps decode on the host until it is
    // met; the floor is judged at `output_len + verify_pos`.
    a.min_tokens = 2;
    assert_eq!(
        speculative_base_logit_bias(&a, 1, Some(THINK_END), true, || HELLO),
        vec![(TOOL_CALL_OPEN, 3.0)]
    );
    assert!(speculative_base_logit_bias(&a, 2, Some(THINK_END), true, || HELLO).is_empty());

    // 2026-09-29: temperature above 0 always runs decode's host sampler.
    a.min_tokens = 0;
    a.temperature = 0.7;
    assert!(speculative_bias_forces_host(&a, true));
}

// 2026-09-29: A144b: verify's final pick uses decode's tie-break. Decode's
// host greedy path (`greedy_pick_last_wins`) and this host path process the
// same dequantised logits (BF16 to F32, no extra rounding either side), so
// an exact tie on a quantised checkpoint is real and common. The host path's
// final argmax used to resolve ties to the first equal id while decode
// resolves them to the last, which produced the K3-vs-spec-off synonym swaps
// (54/60 divergent TEB transcripts at temperature 0).

#[test]
fn a144b_verify_exact_tie_matches_decodes_last_wins_tie_break() {
    // 2026-09-29: HELLO (104) and TOOL_CALL_OPEN (128) tie at the row max,
    // 9.0, which BF16 represents exactly, so the round trip cannot break the
    // tie by accident.
    let mut a = post_think_grammarless_seq();
    a.min_tokens = 0;
    let buf = bf16_rows(&[row(&[(HELLO, 9.0), (TOOL_CALL_OPEN, 9.0)])]);
    let picks = with_ctx(|ctx| pick_positions_from_host(&buf, VOCAB, 2, 1, &mut a, ctx));
    assert_eq!(
        picks,
        vec![TOOL_CALL_OPEN],
        "TOOL_CALL_OPEN (id 128) is the last of the two tied ids (104, 128); \
         decode's `greedy_pick_last_wins` must win here, not first-wins' HELLO"
    );
}
