// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: A146 tests: the verify window and its commit match spec-off
//! decode inside `<think>`, plus the A146 review-fix tests. Moved out of
//! `pick_positions_tests.rs` unchanged to keep it under the file-size cap;
//! the shared fixtures and `FastPathStubModel` stay there.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

// 2026-09-29: A146, spec-in-think parity. With speculation inside `<think>`,
// every committed token must be the token spec-off decode would commit.
// Decode is "pipeline on the live state, then commit" once per token; here
// that is `pick_positions_from_host` with k=1 followed by `emit_token` (the
// k=1 window re-applies its own trail entry, which is exactly decode's
// pipeline-then-commit). The window path picks K positions first, then
// commits an accepted prefix. The tests drive both over the same synthetic
// rows and assert identical picks AND identical commit state.

/// 2026-09-29: a ``` stand-in (any in-vocab id that is neither structural
/// nor HELLO).
const FENCE: u32 = 96;

/// 2026-09-29: mid-reasoning and grammarless (post-think rows then decode
/// free-run).
fn thinking_grammarless_seq() -> ActiveSeq {
    let mut a = thinking_seq();
    a.grammar_state = None;
    a.min_tokens = 0;
    a
}

/// 2026-09-29: the token id marked mid-word (the `MidWordThinkEndMask`
/// input) in every `with_ctx_think` context.
const MID_WORD: u32 = 97;

/// 2026-09-29: [`with_ctx`] plus the tokenizer's ``` id, a boundary mask
/// marking `boundary_ids` and a mid-word mask marking [`MID_WORD`].
fn with_ctx_think<R>(boundary_ids: &[u32], f: impl FnOnce(&LogitsContext) -> R) -> R {
    let scratch = crate::scheduler::sched_ctx::DecodeScratch::default();
    let io = crate::scheduler::io::SchedIo::for_test();
    let mut mask = vec![false; VOCAB];
    for &id in boundary_ids {
        mask[id as usize] = true;
    }
    let mut mid = vec![false; VOCAB];
    mid[MID_WORD as usize] = true;
    let ctx = LogitsContext {
        scratch: &scratch,
        tel: &*io.tel,
        clock: &*io.clock,
        watchdog: crate::scheduler::helpers::WatchdogParams::default(),
        boundary_mask: Some(mask.into()),
        mid_word_mask: Some(mid.into()),
        sampling: SamplingLevers {
            think_ended_gpu_argmax: true,
            ..SamplingLevers::default()
        },
        think_end_token: Some(THINK_END),
        think_start_token: Some(THINK_START),
        tool_call_start_token: Some(TOOL_CALL_OPEN),
        tool_call_end_token: Some(TOOL_CALL_CLOSE),
        code_fence_token: Some(FENCE),
        verify_pos: 0,
    };
    f(&ctx)
}

fn sched_think() -> crate::scheduler::sched_ctx::SchedCtx {
    let mut s = crate::scheduler::sched_ctx::SchedCtx::for_test();
    s.limits.code_fence_token = Some(FENCE);
    s
}

/// 2026-09-29: the commit state every later pick depends on.
type CommitState = (Vec<u32>, bool, bool, u32, bool, u32, u32, bool, u32);

fn commit_state(a: &ActiveSeq) -> CommitState {
    (
        a.output_tokens.clone(),
        a.inside_thinking,
        a.think_ended,
        a.thinking_tokens,
        a.force_end_thinking,
        a.sentence_defer_count,
        a.consecutive_confident,
        a.in_code_fence,
        a.think_watchdog_fires,
    )
}

/// 2026-09-29: the spec-off reference: one position at a time, pipeline
/// then commit.
fn run_serial(mut a: ActiveSeq, rows: &[Vec<f32>], boundary: &[u32]) -> (Vec<u32>, ActiveSeq) {
    let sched = sched_think();
    let mut picks = Vec::new();
    for r in rows {
        let buf = bf16_rows(std::slice::from_ref(r));
        let p = with_ctx_think(boundary, |ctx| {
            pick_positions_from_host(&buf, VOCAB, 2, 1, &mut a, ctx)
        });
        crate::scheduler::emit_step::emit_token(&mut a, p[0], None, &sched);
        picks.push(p[0]);
    }
    (picks, a)
}

/// 2026-09-29: the spec path: windows of `rows`, committing the first
/// `n_commit` picks of each (the accepted prefix plus the bonus). Returns
/// every committed pick.
fn run_windows(
    mut a: ActiveSeq,
    windows: &[(&[Vec<f32>], usize)],
    boundary: &[u32],
) -> (Vec<u32>, ActiveSeq) {
    let sched = sched_think();
    let mut committed = Vec::new();
    for (rows, n_commit) in windows {
        let buf = bf16_rows(rows);
        let picks = with_ctx_think(boundary, |ctx| {
            pick_positions_from_host(&buf, VOCAB, 2, rows.len(), &mut a, ctx)
        });
        for &p in &picks[..*n_commit] {
            crate::scheduler::emit_step::emit_token(&mut a, p, None, &sched);
            committed.push(p);
        }
    }
    (committed, a)
}

fn thinking_grammarless_seq_with(budget: u32, thinking_tokens: u32) -> ActiveSeq {
    let mut a = thinking_grammarless_seq();
    a.thinking_budget = Some(budget);
    a.thinking_tokens = thinking_tokens;
    a
}

#[test]
fn spec_think_budget_arm_mid_window_forces_think_end_at_next_position() {
    // 2026-09-29: the budget is reached by the commit of position 0, so
    // spec-off injects `</think>` as the very next token (hard override:
    // thinking_tokens >= 3 * budget). The window must pick it at position 1
    // (a stale thinking_tokens count picked HELLO), so any draft continuing
    // the reasoning is rejected there and an accepted run never needs
    // truncating.
    let mut a = thinking_grammarless_seq_with(2, 5);
    let rows = vec![row(&[(HELLO, 10.0)]); 3];
    let picks = with_ctx_think(&[], |ctx| {
        pick_positions_from_host(&bf16_rows(&rows), VOCAB, 2, 3, &mut a, ctx)
    });
    assert_eq!(picks[..2], [HELLO, THINK_END]);
    // 2026-09-29: restored: the window only picks.
    assert_eq!(a.thinking_tokens, 5);
    assert!(!a.force_end_thinking && a.inside_thinking && a.output_tokens.is_empty());
    assert_eq!(a.spec_think_trail.len(), 3);

    let (serial_picks, serial) = run_serial(thinking_grammarless_seq_with(2, 5), &rows, &[]);
    let (win_picks, win) = run_windows(thinking_grammarless_seq_with(2, 5), &[(&rows, 3)], &[]);
    assert_eq!(win_picks, serial_picks);
    assert_eq!(commit_state(&win), commit_state(&serial));
}

#[test]
fn spec_think_code_fence_defers_injection_per_position() {
    // 2026-09-29: `</think>` is armed; FENCE and HELLO are sentence
    // boundaries. Position 0 opens a fence, so position 1 (previous token
    // FENCE, a boundary) must DEFER (in_code_fence); without per-position
    // fence tracking the window injected `</think>` inside the code block.
    // Position 2 closes the fence; position 3 (previous FENCE, fence closed)
    // injects.
    let mk = || {
        let mut a = thinking_grammarless_seq();
        a.force_end_thinking = true;
        a
    };
    let rows = vec![
        row(&[(FENCE, 10.0)]),
        row(&[(HELLO, 10.0)]),
        row(&[(FENCE, 10.0)]),
        row(&[(HELLO, 10.0)]),
    ];
    let boundary = [FENCE, HELLO];
    let mut a = mk();
    // 2026-09-29: a non-boundary previous token, so position 0 defers.
    a.output_tokens = vec![1];
    let picks = with_ctx_think(&boundary, |ctx| {
        pick_positions_from_host(&bf16_rows(&rows), VOCAB, 2, 4, &mut a, ctx)
    });
    assert_eq!(picks, vec![FENCE, HELLO, FENCE, THINK_END]);
    assert!(!a.in_code_fence, "fence state restored after the loop");

    let mut s0 = mk();
    s0.output_tokens = vec![1];
    let mut w0 = mk();
    w0.output_tokens = vec![1];
    let (serial_picks, serial) = run_serial(s0, &rows, &boundary);
    let (win_picks, win) = run_windows(w0, &[(&rows[..2], 2), (&rows[2..], 2)], &boundary);
    assert_eq!(serial_picks, picks);
    assert_eq!(win_picks, serial_picks);
    assert_eq!(commit_state(&win), commit_state(&serial));
    assert!(!serial.inside_thinking && !serial.in_code_fence);
}

#[test]
fn spec_think_partial_accept_does_not_leak_pipeline_accumulators() {
    // 2026-09-29: the F2 streak arms `force_end_thinking` inside the
    // pipeline. Window 1 (3 positions) arms it at position 2, but only
    // position 0 commits (rejection at 1). The window used to keep position
    // 2's accumulators (streak 60, armed), so window 2 injected `</think>`
    // one token early. Serial: HELLO, HELLO, `</think>` (F2 arms at step 2
    // and the previous HELLO is a boundary), then post-think HELLO.
    let mk = || {
        let mut a = thinking_grammarless_seq();
        a.thinking_tokens = 400;
        a.consecutive_confident = 57;
        a
    };
    let rows = vec![row(&[(HELLO, 10.0)]); 4];
    let boundary = [HELLO];
    let (serial_picks, serial) = run_serial(mk(), &rows, &boundary);
    assert_eq!(serial_picks, vec![HELLO, HELLO, THINK_END, HELLO]);
    let (win_picks, win) = run_windows(mk(), &[(&rows[..3], 1), (&rows[1..], 3)], &boundary);
    assert_eq!(win_picks, serial_picks);
    assert_eq!(commit_state(&win), commit_state(&serial));
}

#[test]
fn spec_think_loop_watchdog_parity_window_vs_serial() {
    // 2026-09-29: a period-4 reasoning loop. The thinking-loop watchdog must
    // arm at the same committed token on both paths (it used to run on
    // decode only), and the window must see it arm mid-window (history and
    // thinking_tokens advanced per position) so the injection lands on the
    // same position.
    let ids = [10u32, 11, 12, 13];
    let rows: Vec<Vec<f32>> = (0..72).map(|i| row(&[(ids[i % 4], 10.0)])).collect();
    let boundary = [13u32];
    let (serial_picks, serial) = run_serial(thinking_grammarless_seq(), &rows, &boundary);
    assert!(
        serial.think_watchdog_fires >= 1,
        "the thinking-loop watchdog fired on the serial path"
    );
    assert!(serial_picks.contains(&THINK_END));
    let windows: Vec<(&[Vec<f32>], usize)> = rows.chunks(3).map(|c| (c, c.len())).collect();
    let (win_picks, win) = run_windows(thinking_grammarless_seq(), &windows, &boundary);
    assert_eq!(win_picks, serial_picks);
    assert_eq!(commit_state(&win), commit_state(&serial));
}

#[test]
fn spec_think_mid_word_mask_reads_the_window_prev_token() {
    // 2026-09-29: "think-close one sentence late": the committed history
    // ends mid-word; position 0 finishes the sentence (HELLO stands in for
    // '.', not mid-word) and position 1's argmax is `</think>`. Spec-off
    // sees the previous token HELLO and closes. The window used to read the
    // STALE committed `output_tokens.last()` (mid-word) at position 1, mask
    // `</think>`, and keep reasoning with the runner-up.
    let mk = || {
        let mut a = thinking_grammarless_seq();
        a.thinking_tokens = 50;
        a.output_tokens = vec![MID_WORD];
        a
    };
    let rows = vec![
        row(&[(HELLO, 10.0)]),
        row(&[(THINK_END, 10.0), (TOOL_CALL_CLOSE, 9.0)]),
    ];
    let (serial_picks, serial) = run_serial(mk(), &rows, &[]);
    assert_eq!(serial_picks, vec![HELLO, THINK_END]);
    let (win_picks, win) = run_windows(mk(), &[(&rows, 2)], &[]);
    assert_eq!(win_picks, serial_picks);
    assert_eq!(commit_state(&win), commit_state(&serial));
}

#[test]
fn spec_think_penalty_history_includes_earlier_window_picks() {
    // 2026-09-29: with a history penalty armed, position i must be penalised
    // against picks 0..i-1 exactly as decode (which has committed them)
    // penalises it. Position 1: HELLO 10.0 against a 9.0 runner-up;
    // repetition penalty 2.0 on the already-picked HELLO flips it
    // (10/2 < 9).
    let mk = || {
        let mut a = thinking_grammarless_seq();
        a.thinking_tokens = 50;
        a.repetition_penalty = 2.0;
        a
    };
    let rows = vec![
        row(&[(HELLO, 10.0)]),
        row(&[(HELLO, 10.0), (TOOL_CALL_CLOSE, 9.0)]),
    ];
    let (serial_picks, serial) = run_serial(mk(), &rows, &[]);
    assert_eq!(serial_picks, vec![HELLO, TOOL_CALL_CLOSE]);
    let (win_picks, win) = run_windows(mk(), &[(&rows, 2)], &[]);
    assert_eq!(win_picks, serial_picks);
    assert_eq!(commit_state(&win), commit_state(&serial));
}

// 2026-09-29: A146 review fixes.

#[test]
fn fast_path_immunity_sees_earlier_window_picks() {
    // 2026-09-29: `ReduceOnly` (repetition penalty 1.05) grammarless fast
    // path, window argmax [X, X] with X new. Decode and the host path
    // penalise position 1 against position 0's X (9.25 / 1.05 < 9.0, so the
    // runner-up wins). The fast path used to test immunity against the
    // committed history only and returned [X, X].
    const X: u32 = 90;
    const Y: u32 = 91;
    let mk = || {
        let mut a = post_think_grammarless_seq();
        a.min_tokens = 0;
        a.repetition_penalty = 1.05;
        a.logit_bias.clear();
        a
    };
    let rows = [row(&[(X, 9.25)]), row(&[(X, 9.25), (Y, 9.0)])];
    let model = FastPathStubModel::new(VOCAB, &rows);
    let mut a = mk();
    let picks = with_ctx_fast_greedy_chat(|ctx| {
        verify_pick_all_with_pipeline(&model, &[X, X], &mut a, ctx, 0)
    });
    assert_eq!(
        picks,
        vec![X, Y],
        "the fast path must agree with the host path and decode"
    );
    let mut a = mk();
    let slow =
        with_ctx(|ctx| pick_positions_from_host(&bf16_rows(&rows), VOCAB, 2, 2, &mut a, ctx));
    assert_eq!(slow, vec![X, Y]);
}

#[test]
fn stale_trail_never_reaches_a_windowless_verify_commit() {
    // 2026-09-29: a partial accept leaves trail entries j+1..; a later verify
    // that returns from a fast path (no window) emits a token whose
    // (tok, out_len) can coincide with the stale entry. It must not be
    // applied.
    let mut a = post_think_grammarless_seq();
    a.min_tokens = 0;
    a.logit_bias.clear();
    let stale = crate::scheduler::think_commit::SpecThinkTrail {
        tok: HELLO,
        out_len: a.output_tokens.len(),
        consecutive_confident: 42,
        sentence_defer_count: 7,
        force_end_thinking: true,
    };
    a.spec_think_trail.push_back(stale);
    let model = FastPathStubModel::new(VOCAB, &[row(&[(HELLO, 10.0)])]);
    let picks = with_ctx_fast_greedy_chat(|ctx| {
        verify_pick_all_with_pipeline(&model, &[HELLO], &mut a, ctx, 0)
    });
    assert_eq!(picks, vec![HELLO]);
    crate::scheduler::emit_step::emit_token(&mut a, HELLO, None, &sched_think());
    assert_eq!(
        (
            a.consecutive_confident,
            a.sentence_defer_count,
            a.force_end_thinking
        ),
        (0, 0, false),
        "a stale trail entry leaked into a windowless commit"
    );
}

#[test]
fn self_spec_commits_token_0_before_picking_the_window() {
    // 2026-09-29: verify position 0 is the token AFTER token_0. The history
    // ends mid-word; token_0 (HELLO) finishes the word; position 0's argmax
    // is `</think>`. Spec-off commits HELLO, then sees the previous token
    // HELLO and closes. Picking the window before committing token_0 saw the
    // previous token MID_WORD and masked `</think>`.
    let mk = || {
        let mut a = thinking_grammarless_seq();
        a.thinking_tokens = 50;
        a.output_tokens = vec![MID_WORD];
        a
    };
    let rows = vec![
        row(&[(THINK_END, 10.0), (TOOL_CALL_CLOSE, 9.0)]),
        row(&[(HELLO, 10.0)]),
    ];
    let (serial_picks, serial) = run_serial(
        {
            let mut a = mk();
            crate::scheduler::emit_step::emit_token(&mut a, HELLO, None, &sched_think());
            a
        },
        &rows[..1],
        &[],
    );
    assert_eq!(serial_picks, vec![THINK_END]);

    let mut a = mk();
    let sched = sched_think();
    let buf = bf16_rows(&rows);
    let n = crate::scheduler::spec_step::self_spec_commit(&mut a, HELLO, &[HELLO], &sched, |a| {
        with_ctx_think(&[], |ctx| {
            pick_positions_from_host(&buf, VOCAB, 2, 2, a, ctx)
        })
    });
    assert_eq!(n, Some(0), "draft HELLO rejected by the forced-free close");
    assert_eq!(a.last_token, THINK_END);
    assert_eq!(commit_state(&a), commit_state(&serial));
}
