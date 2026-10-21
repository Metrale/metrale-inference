// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Rank 0's choice of the fused K=2 announcement's flags: the device verdict
//! (`verify_k2_device_step::identity_picks`) and the screened head (`METRALE_SCREEN_HEAD`,
//! model-engine `screen_head.rs`).
//!
//! A screened verify returns both argmaxes exactly but writes no logits; a host read of them
//! afterwards is recomputed exactly on rank 0 (`screen_head_lazy.rs`): a few logits cheaply, whole
//! rows at about two full-head passes. So correctness never depends on this choice; the choice
//! asks for the screen only where the picks are expected to read nothing or single logits, and
//! declines where they read whole rows on most steps:
//!
//! - temperature above 0 without `force_temp_zero` (the host path samples the rows);
//! - `top_logprobs` (the rows are copied every step);
//! - a grammar (rows whenever the argmax is disallowed);
//! - penalties the fast paths classify `Blocked` (repetition below 1, negative presence or
//!   frequency, LZ, DRY): the host pipeline reads the rows;
//! - an EOS hold (`min_tokens`, a required tool call);
//! - a pick that neither fast path takes before it sees the argmax: the masked chat path is off or
//!   one of its stages may act (`fast_masked::stages_may_act`: the F2 confidence stop, a forced
//!   or deferred `</think>`, the tool-call pin), and the grammarless greedy path is off or the
//!   sequence is inside thinking.
//!
//! It allows `logit_bias` (the tools-active `<tool_call>` bias on every tool request) and
//! reduce-only penalties: their checks read single logits.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;
use metrale_model_layers::speculative::{DEVICE_DRAFT, K2Flags};

/// 2026-10-21: The announcement's flags for a fused K=2 verify of `draft0` (row 1's token).
pub(super) fn k2_flags(
    fused: bool,
    draft0: u32,
    a: &ActiveSeq,
    ctx: &crate::scheduler::logit_processors::LogitsContext,
) -> K2Flags {
    K2Flags {
        device_verdict: fused
            && draft0 == DEVICE_DRAFT
            && super::verify_k2_device_step::identity_picks(a, ctx),
        screen_head: fused && screen_picks_ok(a, ctx),
    }
}

/// 2026-10-21: Whether `a`'s K=2 picks need nothing but the argmaxes and single logits (module
/// docs).
pub(super) fn screen_picks_ok(
    a: &ActiveSeq,
    ctx: &crate::scheduler::logit_processors::LogitsContext,
) -> bool {
    let penalties = crate::scheduler::sample_step::penalty_params_for(
        a,
        crate::scheduler::sample_step::PositionKind::Verify,
        0.0,
        None,
        a.logit_bias.clone(),
        ctx.watchdog.min_reasoning_floor,
    );
    screen_rule(ScreenInputs {
        greedy: a.temperature == 0.0 || ctx.sampling.force_temp_zero,
        logprobs: a.top_logprobs.is_some(),
        grammar: a.grammar_state.is_some(),
        penalties_blocked: crate::scheduler::fast_greedy::classify_penalties_except_bias(
            &penalties,
        ) == crate::scheduler::fast_greedy::PenaltyGate::Blocked,
        eos_held: crate::scheduler::eos_hold::presample_hold_for(a, a.output_tokens.len()).any(),
        masked_path: ctx.sampling.dflash_masked_verify
            && ctx.sampling.fast_masked
            && !ctx.sampling.adadec_diagnostic
            && !crate::scheduler::verify_pipeline_helper::fast_masked::stages_may_act(a, ctx, 2),
        chat_path: ctx.sampling.fast_greedy_chat && !a.inside_thinking,
    })
}

/// 2026-10-21: The inputs of [`screen_picks_ok`], so its rule is tested without a sequence.
pub(super) struct ScreenInputs {
    pub greedy: bool,
    pub logprobs: bool,
    pub grammar: bool,
    pub penalties_blocked: bool,
    pub eos_held: bool,
    /// 2026-10-21: The masked chat fast path runs for this window.
    pub masked_path: bool,
    /// 2026-10-21: The grammarless greedy fast path runs for this window.
    pub chat_path: bool,
}

pub(super) fn screen_rule(i: ScreenInputs) -> bool {
    i.greedy
        && !i.logprobs
        && !i.grammar
        && !i.penalties_blocked
        && !i.eos_held
        && (i.masked_path || i.chat_path)
}

#[cfg(test)]
#[path = "screen_picks_tests.rs"]
mod thinking_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn plain() -> ScreenInputs {
        ScreenInputs {
            greedy: true,
            logprobs: false,
            grammar: false,
            penalties_blocked: false,
            eos_held: false,
            masked_path: true,
            chat_path: true,
        }
    }

    /// 2026-10-21: A plain greedy request takes the screen, through either fast path.
    #[test]
    fn a_plain_greedy_request_takes_the_screen() {
        assert!(screen_rule(plain()));
        assert!(screen_rule(ScreenInputs {
            masked_path: false,
            ..plain()
        }));
        assert!(screen_rule(ScreenInputs {
            chat_path: false,
            ..plain()
        }));
    }

    /// 2026-10-21: Every case that reads whole logits rows declines it, each on its own.
    #[test]
    fn each_whole_row_reader_declines_the_screen() {
        let cases: [(&str, ScreenInputs); 6] = [
            (
                "sampling",
                ScreenInputs {
                    greedy: false,
                    ..plain()
                },
            ),
            (
                "logprobs",
                ScreenInputs {
                    logprobs: true,
                    ..plain()
                },
            ),
            (
                "grammar",
                ScreenInputs {
                    grammar: true,
                    ..plain()
                },
            ),
            (
                "blocked penalties",
                ScreenInputs {
                    penalties_blocked: true,
                    ..plain()
                },
            ),
            (
                "EOS hold",
                ScreenInputs {
                    eos_held: true,
                    ..plain()
                },
            ),
            (
                "no fast path",
                ScreenInputs {
                    masked_path: false,
                    chat_path: false,
                    ..plain()
                },
            ),
        ];
        for (name, i) in cases {
            assert!(!screen_rule(i), "{name} must decline the screen");
        }
    }
}
