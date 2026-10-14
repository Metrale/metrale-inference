// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: Whether a sampled EOS ends the turn, decided the way vLLM
//! decides it. `decode_logits_step` and `emit_step` both ask [`eos_hold`].
//!
//! vLLM v1 has no coupling between EOS and the reasoning block. Its sampler
//! masks EOS for only two reasons: `min_tokens` not yet reached, and a
//! structured-output grammar whose matcher cannot stop. The grammar is not
//! applied inside the reasoning block (`enable_in_reasoning` is false), so
//! an EOS sampled inside `<think>` ends the request. The GLM-5.3 parser
//! (`glm47_moe.py`, GLM53-REASONING-ALWAYS) then reports the whole output
//! as reasoning and the content as null.
//!
//! Owner: scheduler.
//! Invariants:
//! - Nothing about the thinking state holds an EOS back. Inside thinking
//!   only `min_tokens` can.

use super::types::ActiveSeq;

/// 2026-10-13: The reasons a sampled EOS is held back. All false means it
/// ends the turn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct EosHold {
    /// 2026-10-13: fewer than `min_tokens` tokens came before this EOS
    /// (vLLM's `MinTokensLogitsProcessor`).
    pub min_tokens: bool,
    /// 2026-10-13: outside thinking, the grammar cannot stop at the
    /// matcher's position (vLLM's structured-output bitmask).
    pub grammar: bool,
    /// 2026-10-13: outside thinking, `tool_choice` requires a call, no
    /// grammar is attached, and no `<tool_call>` has opened yet. This is
    /// the stand-in for the grammar vLLM always attaches to such a request.
    pub required_tool_call: bool,
}

impl EosHold {
    /// 2026-10-13: Is the EOS held back for any reason?
    pub(crate) fn any(self) -> bool {
        self.min_tokens || self.grammar || self.required_tool_call
    }
}

/// 2026-10-13: The part of [`EosHold`] known before sampling, with no
/// grammar: `min_tokens` and the required call. Every pre-sample EOS mask
/// (the pipeline stage, the verify fast paths, the device-argmax gates, the
/// MTP and spec bootstraps, the first token) asks this one function.
/// The grammar part is the grammar bitmask, applied by its own stage.
pub(crate) fn presample_hold(
    generated_before: usize,
    min_tokens: usize,
    inside_thinking: bool,
    require_tool_call: bool,
) -> EosHold {
    EosHold {
        min_tokens: generated_before < min_tokens,
        grammar: false,
        required_tool_call: !inside_thinking && require_tool_call,
    }
}

/// 2026-10-13: [`presample_hold`] for `a` at `generated_before` tokens.
pub(crate) fn presample_hold_for(a: &ActiveSeq, generated_before: usize) -> EosHold {
    presample_hold(
        generated_before,
        a.min_tokens,
        a.inside_thinking,
        a.require_tool_call,
    )
}

/// 2026-10-13: The EOS ids to mask before sampling `a`'s next token at
/// `generated_before`: all of them while held, none otherwise.
pub(crate) fn eos_mask_ids(a: &ActiveSeq, generated_before: usize) -> &[u32] {
    if presample_hold_for(a, generated_before).any() {
        &a.eos_tokens
    } else {
        &[]
    }
}

/// 2026-10-13: Is `tok` an EOS that the pre-sample mask removes at
/// `generated_before`? A fast path that picked it from unmasked logits
/// must decline and let the masked path sample.
pub(crate) fn eos_masked_at(a: &ActiveSeq, tok: u32, generated_before: usize) -> bool {
    a.eos_tokens.contains(&tok) && presample_hold_for(a, generated_before).any()
}

/// 2026-10-13: Does any pick of a verify window, position `i` being output
/// token `output_tokens.len() + i`, land on an EOS the pre-sample mask
/// removes? The GPU-argmax fast paths decline then, and the host pipeline
/// (which masks) picks the window.
pub(crate) fn eos_masked_in_window(a: &ActiveSeq, picks: &[u32]) -> bool {
    let base = a.output_tokens.len();
    picks
        .iter()
        .enumerate()
        .any(|(i, &tok)| eos_masked_at(a, tok, base + i))
}

/// 2026-10-13: Decide whether the EOS just sampled for `a` is held back.
///
/// `generated_before` counts the tokens generated before this EOS. Both
/// token paths ask before they push anything, so it is
/// `output_tokens.len()`.
///
/// `tool_eos_escape` is `SchedLevers::tool_eos_escape`. Once a tool call has
/// completed, and outside a tool body, the grammar no longer holds EOS
/// back. vLLM's required-call grammar accepts a stop after one complete
/// call, and an `auto` turn has no grammar there at all.
///
/// The grammar is consulted (a bitmask fill) only outside thinking.
pub(crate) fn eos_hold(
    a: &mut ActiveSeq,
    generated_before: usize,
    tool_eos_escape: bool,
) -> EosHold {
    let escaped = tool_eos_escape && a.tool_call_completed && !a.inside_tool_body;
    EosHold {
        grammar: !a.inside_thinking
            && !escaped
            && crate::grammar::grammar_blocks_stop(a.grammar_state.as_mut(), &a.eos_tokens),
        ..presample_hold_for(a, generated_before)
    }
}

/// 2026-10-14: Record an EOS that ends `a`'s turn, as vLLM does: it is an
/// output token (in `output_tokens`, charged to `max_tokens`, not streamed),
/// and inside the reasoning block it counts as reasoning, since vLLM counts
/// every id before `</think>` as reasoning. Both token paths call this once
/// [`eos_hold`] lets the EOS through.
pub(crate) fn end_turn_at_eos(
    a: &mut ActiveSeq,
    tok: u32,
    logprobs: Option<crate::api::TokenLogprobs>,
) {
    if let Some(lp) = logprobs {
        a.logprobs_data.push(lp);
    }
    a.output_tokens.push(tok);
    if a.inside_thinking {
        a.thinking_tokens += 1;
    }
    crate::scheduler::emit_step::update_tool_param_state(a, tok);
    a.consume_generation_budget();
    a.finished = true;
}
