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

/// 2026-10-13: Decide whether the EOS just sampled for `a` is held back.
///
/// `generated_before` counts the tokens generated before this EOS. The
/// decode path asks before pushing the token and the emit path after
/// pushing it, so each passes its own count.
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
    let constrained = !a.inside_thinking;
    let escaped = tool_eos_escape && a.tool_call_completed && !a.inside_tool_body;
    EosHold {
        min_tokens: generated_before < a.min_tokens,
        grammar: constrained
            && !escaped
            && crate::grammar::grammar_blocks_stop(a.grammar_state.as_mut(), &a.eos_tokens),
        required_tool_call: constrained && a.require_tool_call,
    }
}
