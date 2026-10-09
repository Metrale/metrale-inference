// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The end of the reasoning phase in `handle_token`, shared by the two
//! tokens that end it: `</think>`, and `<tool_call>` under a format whose tool call
//! closes the reasoning.
//!
//! Owner: server streaming API.
//! Invariants:
//! - On return `state.thinking_done` is set and the decode state is empty, so the next
//!   token decodes as the first of the answer.

use crate::ir::StreamDelta;

use super::super::ctx::StreamCtx;
use super::super::state::StreamState;
use super::DeltaVec;

/// 2026-09-26: Close the reasoning at the last token in `state.all_toks`, which ends it
/// and is not part of it: emit the reasoning not yet streamed and the tail the reasoning
/// sanitizer held back, reset the detector, and clear the decode state.
pub(super) fn close_reasoning(state: &mut StreamState, ctx: &StreamCtx, deltas: &mut DeltaVec) {
    state.thinking_done = true;
    // 2026-09-26: Emit only the reasoning bytes past `state.emitted` (for example a
    // held-back incomplete UTF-8 tail); the rest was already streamed.
    if ctx.enable_thinking && state.all_toks.len() > 1 {
        let full = ctx
            .state
            .tokenizer
            .decode(&state.all_toks[..state.all_toks.len() - 1])
            .unwrap_or_default();
        let stable = full.trim_end_matches('\u{FFFD}');
        if stable.len() > state.emitted {
            let residual = &stable[state.emitted..];
            // 2026-09-26: A whitespace-only residual is real text; skip only an
            // empty one.
            if !residual.is_empty() {
                deltas.push(StreamDelta::Reasoning {
                    text: residual.to_string(),
                    token_ids: state.take_ids_if(ctx.req_return_token_ids),
                });
            }
        }
    }
    // 2026-09-26: Flush the tail the reasoning sanitizer held back, unless it is
    // suppressing a leak.
    if !state.reasoning_suppressing_leak && !state.reasoning_tag_scan_buf.is_empty() {
        let tail = std::mem::take(&mut state.reasoning_tag_scan_buf);
        if !tail.is_empty() {
            deltas.push(StreamDelta::Reasoning {
                text: tail,
                token_ids: Vec::new(),
            });
        }
    }
    if let Some(ref mut det) = state.detector {
        det.reset();
    }
    state.emitted = 0;
    state.all_toks.clear();
    state.content_decoded.clear();
    state.detok_prefix_offset = 0;
    state.detok_read_offset = 0;
}
