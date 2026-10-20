// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Whitespace shaping of the streamed deltas, as vLLM's parser
//! engine does it (`ParserEngine._events_to_delta` and
//! `_strip_trailing_reasoning`, with the default config: trailing reasoning
//! whitespace stripped, whitespace-only content dropped before tool calls).
//! Applied to a reasoning format that asks for it
//! (`ReasoningParser::shapes_stream_whitespace`).
//!
//! - Reasoning: trailing whitespace is held back until more reasoning
//!   follows. It is dropped when anything else arrives, and at the finish.
//! - Content: until it has a non-whitespace character, whitespace-only
//!   content is held back. It goes out in front of the first non-whitespace
//!   content, is dropped once a tool call has started, and goes out at the
//!   finish when there is no tool call.
//!
//! The token ids of a held-back or dropped delta move to the next delta that
//! goes out, so the stream still carries every streamed id.
//!
//! Owner: server streaming API.
//! Invariants:
//! - The concatenated reasoning text is vLLM's: the input's, right-trimmed,
//!   with inner whitespace kept.
//! - No delta that goes out has empty text.

use crate::ir::StreamDelta;

/// 2026-10-14: The held-back state of one stream (module doc).
#[derive(Debug, Default)]
pub(super) struct VllmDeltaShape {
    held_reasoning: String,
    held_content: String,
    content_has_text: bool,
    tool_seen: bool,
    carried_ids: Vec<u32>,
}

impl VllmDeltaShape {
    /// 2026-10-14: Shape one batch of deltas, in order.
    pub(super) fn apply(&mut self, deltas: Vec<StreamDelta>) -> Vec<StreamDelta> {
        let mut out = Vec::with_capacity(deltas.len());
        for delta in deltas {
            self.shape(delta, &mut out);
        }
        out
    }

    fn shape(&mut self, delta: StreamDelta, out: &mut Vec<StreamDelta>) {
        match delta {
            StreamDelta::Reasoning { text, token_ids } => {
                let combined = std::mem::take(&mut self.held_reasoning) + &text;
                let trimmed = combined.trim_end();
                self.held_reasoning = combined[trimmed.len()..].to_string();
                let token_ids = self.with_carried(token_ids);
                if trimmed.is_empty() {
                    self.carried_ids = token_ids;
                } else {
                    out.push(StreamDelta::Reasoning {
                        text: trimmed.to_string(),
                        token_ids,
                    });
                }
            }
            StreamDelta::Content { text, token_ids } => {
                self.held_reasoning.clear();
                let token_ids = self.with_carried(token_ids);
                if self.content_has_text {
                    out.push(StreamDelta::Content { text, token_ids });
                    return;
                }
                let combined = std::mem::take(&mut self.held_content) + &text;
                if !combined.trim().is_empty() {
                    self.content_has_text = true;
                    out.push(StreamDelta::Content {
                        text: combined,
                        token_ids,
                    });
                } else {
                    if !self.tool_seen {
                        self.held_content = combined;
                    }
                    self.carried_ids = token_ids;
                }
            }
            StreamDelta::ToolCallStart { .. } => {
                self.held_reasoning.clear();
                self.held_content.clear();
                self.tool_seen = true;
                out.push(delta);
            }
            StreamDelta::ToolCallArgs {
                index,
                fragment,
                token_ids,
            } => {
                let token_ids = self.with_carried(token_ids);
                out.push(StreamDelta::ToolCallArgs {
                    index,
                    fragment,
                    token_ids,
                });
            }
            StreamDelta::Finish {
                reason,
                usage,
                token_ids,
            } => {
                self.held_reasoning.clear();
                let held = std::mem::take(&mut self.held_content);
                if !held.is_empty() && !self.tool_seen {
                    out.push(StreamDelta::Content {
                        text: held,
                        token_ids: std::mem::take(&mut self.carried_ids),
                    });
                }
                let token_ids = self.with_carried(token_ids);
                out.push(StreamDelta::Finish {
                    reason,
                    usage,
                    token_ids,
                });
            }
            StreamDelta::Prompt { .. } => out.push(delta),
            StreamDelta::Refusal { .. } | StreamDelta::Error { .. } => out.push(delta),
        }
    }

    fn with_carried(&mut self, token_ids: Vec<u32>) -> Vec<u32> {
        let mut ids = std::mem::take(&mut self.carried_ids);
        ids.extend(token_ids);
        ids
    }
}
