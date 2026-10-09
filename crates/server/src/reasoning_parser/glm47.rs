// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The GLM-4.7 reasoning format, used by GLM-5.3 (`glm5_next`).
//! The chat template ends every generation prompt with `<think>`, with
//! thinking on or off (off means low effort, not no reasoning), so output
//! starts inside the block on every request the prompt opened it for. The
//! block ends at `</think>`, and also at `<tool_call>`: a call opened while
//! reasoning ends the reasoning there, and the call and what follows are the
//! answer.
//!
//! Splitting is a small state machine over the markers:
//!
//! | state | `<think>` | `</think>` | `<tool_call>` | `</tool_call>` |
//! |---|---|---|---|---|
//! | reasoning | dropped | to answer | to call (kept) | text |
//! | answer | to reasoning | dropped | to call (kept) | text |
//! | call | text | text | text | to answer (kept) |
//!
//! Inside a call every marker is text, so a call that swallowed a `</think>`
//! reaches the tool parser intact (`tool_parser/glm47/recover.rs` reads it).
//!
//! Owner: server.
//! Invariants:
//! - Every byte of the output outside the dropped markers lands in exactly
//!   one of reasoning and answer, in order.

use super::ReasoningParser;

const THINK: &str = "<think>";
const THINK_END: &str = "</think>";
const TOOL_CALL: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";

/// 2026-10-08: The GLM-4.7 reasoning parser (module doc).
pub(super) struct Glm47ReasoningParser;

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Reasoning,
    Answer,
    Call,
}

impl ReasoningParser for Glm47ReasoningParser {
    fn name(&self) -> &str {
        "glm47"
    }
    fn start_tag(&self) -> &str {
        THINK
    }
    fn end_tag(&self) -> &str {
        THINK_END
    }
    fn tool_call_closes_reasoning(&self) -> bool {
        true
    }

    /// 2026-10-08: Split `text` with the module's state machine, starting in
    /// the reasoning when `enable_thinking` (the prompt opened the block) and
    /// in the answer otherwise. The reasoning is returned whatever
    /// `enable_thinking` says, with trailing whitespace removed, and `None`
    /// when empty; the answer is returned as written.
    fn extract_thinking(&self, text: &str, enable_thinking: bool) -> (Option<String>, String) {
        let mut state = if enable_thinking {
            State::Reasoning
        } else {
            State::Answer
        };
        let mut reasoning = String::new();
        let mut answer = String::new();
        let mut rest = text;
        while !rest.is_empty() {
            let markers: &[&str] = match state {
                State::Call => &[TOOL_CALL_END],
                _ => &[THINK, THINK_END, TOOL_CALL],
            };
            let Some((at, marker)) = markers
                .iter()
                .filter_map(|m| rest.find(m).map(|at| (at, *m)))
                .min_by_key(|&(at, _)| at)
            else {
                sink(state, &mut reasoning, &mut answer).push_str(rest);
                break;
            };
            sink(state, &mut reasoning, &mut answer).push_str(&rest[..at]);
            state = match (state, marker) {
                (State::Call, _) => {
                    answer.push_str(marker);
                    State::Answer
                }
                (_, TOOL_CALL) => {
                    answer.push_str(marker);
                    State::Call
                }
                (_, THINK) => State::Reasoning,
                (_, _) => State::Answer,
            };
            rest = &rest[at + marker.len()..];
        }
        let reasoning = reasoning.trim_end();
        (
            (!reasoning.is_empty()).then(|| reasoning.to_string()),
            answer,
        )
    }
}

fn sink<'a>(state: State, reasoning: &'a mut String, answer: &'a mut String) -> &'a mut String {
    match state {
        State::Reasoning => reasoning,
        State::Answer | State::Call => answer,
    }
}
