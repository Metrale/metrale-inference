// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: Tool calls in a completed GLM-4.7 answer: every
//! `<tool_call>...</tool_call>` envelope, judged by `judge`, and the text
//! outside them as content. Only this envelope is a call: other formats'
//! markup is content.
//!
//! Owner: server (tool parsing).
//! Invariants:
//! - Each envelope yields exactly one `Verdict`, in output order; an envelope
//!   the output cut short (no `</tool_call>`) is judged on what was written.

use super::super::ToolDefinition;
use super::judge::{Verdict, judge};

pub(crate) const TOOL_CALL: &str = "<tool_call>";
pub(crate) const TOOL_CALL_END: &str = "</tool_call>";

/// 2026-10-08: `(content, verdicts)` for the answer `text` (the reasoning
/// already split off). With at least one call, the content is trimmed and an
/// empty one is `None`; with none, `text` is returned unchanged.
pub fn parse_glm47_answer(text: &str, tools: &[ToolDefinition]) -> (Option<String>, Vec<Verdict>) {
    let mut content = String::new();
    let mut verdicts = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find(TOOL_CALL) {
        content.push_str(&rest[..open]);
        let body_and_tail = &rest[open + TOOL_CALL.len()..];
        match body_and_tail.find(TOOL_CALL_END) {
            Some(close) => {
                verdicts.push(judge(&body_and_tail[..close], tools));
                rest = &body_and_tail[close + TOOL_CALL_END.len()..];
            }
            None => {
                verdicts.push(judge(body_and_tail, tools));
                rest = "";
            }
        }
    }
    content.push_str(rest);
    if verdicts.is_empty() {
        return (Some(content).filter(|c| !c.is_empty()), verdicts);
    }
    let trimmed = content.trim();
    ((!trimmed.is_empty()).then(|| trimmed.to_string()), verdicts)
}
