// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Conversation construction: how one turn's outcome becomes the
//! next turn's history, and how that history is shortened past `HISTORY_BUDGET`.
//!
//! Owner: bench, agentic.
//! Invariants:
//! - `compact` rewrites message contents and never removes or reorders a
//!   message.

use serde_json::{Value, json};

use super::{HISTORY_BUDGET, LIVE_REASONING, LIVE_TOOL_RESULTS};

/// 2026-09-26: The `tool_call_id` for call `nth` of `turn`. It depends only on
/// those two numbers, not on the id the server returned. The assistant message
/// and the tool reply both take it from here, so they always pair.
pub(super) fn call_id(turn: usize, nth: usize) -> String {
    format!("call_{turn}_{nth}")
}

/// 2026-09-26: While the conversation is over `HISTORY_BUDGET` characters,
/// replace the oldest reasoning (all but the last `LIVE_REASONING`), then the
/// oldest tool results (all but the last `LIVE_TOOL_RESULTS`), with a marker
/// that states how many characters were elided. It never removes a message, so
/// every assistant tool call keeps its tool reply.
pub(super) fn compact(messages: &mut [Value]) {
    let size = |m: &Value| {
        m["content"].as_str().map_or(64, str::len)
            + m["reasoning_content"].as_str().map_or(0, str::len)
    };
    let mut total: usize = messages.iter().map(size).sum();
    // 2026-09-26: Old reasoning goes first. It is replaced by a marker, not
    // removed, so the message keeps a non-empty `reasoning_content`.
    let think: Vec<usize> = (0..messages.len())
        .filter(|i| messages[*i]["reasoning_content"].is_string())
        .collect();
    for &i in think
        .iter()
        .take(think.len().saturating_sub(LIVE_REASONING))
    {
        if total <= HISTORY_BUDGET {
            return;
        }
        let was = messages[i]["reasoning_content"]
            .as_str()
            .map_or(0, str::len);
        let marker = format!("[{was} characters of earlier reasoning elided]");
        total = total - was + marker.len();
        messages[i]["reasoning_content"] = Value::String(marker);
    }
    let tools: Vec<usize> = (0..messages.len())
        .filter(|i| messages[*i]["role"] == "tool")
        .collect();
    for &i in tools
        .iter()
        .take(tools.len().saturating_sub(LIVE_TOOL_RESULTS))
    {
        if total <= HISTORY_BUDGET {
            return;
        }
        let was = size(&messages[i]);
        let marker = format!("[{was} characters elided to stay inside the context window]");
        total = total - was + marker.len();
        messages[i]["content"] = Value::String(marker);
    }
}

/// 2026-10-17: `preserve_thinking` is the resolved `METRALE_AGENTIC_PRESERVE_THINKING`: echo
/// the turn's non-empty reasoning as `reasoning_content` (`request_body` sends the matching
/// `preserve_thinking: true` template kwarg).
pub(super) fn assistant_message(
    outcome: &crate::http::ChatOutcome,
    turn: usize,
    preserve_thinking: bool,
) -> Value {
    let calls: Vec<Value> = outcome
        .tool_calls
        .iter()
        .enumerate()
        .map(|(i, c)| {
            json!({"id": call_id(turn, i), "type": "function", "function": {"name": c.name,
                // 2026-09-26: Empty arguments are sent as `{}`, which is valid JSON.
                "arguments": if c.arguments.is_empty() { "{}" } else { &c.arguments }}})
        })
        .collect();
    let text = &outcome.text;
    let mut msg = json!({"role": "assistant", "tool_calls": calls,
        "content": if text.is_empty() { Value::Null } else { Value::String(text.clone()) }});
    if preserve_thinking && !outcome.reasoning.trim().is_empty() {
        msg["reasoning_content"] = Value::String(outcome.reasoning.clone());
    }
    msg
}
