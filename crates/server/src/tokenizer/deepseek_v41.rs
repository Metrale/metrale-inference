// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: DeepSeek-V4.1 message encoding in Rust, ported from the publisher's
//! `encoding/encoding.py`. `ChatTokenizer` uses it instead of a Jinja template for model
//! type `deepseek_v41`; the tests compare it with the publisher's reference outputs in
//! test_data/deepseek_v41_encoding.
//!
//! V4.1 differs from V4 in four ways the encoder owns: DSML tag names carry a leading
//! space (`<｜DSML｜ calls>`), the reasoning effort is a numeric budget behind the
//! `<｜System｜>` token, a system message after index 0 gets its own `<｜System｜>` token
//! and the assistant header like a user turn, and tool names may be namespaced
//! (`ns::name`).
//!
//! Owner: server (tokenizer).
//! Invariants: the effort is validated before any message renders, in chat mode too, as
//! the publisher does; an unknown effort, role or task is an error, never a fallback.

use anyhow::{Result, bail};
use serde_json::Value;

mod preprocess;
mod render;

pub(super) const SYSTEM: &str = "<｜System｜>";
pub(super) const IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";

/// 2026-10-18: The publisher's numeric reasoning budget (range 1-100). Requests name a
/// tier; `low`, `high` and `max` map to 50, 75 and 100, and no effort means `high`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ReasoningBudget(u8);

impl ReasoningBudget {
    const DEFAULT_TIER: &'static str = "high";

    /// 2026-10-18: `xhigh` is the engine's wire spelling of `max`
    /// (`ir::ReasoningEffort::Max`). `medium` has no publisher tier and is rejected, as
    /// the publisher rejects it.
    pub(super) fn from_tier(tier: Option<&str>) -> Result<Self> {
        Ok(Self(match tier.unwrap_or(Self::DEFAULT_TIER) {
            "low" => 50,
            "high" => 75,
            "max" | "xhigh" => 100,
            other => bail!(
                "invalid DeepSeek-V4.1 reasoning effort '{other}': expected one of low, high, max"
            ),
        }))
    }

    pub(super) fn prompt(self) -> String {
        format!(
            "Reasoning Effort: {} (range 1-100, the higher the value, the more thorough the reasoning)\n\n",
            self.0
        )
    }
}

/// 2026-10-18: Encode OpenAI-format `messages` into the V4.1 prompt string. Request-level
/// `tools` go on a leading system message (inserted when the conversation has none) unless
/// a message already carries tools. `enable_thinking` selects the publisher's
/// `thinking_mode`; earlier turns' reasoning is dropped unless any message has tools.
pub(super) fn encode_messages(
    messages: &[Value],
    tools: Option<&[Value]>,
    enable_thinking: bool,
    reasoning_effort: Option<&str>,
) -> Result<String> {
    let budget = ReasoningBudget::from_tier(reasoning_effort)?;
    let mut messages = messages.to_vec();
    preprocess::attach_top_level_tools(&mut messages, tools)?;
    for message in &messages {
        preprocess::reject_image_placeholder(message)?;
    }
    let mut messages = preprocess::merge_tool_messages(&messages)?;
    super::deepseek_v4::sort_tool_results_by_call_order(&mut messages);
    let mut messages = preprocess::replace_image_blocks(messages)?;

    let drop_thinking = !messages.iter().any(|m| m.get("tools").is_some_and(truthy));
    if enable_thinking && drop_thinking {
        messages = preprocess::drop_thinking_messages(&messages);
    }
    render::render_messages(&messages, enable_thinking, drop_thinking, budget)
}

/// 2026-10-18: Python truthiness of a JSON value; the publisher gates on it throughout.
pub(super) fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

pub(super) fn role(message: &Value) -> Option<&str> {
    message.get("role").and_then(Value::as_str)
}

/// 2026-10-18: The index of the last message that starts a turn: a user message, or a
/// system message after index 0. A user message whose blocks hold no text or image (only
/// tool results) does not start one.
pub(super) fn last_user_index(messages: &[Value]) -> Option<usize> {
    messages.iter().enumerate().rposition(|(index, message)| {
        let starts_turn = match role(message) {
            Some("user") => true,
            Some("system") => index > 0,
            _ => false,
        };
        starts_turn
            && !message
                .get("content_blocks")
                .and_then(Value::as_array)
                .is_some_and(|blocks| {
                    !blocks.is_empty()
                        && !blocks.iter().any(|block| {
                            matches!(
                                block.get("type").and_then(Value::as_str),
                                Some("text" | "image" | "image_url")
                            )
                        })
                })
    })
}
