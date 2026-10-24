// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the blocking chat path's token split, stop-token trim and
//! tool-call hoisting from reasoning.
//!
//! Owner: server chat API.
//! Invariants: none beyond the types.

use super::{
    extract_hoisted_tool_calls, merge_hoisted_tool_calls, output_tokens_without_stop,
    split_at_first_think_end,
};
use crate::tool_parser::{FunctionCall, ToolCall};

fn tool_call(id: &str, city: &str) -> ToolCall {
    ToolCall {
        id: id.into(),
        call_type: "function".into(),
        function: FunctionCall {
            name: "get_weather".into(),
            arguments: format!(r#"{{"city":"{city}"}}"#),
        },
    }
}

#[test]
fn duplicate_call_across_reasoning_and_content_is_emitted_once() {
    let merged = merge_hoisted_tool_calls(
        vec![tool_call("reasoning", "Boston")],
        vec![tool_call("content", "Boston")],
    );

    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].id, "content");
}

#[test]
fn distinct_calls_across_reasoning_and_content_are_preserved() {
    let merged = merge_hoisted_tool_calls(
        vec![tool_call("reasoning", "Boston")],
        vec![tool_call("content", "Seattle")],
    );

    assert_eq!(merged.len(), 2);
}

#[test]
fn blocking_decode_excludes_terminal_stop_token() {
    let tokens = [10, 11, 24];

    assert_eq!(output_tokens_without_stop(&tokens, "stop"), &[10, 11]);
    assert_eq!(output_tokens_without_stop(&tokens, "length"), &tokens);
}

#[test]
fn blocking_thinking_split_uses_first_end_token() {
    let tokens = [10, 19, 20, 19, 30];

    let split = split_at_first_think_end(&tokens, 19, true);

    assert_eq!(split, Some((&[10][..], &[20, 19, 30][..])));
    assert_eq!(split_at_first_think_end(&tokens, 19, false), None);
}

#[test]
fn poolside_tool_call_in_reasoning_is_not_hoisted() {
    let reasoning = "plan <tool_call>write_file<arg_key>path</arg_key>\
        <arg_value>/tmp/x</arg_value></tool_call> more";

    let (preserved, calls) = extract_hoisted_tool_calls(Some(reasoning), Some("poolside_v1"));

    assert_eq!(preserved.as_deref(), Some(reasoning));
    assert!(calls.is_empty());
}

#[test]
fn non_poolside_tool_call_in_reasoning_is_still_hoisted() {
    let reasoning =
        r#"plan <tool_call>{"name":"get_weather","arguments":{"city":"Boston"}}</tool_call>"#;

    let (_scrubbed, calls) = extract_hoisted_tool_calls(Some(reasoning), Some("hermes"));

    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
}

/// 2026-10-13: The GLM blocking split as `build_choice_message` makes it: the
/// GLM-4.7 parser with the prompt-opened `<think>`, then vLLM's content rule.
fn glm_split(text: &str, forced_tool_choice: bool) -> (Option<String>, Option<String>) {
    let parser = crate::reasoning_parser::ReasoningFormat::Glm47.into_parser();
    assert!(parser.tool_call_closes_reasoning());
    let (reasoning, answer) = parser.extract_thinking(text, true);
    (
        reasoning,
        super::super::chat_blocking_choice::vllm_split_content(Some(answer), forced_tool_choice),
    )
}

#[test]
fn glm_unclosed_think_is_all_reasoning_with_null_content() {
    // 2026-10-13: vLLM's GLM53 parser starts in reasoning; with no `</think>`
    // the whole output is reasoning and the content is null.
    assert_eq!(glm_split("4", false), (Some("4".into()), None));
    assert_eq!(
        glm_split("The answer is 4.\n", false),
        (Some("The answer is 4.".into()), None)
    );
}

#[test]
fn glm_closed_think_splits_reasoning_from_content() {
    assert_eq!(
        glm_split("Let me think.</think>The answer is 4.", false),
        (
            Some("Let me think.".into()),
            Some("The answer is 4.".into())
        )
    );
}

#[test]
fn glm_empty_think_has_null_reasoning() {
    // 2026-10-13: An empty reasoning block is `None` in vLLM, not `""`.
    assert_eq!(
        glm_split("</think>The answer is 4.", false),
        (None, Some("The answer is 4.".into()))
    );
}

#[test]
fn glm_forced_tool_choice_sends_empty_string_content() {
    // 2026-10-13: Under `tool_choice` required or named, vLLM sends `""`
    // where it would otherwise send null, and keeps non-empty content.
    assert_eq!(
        glm_split("4", true),
        (Some("4".into()), Some(String::new()))
    );
    assert_eq!(
        glm_split("x</think>done", true),
        (Some("x".into()), Some("done".into()))
    );
}
