// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The GLM-4.7 reasoning format: reasoning is tracked whatever the
//! request's thinking flag says, `<tool_call>` closes the block, and markers
//! inside a call are call text.
//!
//! Owner: server tests.
//! Invariants: none beyond the types.

use super::{ReasoningFormat, ReasoningParser};

fn glm() -> Box<dyn ReasoningParser> {
    ReasoningFormat::Glm47.into_parser()
}

fn split(text: &str, enable_thinking: bool) -> (Option<String>, String) {
    glm().extract_thinking(text, enable_thinking)
}

#[test]
fn glm5_next_maps_to_glm47_and_both_names_parse() {
    let defaults: toml::Value =
        toml::from_str(include_str!("../../tool_defaults.toml")).expect("tool_defaults parses");
    let name = defaults["reasoning"]["glm5_next"].as_str();
    assert_eq!(name, Some("glm47"));
    for n in ["glm47", "glm45", "GLM47"] {
        assert_eq!(n.parse::<ReasoningFormat>(), Ok(ReasoningFormat::Glm47));
    }
    assert!(glm().tool_call_closes_reasoning());
    assert_eq!(glm().name(), "glm47");
    assert!(
        !ReasoningFormat::Qwen
            .into_parser()
            .tool_call_closes_reasoning()
    );
}

#[test]
fn output_starts_inside_the_block_the_prompt_opened() {
    assert_eq!(
        split("Plan it.\n</think>\n\nHello", true),
        (Some("Plan it.".into()), "\n\nHello".into()),
        "reasoning loses trailing whitespace only; the answer is verbatim"
    );
    assert_eq!(
        split("still thinking", true),
        (Some("still thinking".into()), String::new())
    );
}

#[test]
fn reasoning_is_returned_even_when_the_request_had_thinking_off() {
    // 2026-10-08: A prompt that did not open the block: the model opened its own.
    assert_eq!(
        split("<think>brief</think>Hi", false),
        (Some("brief".into()), "Hi".into())
    );
    // 2026-10-08: Negative control: the qwen parser drops it with thinking off.
    let (qwen_reasoning, _) = ReasoningFormat::Qwen
        .into_parser()
        .extract_thinking("<think>brief</think>Hi", false);
    assert_eq!(qwen_reasoning, None);
}

#[test]
fn a_tool_call_closes_the_reasoning_and_opens_the_answer() {
    let call = "<tool_call>bash<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>";
    assert_eq!(
        split(&format!("Need a listing.{call}"), true),
        (Some("Need a listing.".into()), call.into())
    );
    // 2026-10-08: Negative control: the qwen parser keeps the call in the reasoning.
    let (_, qwen_answer) = ReasoningFormat::Qwen
        .into_parser()
        .extract_thinking(&format!("Need a listing.{call}"), true);
    assert_eq!(qwen_answer, "");
}

#[test]
fn markers_inside_a_call_are_call_text() {
    let text = "r1<tool_call>bash<arg_key>command</arg_key><arg_value>ls\nmore</think>\
                <tool_call>web_search<arg_key>query</arg_key><arg_value>q</arg_value></tool_call>tail";
    let (reasoning, answer) = split(text, true);
    assert_eq!(reasoning.as_deref(), Some("r1"));
    assert_eq!(
        answer,
        "<tool_call>bash<arg_key>command</arg_key><arg_value>ls\nmore</think>\
         <tool_call>web_search<arg_key>query</arg_key><arg_value>q</arg_value></tool_call>tail"
    );
}

#[test]
fn stray_markers_in_the_answer_are_dropped_or_reopen_reasoning() {
    assert_eq!(
        split("<think>a</think>b</think>c<think>d</think>e", true),
        (Some("ad".into()), "bce".into())
    );
}
