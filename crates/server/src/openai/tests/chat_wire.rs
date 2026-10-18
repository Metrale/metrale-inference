// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the chat-completions wire format: `token_ids` appears only when
//! set, and reasoning is serialized under one key. 2026-10-14: that key is
//! `reasoning_content`, or vLLM's `reasoning` for a `ReasoningWire::Vllm` format.
//!
//! Owner: server (OpenAI API layer) tests.
//! Invariants: none beyond the types.

use crate::openai::*;
use crate::reasoning_parser::ReasoningWire;

#[test]
fn token_ids_absent_by_default_keeps_wire_byte_identical() {
    let chunk = ChatCompletionChunk::content_chunk("m", "id", "hi".into());
    let json = serde_json::to_string(&chunk).unwrap();
    assert!(!json.contains("token_ids"), "default wire changed: {json}");
    let chunk =
        ChatCompletionChunk::content_chunk("m", "id", "hi".into()).with_token_ids(Vec::new());
    let json = serde_json::to_string(&chunk).unwrap();
    assert!(!json.contains("token_ids"));
}

#[test]
fn with_token_ids_stamps_first_choice() {
    let chunk =
        ChatCompletionChunk::content_chunk("m", "id", "hi".into()).with_token_ids(vec![10, 20, 30]);
    assert_eq!(chunk.choices[0].token_ids, vec![10, 20, 30]);
    let json = serde_json::to_string(&chunk).unwrap();
    assert!(json.contains("\"token_ids\":[10,20,30]"), "{json}");
    let usage = Usage {
        prompt_tokens: 1,
        completion_tokens: 1,
        total_tokens: 2,
        prompt_tokens_details: None,
        completion_tokens_details: None,
        time_to_first_token_ms: 0.0,
        response_tokens_per_second: 0.0,
        decode_time_ms: 0.0,
        total_time_ms: 0.0,
    };
    let chunk = ChatCompletionChunk::usage_only_chunk("m", "id", usage).with_token_ids(vec![1, 2]);
    assert!(chunk.choices.is_empty());
}

#[test]
fn reasoning_delta_emits_only_reasoning_content() {
    let chunk = ChatCompletionChunk::reasoning_chunk(
        "m",
        "id",
        ReasoningWire::ReasoningContent,
        "thinking".into(),
    );
    let json = serde_json::to_string(&chunk).unwrap();
    assert!(
        json.contains("\"reasoning_content\":\"thinking\""),
        "reasoning_content missing: {json}"
    );
    assert!(
        !json.contains("\"reasoning\":"),
        "mirror `reasoning` field leaked into stream delta: {json}"
    );
}

#[test]
fn blocking_message_emits_only_reasoning_content() {
    let msg = ChatMessage {
        role: "assistant".into(),
        reasoning: WireReasoning::message(ReasoningWire::ReasoningContent, Some("thinking".into())),
        content: Some("hi".into()),
        tool_calls: None,
        annotations: None,
        refusal: None,
    };
    let json = serde_json::to_string(&msg).unwrap();
    assert!(
        json.contains("\"reasoning_content\":\"thinking\""),
        "reasoning_content missing: {json}"
    );
    assert!(
        !json.contains("\"reasoning\":"),
        "mirror `reasoning` field leaked into message: {json}"
    );
}

fn test_usage() -> Usage {
    Usage {
        prompt_tokens: 1,
        completion_tokens: 1,
        total_tokens: 2,
        prompt_tokens_details: None,
        completion_tokens_details: None,
        time_to_first_token_ms: 0.0,
        response_tokens_per_second: 0.0,
        decode_time_ms: 0.0,
        total_time_ms: 0.0,
    }
}

#[test]
fn new_response_carries_reasoning_content() {
    let resp = ChatCompletionResponse::new(
        "m",
        "hi".into(),
        Some("thinking".into()),
        test_usage(),
        "stop",
    );
    assert_eq!(resp.choices[0].message.reasoning.text(), Some("thinking"));
    let json = serde_json::to_string(&resp).unwrap();
    assert!(
        json.contains("\"reasoning_content\":\"thinking\""),
        "reasoning_content missing: {json}"
    );
    let resp = ChatCompletionResponse::new("m", "hi".into(), None, test_usage(), "stop");
    let json = serde_json::to_string(&resp).unwrap();
    assert!(!json.contains("reasoning_content"), "{json}");
}

#[test]
fn tool_call_response_carries_reasoning_content() {
    let resp = ChatCompletionResponse::with_tool_calls(
        "m",
        None,
        Some("need the tool".into()),
        vec![crate::tool_parser::ToolCall {
            id: "call_1".into(),
            call_type: "function".into(),
            function: crate::tool_parser::FunctionCall {
                name: "get_weather".into(),
                arguments: "{}".into(),
            },
        }],
        test_usage(),
    );
    assert_eq!(
        resp.choices[0].message.reasoning.text(),
        Some("need the tool")
    );
    assert_eq!(resp.choices[0].finish_reason, "tool_calls");
    let json = serde_json::to_string(&resp).unwrap();
    assert!(
        json.contains("\"reasoning_content\":\"need the tool\""),
        "reasoning_content missing: {json}"
    );
}

/// 2026-10-14: The blocking message under each wire, as a JSON object.
fn message_json(wire: ReasoningWire, reasoning: Option<&str>) -> serde_json::Value {
    let msg = ChatMessage {
        role: "assistant".into(),
        reasoning: WireReasoning::message(wire, reasoning.map(str::to_string)),
        content: Some("4".into()),
        tool_calls: None,
        annotations: None,
        refusal: None,
    };
    serde_json::to_value(&msg).unwrap()
}

#[test]
fn vllm_wire_message_keeps_a_null_reasoning_key_when_there_is_none() {
    // 2026-10-14: The 04:43Z enable_thinking=false reply: GLM closed the think block at
    // once, so there is no reasoning text. vLLM's `model_dump()` still writes
    // `"reasoning": null`; it never writes `reasoning_content`.
    let v = message_json(ReasoningWire::Vllm, None);
    let obj = v.as_object().unwrap();
    assert!(obj.contains_key("reasoning"), "reasoning key missing: {v}");
    assert!(obj["reasoning"].is_null(), "{v}");
    assert!(!obj.contains_key("reasoning_content"), "{v}");
    assert_eq!(obj["content"], "4");
    // Control: the default wire leaves both keys out, which is the reply the live check
    // got from :8890.
    let v = message_json(ReasoningWire::ReasoningContent, None);
    let obj = v.as_object().unwrap();
    assert!(!obj.contains_key("reasoning"), "{v}");
    assert!(!obj.contains_key("reasoning_content"), "{v}");
}

#[test]
fn vllm_wire_message_carries_reasoning_text_under_reasoning_only() {
    let v = message_json(ReasoningWire::Vllm, Some("User wants 4."));
    assert_eq!(v["reasoning"], "User wants 4.", "{v}");
    assert!(v.get("reasoning_content").is_none(), "mirrored: {v}");
    // Control: the default wire writes the same text under `reasoning_content` only.
    let v = message_json(ReasoningWire::ReasoningContent, Some("User wants 4."));
    assert_eq!(v["reasoning_content"], "User wants 4.", "{v}");
    assert!(v.get("reasoning").is_none(), "mirrored: {v}");
}

#[test]
fn vllm_wire_delta_writes_reasoning_only_on_reasoning_chunks() {
    // 2026-10-14: vLLM streams with `exclude_unset=True`: a delta has `reasoning` only
    // when it carries reasoning, never a null one.
    let chunk = ChatCompletionChunk::reasoning_chunk("m", "id", ReasoningWire::Vllm, "hm".into());
    let v = serde_json::to_value(&chunk).unwrap();
    let delta = v["choices"][0]["delta"].as_object().unwrap();
    assert_eq!(delta["reasoning"], "hm", "{v}");
    assert!(!delta.contains_key("reasoning_content"), "{v}");
    for chunk in [
        ChatCompletionChunk::content_chunk("m", "id", "4".into()),
        ChatCompletionChunk::role_chunk("m", "id"),
    ] {
        let v = serde_json::to_value(&chunk).unwrap();
        let delta = v["choices"][0]["delta"].as_object().unwrap();
        assert!(!delta.contains_key("reasoning"), "{v}");
        assert!(!delta.contains_key("reasoning_content"), "{v}");
    }
}
