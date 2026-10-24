// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Tests for the SSE stream encoder (`encode_stream.rs`).
//!
//! Owner: server (OpenAI adapter).
//! Invariants: none beyond the types.

use super::*;
use crate::ir::{FinishReason, StreamDelta};

/// 2026-09-26: Parse a chunk payload and zero `created`, so two chunks
/// built across a second boundary still compare equal.
fn norm(payload: &str) -> serde_json::Value {
    let mut v: serde_json::Value = serde_json::from_str(payload).expect("valid chunk JSON");
    v["created"] = serde_json::json!(0);
    v
}

#[test]
fn stored_accum_builds_the_blocking_body_shape() {
    let mut a = StoredAccum::default();
    a.observe(&StreamDelta::Content {
        text: "he".into(),
        token_ids: Vec::new(),
    });
    a.observe(&StreamDelta::Content {
        text: "llo".into(),
        token_ids: Vec::new(),
    });
    a.observe(&StreamDelta::ToolCallStart {
        index: 0,
        id: "call_1".into(),
        name: "get_weather".into(),
    });
    a.observe(&StreamDelta::ToolCallArgs {
        index: 0,
        fragment: "{\"city\":".into(),
        token_ids: Vec::new(),
    });
    a.observe(&StreamDelta::Finish {
        reason: FinishReason::ToolCalls,
        usage: crate::ir::Usage {
            prompt_tokens: 7,
            completion_tokens: 4,
            cached_prompt_tokens: 0,
            reasoning_tokens: 0,
            accepted_prediction_tokens: 0,
            time_to_first_token_ms: 1.0,
            decode_time_ms: 2.0,
            response_tokens_per_second: 3.0,
        },
        token_ids: Vec::new(),
    });
    let v = a.to_json("chatcmpl-x", "m");
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["content"], "hello");
    assert_eq!(
        v["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
        "{\"city\":"
    );
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(v["usage"]["prompt_tokens"], 7);

    // Content-only finish: no tool_calls key.
    let mut a = StoredAccum::default();
    a.observe(&StreamDelta::Content {
        text: "hi".into(),
        token_ids: Vec::new(),
    });
    a.observe(&StreamDelta::Finish {
        reason: FinishReason::Stop,
        usage: crate::ir::Usage::default(),
        token_ids: Vec::new(),
    });
    let v = a.to_json("chatcmpl-x", "m");
    assert!(v["choices"][0]["message"].get("tool_calls").is_none());
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
}

#[test]
fn role_chunk_carries_an_empty_content_string_as_vllm_does() {
    let v =
        norm(&serde_json::to_string(&ChatCompletionChunk::role_chunk("m", "chatcmpl-x")).unwrap());
    assert_eq!(
        v["choices"][0]["delta"],
        serde_json::json!({"role": "assistant", "content": ""})
    );
}

#[test]
fn content_delta_encodes_via_content_chunk() {
    let d = StreamDelta::Content {
        text: "hi".into(),
        token_ids: Vec::new(),
    };
    let p = delta_to_payloads(
        &d,
        "m",
        "chatcmpl-x",
        false,
        ReasoningWire::ReasoningContent,
    );
    assert_eq!(p.len(), 1);
    assert!(p[0].contains("\"content\":\"hi\""), "payload: {}", p[0]);
    assert!(
        !p[0].contains("token_ids"),
        "opt-out payload must omit token_ids: {}",
        p[0]
    );
    let expected = serde_json::to_string(&ChatCompletionChunk::content_chunk(
        "m",
        "chatcmpl-x",
        "hi".to_string(),
    ))
    .unwrap();
    assert_eq!(norm(&p[0]), norm(&expected));

    let d = StreamDelta::Content {
        text: "hi".into(),
        token_ids: vec![1, 2],
    };
    let p = delta_to_payloads(
        &d,
        "m",
        "chatcmpl-x",
        false,
        ReasoningWire::ReasoningContent,
    );
    assert!(p[0].contains("\"token_ids\":[1,2]"), "payload: {}", p[0]);
}

#[test]
fn reasoning_delta_takes_the_wire_it_is_given() {
    // 2026-10-14: The SSE encoder writes the key of the wire `encode_sse_response`
    // passes (`AppState::reasoning_wire`); for GLM that is vLLM's `reasoning`.
    let d = StreamDelta::Reasoning {
        text: "mull".into(),
        token_ids: vec![9],
    };
    let p = delta_to_payloads(&d, "m", "id-1", false, ReasoningWire::Vllm);
    assert!(p[0].contains("\"reasoning\":\"mull\""), "payload: {}", p[0]);
    assert!(!p[0].contains("reasoning_content"), "payload: {}", p[0]);
}

#[test]
fn reasoning_delta_encodes_via_reasoning_chunk() {
    let d = StreamDelta::Reasoning {
        text: "mull".into(),
        token_ids: vec![9],
    };
    let p = delta_to_payloads(&d, "m", "id-1", false, ReasoningWire::ReasoningContent);
    assert_eq!(p.len(), 1);
    assert!(
        p[0].contains("\"reasoning_content\":\"mull\""),
        "payload: {}",
        p[0]
    );
    let expected = serde_json::to_string(
        &ChatCompletionChunk::reasoning_chunk(
            "m",
            "id-1",
            ReasoningWire::ReasoningContent,
            "mull".to_string(),
        )
        .with_token_ids(vec![9]),
    )
    .unwrap();
    assert_eq!(norm(&p[0]), norm(&expected));
}

#[test]
fn tool_call_start_and_args_deltas_encode_openai_tool_chunks() {
    let start = StreamDelta::ToolCallStart {
        index: 3,
        id: "call_abc".into(),
        name: "get_weather".into(),
    };
    let p = delta_to_payloads(&start, "m", "id-1", false, ReasoningWire::ReasoningContent);
    assert_eq!(p.len(), 1);
    assert!(p[0].contains("\"role\":\"assistant\""), "payload: {}", p[0]);
    assert!(
        p[0].contains(
            "\"tool_calls\":[{\"index\":3,\"id\":\"call_abc\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]"
        ),
        "payload: {}",
        p[0]
    );

    let args = StreamDelta::ToolCallArgs {
        index: 3,
        fragment: "{\"city\":".into(),
        token_ids: Vec::new(),
    };
    let p = delta_to_payloads(&args, "m", "id-1", false, ReasoningWire::ReasoningContent);
    assert_eq!(p.len(), 1);
    let expected = serde_json::to_string(&ChatCompletionChunk::tool_call_args_fragment(
        "m",
        "id-1",
        3,
        "{\"city\":",
    ))
    .unwrap();
    assert_eq!(norm(&p[0]), norm(&expected));
}

/// 2026-09-26: The usage-only chunk carries the timing fields that
/// `Usage::from(&ir::Usage)` produces.
#[test]
fn finish_usage_chunk_carries_the_raw_timing_components() {
    let usage = crate::ir::Usage {
        prompt_tokens: 10,
        completion_tokens: 5,
        cached_prompt_tokens: 0,
        reasoning_tokens: 0,
        accepted_prediction_tokens: 0,
        time_to_first_token_ms: 12.5,
        decode_time_ms: 100.0,
        response_tokens_per_second: 40.0,
    };
    let d = StreamDelta::Finish {
        reason: FinishReason::Stop,
        usage,
        token_ids: Vec::new(),
    };
    let p = delta_to_payloads(&d, "m", "id-1", true, ReasoningWire::ReasoningContent);
    let v: serde_json::Value = serde_json::from_str(&p[0]).unwrap();
    assert_eq!(v["usage"]["decode_time_ms"], 100.0, "{}", p[0]);
    assert_eq!(v["usage"]["total_time_ms"], 112.5, "{}", p[0]);
    assert_eq!(v["usage"]["time_to_first_token_ms"], 12.5, "{}", p[0]);
    assert_eq!(v["usage"]["response_token/s"], 40.0, "{}", p[0]);
}

#[test]
fn finish_framing_matches_include_usage_modes() {
    let usage = crate::ir::Usage {
        prompt_tokens: 10,
        completion_tokens: 5,
        cached_prompt_tokens: 2,
        reasoning_tokens: 3,
        accepted_prediction_tokens: 4,
        time_to_first_token_ms: 12.5,
        decode_time_ms: 100.0,
        response_tokens_per_second: 40.0,
    };
    let d = StreamDelta::Finish {
        reason: FinishReason::Stop,
        usage,
        token_ids: vec![7],
    };

    let p = delta_to_payloads(&d, "m", "id-1", true, ReasoningWire::ReasoningContent);
    assert_eq!(p.len(), 2);
    assert!(p[0].contains("\"choices\":[]"), "payload: {}", p[0]);
    assert!(p[0].contains("\"prompt_tokens\":10"), "payload: {}", p[0]);
    assert!(
        p[0].contains("\"completion_tokens\":5"),
        "payload: {}",
        p[0]
    );
    assert!(p[0].contains("\"total_tokens\":15"), "payload: {}", p[0]);
    assert!(p[0].contains("\"cached_tokens\":2"), "payload: {}", p[0]);
    assert!(p[0].contains("\"reasoning_tokens\":3"), "payload: {}", p[0]);
    assert!(
        p[0].contains("\"accepted_prediction_tokens\":4"),
        "payload: {}",
        p[0]
    );
    assert!(
        p[0].contains("\"rejected_prediction_tokens\":0"),
        "payload: {}",
        p[0]
    );
    assert!(
        p[0].contains("\"time_to_first_token_ms\":12.5"),
        "payload: {}",
        p[0]
    );
    assert!(
        p[0].contains("\"response_token/s\":40.0"),
        "payload: {}",
        p[0]
    );
    assert!(
        p[1].contains("\"finish_reason\":\"stop\""),
        "payload: {}",
        p[1]
    );
    assert!(
        !p[1].contains("\"usage\""),
        "final chunk must omit usage when include_usage=true: {}",
        p[1]
    );
    assert!(p[1].contains("\"token_ids\":[7]"), "payload: {}", p[1]);

    let p = delta_to_payloads(&d, "m", "id-1", false, ReasoningWire::ReasoningContent);
    assert_eq!(p.len(), 1);
    assert!(
        p[0].contains("\"finish_reason\":\"stop\""),
        "payload: {}",
        p[0]
    );
    assert!(p[0].contains("\"usage\":{"), "payload: {}", p[0]);
    assert!(p[0].contains("\"total_tokens\":15"), "payload: {}", p[0]);
    assert!(p[0].contains("\"token_ids\":[7]"), "payload: {}", p[0]);
}

#[test]
fn error_delta_payload_is_verbatim() {
    let msg = r#"{"error":{"message":"boom","type":"server_error","code":500}}"#;
    let d = StreamDelta::Error {
        message: msg.to_string(),
    };
    let p = delta_to_payloads(&d, "m", "id-1", false, ReasoningWire::ReasoningContent);
    assert_eq!(p, vec![msg.to_string()]);
}
