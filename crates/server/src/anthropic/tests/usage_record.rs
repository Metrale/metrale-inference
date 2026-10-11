// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: `/v1/messages` carries the `InferenceUsage` record in `usage` (the blocking
//! body and the streamed `message_delta`), omitted when withheld, and a stream error event
//! keeps the request and trace ids of the error it translates.
//!
//! Owner: server (Anthropic adapter) tests.
//! Invariants: none beyond the types.

use metrale_qci_dump::contract::usage::InferenceUsage;

use super::super::translator::AnthropicTranslator;
use super::super::types::MessagesResponse;
use crate::ir::response::{FinishReason, Usage};
use crate::ir::stream::StreamDelta;

/// 2026-10-11: The vendored contract's valid `completed` record.
fn record() -> InferenceUsage {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../schemas/platform/v1/fixtures/inference-usage/valid/completed.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn usage() -> Usage {
    Usage {
        prompt_tokens: 3,
        completion_tokens: 1,
        cached_prompt_tokens: 0,
        reasoning_tokens: 0,
        accepted_prediction_tokens: 0,
        time_to_first_token_ms: 0.0,
        decode_time_ms: 0.0,
        response_tokens_per_second: 0.0,
    }
}

fn response(inference_usage: Option<InferenceUsage>) -> serde_json::Value {
    let ir = crate::ir::ChatResponse {
        id: "abc".into(),
        model: "m".into(),
        created: 1,
        choices: vec![crate::ir::Choice {
            index: 0,
            content: Some("hi".into()),
            reasoning: None,
            tool_calls: Vec::new(),
            refusal: None,
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            logprobs: None,
        }],
        usage: usage(),
        inference_usage,
    };
    serde_json::to_value(MessagesResponse::from(ir)).unwrap()
}

fn message_delta_usage(inference_usage: Option<InferenceUsage>) -> serde_json::Value {
    let mut t = AnthropicTranslator::new("m".to_string());
    let mut out = Vec::new();
    t.on_delta(
        &StreamDelta::Finish {
            reason: FinishReason::Stop,
            usage: usage(),
            token_ids: Vec::new(),
            inference_usage: inference_usage.map(Box::new),
        },
        &mut out,
    );
    let delta = out.iter().find(|e| e.event == "message_delta").unwrap();
    delta.data["usage"].clone()
}

#[test]
fn the_blocking_body_carries_the_record_or_omits_it() {
    let r = record();
    let v = response(Some(r.clone()));
    assert_eq!(
        v["usage"]["inference_usage"],
        serde_json::to_value(&r).unwrap()
    );
    assert_eq!(v["usage"]["input_tokens"], 3);
    assert!(response(None)["usage"].get("inference_usage").is_none());
}

#[test]
fn the_streamed_message_delta_carries_the_record_or_omits_it() {
    let r = record();
    assert_eq!(
        message_delta_usage(Some(r.clone()))["inference_usage"],
        serde_json::to_value(&r).unwrap()
    );
    let without = message_delta_usage(None);
    assert!(without.get("inference_usage").is_none(), "{without}");
    assert_eq!(without["output_tokens"], 1);
}

#[test]
fn a_stream_error_event_keeps_the_ids_of_the_error_it_translates() {
    let mut t = AnthropicTranslator::new("m".to_string());
    let mut out = Vec::new();
    let payload = serde_json::json!({"error": {
        "message": "KV cache exhausted", "type": "server_error", "code": 500,
        "request_id": "0a".repeat(16), "trace_id": "4b".repeat(16)}});
    t.on_delta(
        &StreamDelta::Error {
            message: payload.to_string(),
        },
        &mut out,
    );
    let err = &out.iter().find(|e| e.event == "error").unwrap().data["error"];
    assert_eq!(
        *err,
        serde_json::json!({"type": "api_error", "message": "KV cache exhausted",
                           "request_id": "0a".repeat(16), "trace_id": "4b".repeat(16)})
    );
}
