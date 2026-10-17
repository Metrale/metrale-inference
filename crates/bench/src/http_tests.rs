// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the HTTP client: SSE chunk folding and the
//! blocking error path. 2026-10-17: the reader framing and error-body tests
//! moved to `metrale-host-probe` with the reader. The child module `itl` uses the `sse` helper
//! here, and `stream_end` the loopback mock `endpoint_answering`.
//!
//! Owner: bench (HTTP client).
//! Invariants: none beyond the types.

use super::*;

fn sse(payload: &str) -> Value {
    serde_json::from_str(payload).unwrap()
}

async fn endpoint_answering(response: String) -> TargetEndpoint {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback");
    let port = listener.local_addr().expect("address").port();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let mut request = [0u8; 4096];
        let _ = socket.read(&mut request).await.expect("read request");
        socket.write_all(response.as_bytes()).await.expect("reply");
    });
    TargetEndpoint::local(port, "mock")
}

#[tokio::test]
async fn endpoint_probe_rejects_a_status_code_lookalike() {
    let target = endpoint_answering(
        "HTTP/1.1 2000 Not-A-Status\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".into(),
    )
    .await;
    let err = probe(&target, Duration::from_secs(2))
        .await
        .expect_err("only status 200 is reachable");
    assert!(err.to_string().contains("2000"), "{err}");
}

#[test]
fn content_deltas_accumulate_text_and_token_count() {
    let mut out = ChatOutcome::default();
    assert!(apply_chunk(
        &sse(r#"{"choices":[{"delta":{"content":"He"}}]}"#),
        &mut out
    ));
    assert!(apply_chunk(
        &sse(r#"{"choices":[{"delta":{"content":"llo"}}]}"#),
        &mut out
    ));
    // 2026-09-26: A role-only chunk carries no token.
    assert!(!apply_chunk(
        &sse(r#"{"choices":[{"delta":{"role":"assistant"}}]}"#),
        &mut out
    ));
    assert_eq!(out.text, "Hello");
    assert_eq!(out.completion_tokens, 2);
}

#[test]
fn a_reasoning_delta_is_a_token_and_starts_the_clock() {
    // 2026-09-26: A thinking model streams `reasoning_content` first; that
    // delta must count as a token so it starts the TTFT clock.
    let mut out = ChatOutcome::default();
    assert!(
        apply_chunk(
            &sse(r#"{"choices":[{"delta":{"reasoning_content":"Let me think"}}]}"#),
            &mut out
        ),
        "a reasoning delta must count as carried, or it cannot start the TTFT clock"
    );
    assert!(apply_chunk(
        &sse(r#"{"choices":[{"delta":{"content":"4"}}]}"#),
        &mut out
    ));
    // 2026-09-26: Reasoning stays out of `text`, which scorers parse for the
    // answer.
    assert_eq!(out.text, "4", "reasoning must not leak into the answer");
    assert_eq!(out.reasoning, "Let me think");
    assert_eq!(
        out.completion_tokens, 2,
        "both are decoded tokens -- the server's usage.completion_tokens \
         includes reasoning_tokens, and the streamed count must agree"
    );
}

#[test]
fn an_empty_reasoning_delta_carries_nothing() {
    let mut out = ChatOutcome::default();
    assert!(!apply_chunk(
        &sse(r#"{"choices":[{"delta":{"reasoning_content":""}}]}"#),
        &mut out
    ));
    assert_eq!(out.completion_tokens, 0);
}

#[test]
fn tool_call_deltas_assemble_by_index() {
    let mut out = ChatOutcome::default();
    assert!(apply_chunk(
        &sse(r#"{"choices":[{"delta":{"tool_calls":[
                {"index":0,"id":"c1","function":{"name":"get_","arguments":"{\"a\""}}]}}]}"#),
        &mut out,
    ));
    assert!(apply_chunk(
        &sse(r#"{"choices":[{"delta":{"tool_calls":[
                {"index":1,"id":"c2","function":{"name":"clock","arguments":"{}"}},
                {"index":0,"function":{"name":"weather","arguments":":1}"}}]}}]}"#),
        &mut out,
    ));
    assert_eq!(
        out.tool_calls,
        [
            ToolCall {
                id: "c1".into(),
                name: "get_weather".into(),
                arguments: r#"{"a":1}"#.into(),
            },
            ToolCall {
                id: "c2".into(),
                name: "clock".into(),
                arguments: "{}".into(),
            },
        ]
    );
}

#[test]
fn server_usage_overrides_the_streamed_delta_count() {
    let mut out = ChatOutcome::default();
    apply_chunk(&sse(r#"{"choices":[{"delta":{"content":"x"}}]}"#), &mut out);
    apply_chunk(
        &sse(r#"{"usage":{"completion_tokens":37,"prompt_tokens":12,
                "prompt_tokens_details":{"cached_tokens":8}},"choices":[]}"#),
        &mut out,
    );
    assert_eq!(out.completion_tokens, 37);
    assert_eq!(out.prompt_tokens, 12);
    assert_eq!(out.cached_prompt_tokens, 8);
}

/// 2026-09-26: The server's timing extensions ride in `usage` and are kept
/// as sent (`quick-speed-bench` derives its TPOT from `server_tps`). Absent
/// extensions stay `None`, not 0.
#[test]
fn server_timing_extensions_are_captured_and_absent_ones_stay_none() {
    let mut out = ChatOutcome::default();
    apply_chunk(
        &sse(r#"{"usage":{"completion_tokens":49,"prompt_tokens":12,
                "time_to_first_token_ms":1451.2,"response_token/s":59.9},"choices":[]}"#),
        &mut out,
    );
    assert_eq!(out.server_ttft_ms, Some(1451.2));
    assert_eq!(out.server_tps, Some(59.9));

    let mut bare = ChatOutcome::default();
    apply_chunk(
        &sse(r#"{"usage":{"completion_tokens":3,"prompt_tokens":2},"choices":[]}"#),
        &mut bare,
    );
    assert_eq!(bare.server_ttft_ms, None);
    assert_eq!(bare.server_tps, None);
}

/// 2026-09-26: The accept count rides in `completion_tokens_details` and
/// keeps three cases apart, as decode-floor does: a reported count
/// (`Some(n)`), a reported zero (`Some(0)`), and no details object at all
/// (`None`), which is never turned into 0.
#[test]
fn accepted_prediction_tokens_are_captured_and_absence_stays_none() {
    let mut out = ChatOutcome::default();
    apply_chunk(
        &sse(r#"{"usage":{"completion_tokens":49,"prompt_tokens":12,
                "completion_tokens_details":{"reasoning_tokens":0,
                "accepted_prediction_tokens":31}},"choices":[]}"#),
        &mut out,
    );
    assert_eq!(out.accepted_prediction_tokens, Some(31));

    let mut zero = ChatOutcome::default();
    apply_chunk(
        &sse(r#"{"usage":{"completion_tokens":3,"prompt_tokens":2,
                "completion_tokens_details":{"accepted_prediction_tokens":0}},"choices":[]}"#),
        &mut zero,
    );
    assert_eq!(zero.accepted_prediction_tokens, Some(0));

    let mut bare = ChatOutcome::default();
    apply_chunk(
        &sse(r#"{"usage":{"completion_tokens":3,"prompt_tokens":2},"choices":[]}"#),
        &mut bare,
    );
    assert_eq!(bare.accepted_prediction_tokens, None);
}

#[test]
fn finish_reason_is_captured() {
    let mut out = ChatOutcome::default();
    apply_chunk(
        &sse(r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#),
        &mut out,
    );
    assert_eq!(out.finish_reason.as_deref(), Some("stop"));
}

const NO_MODEL_BODY: &str = r#"{"error":{"message":"no model is loaded — Open the Library (press 4), choose a model and a recipe, and start it; then retry this request.","type":"model_not_loaded"}}"#;

#[tokio::test]
async fn blocking_client_reports_the_decoded_openai_error() {
    let response = format!(
        "HTTP/1.1 503 Service Unavailable\r\nTransfer-Encoding: chunked\r\n\
         Connection: close\r\n\r\n{:X}\r\n{NO_MODEL_BODY}\r\n0\r\n\r\n",
        NO_MODEL_BODY.len()
    );
    let target = endpoint_answering(response).await;
    let err = chat_blocking(
        &target,
        &serde_json::json!({"messages": []}),
        Duration::from_secs(2),
    )
    .await
    .expect_err("503 is an error");
    let message = err.to_string();
    assert!(message.contains("503"), "{message}");
    assert!(message.contains("Library"), "{message}");
    assert!(!message.contains(r#"{"error"#), "decoded detail: {message}");
}

#[path = "http_itl_tests.rs"]
mod itl;

#[path = "http_stream_end_tests.rs"]
mod stream_end;
