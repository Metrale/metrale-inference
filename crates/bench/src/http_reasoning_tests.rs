// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Tests that a reasoning delta starts the TTFT clock under either
//! wire name, vLLM's `reasoning` and the legacy `reasoning_content`, and that
//! a content-only stream is clocked as before.
//!
//! Owner: bench (HTTP client).
//! Invariants: none beyond the types. A child of `http_tests`, so it uses
//! that module's `sse` helper, phased mock and `SSE_HEAD`.

use super::*;

/// 2026-10-14: The pause between the first token and the rest of the stream.
/// A clock that starts at the first token ends at least this long before the
/// stream does; one that waits for the later token does not.
const PAUSE: Duration = Duration::from_millis(400);

fn delta(key: &str, text: &str) -> String {
    format!("data: {{\"choices\":[{{\"delta\":{{\"{key}\":\"{text}\"}}}}]}}\n\n")
}

/// 2026-10-14: `first`, then `PAUSE`, then a content token and the end of the
/// stream.
async fn stream_with_first(first: String) -> ChatOutcome {
    let target = endpoint_answering_in_phases(vec![
        (format!("{SSE_HEAD}{first}"), PAUSE),
        (
            format!("{}data: [DONE]\n\n", delta("content", "4")),
            Duration::ZERO,
        ),
    ])
    .await;
    chat_stream(&target, &serde_json::json!({}), Duration::from_secs(5))
        .await
        .expect("stream")
}

/// 2026-10-14: The client clock started before `PAUSE`, i.e. at the first
/// delta and not at the content token after it.
fn assert_clock_started_at_the_first_delta(out: &ChatOutcome, shape: &str) {
    let ttft = out.ttft_ms.expect("a token arrived");
    let floor = PAUSE.as_secs_f64() * 1000.0 / 2.0;
    assert!(
        out.e2e_ms - ttft >= floor,
        "{shape}: TTFT {ttft:.1} ms is within {floor} ms of the end ({:.1} ms), so the \
         clock waited for the content token instead of starting at the reasoning delta",
        out.e2e_ms
    );
}

#[tokio::test]
async fn a_vllm_reasoning_delta_starts_the_clock() {
    let out = stream_with_first(delta("reasoning", "Let me think")).await;
    assert_clock_started_at_the_first_delta(&out, "vLLM `reasoning`");
    assert_eq!(out.reasoning, "Let me think");
    assert_eq!(out.text, "4", "reasoning must not leak into the answer");
    assert_eq!(out.completion_tokens, 2);
}

#[tokio::test]
async fn a_legacy_reasoning_content_delta_starts_the_clock() {
    let out = stream_with_first(delta("reasoning_content", "Let me think")).await;
    assert_clock_started_at_the_first_delta(&out, "legacy `reasoning_content`");
    assert_eq!(out.reasoning, "Let me think");
    assert_eq!(out.text, "4");
    assert_eq!(out.completion_tokens, 2);
}

#[tokio::test]
async fn a_content_only_stream_is_clocked_at_its_first_content_token() {
    let out = stream_with_first(delta("content", "2+2=")).await;
    assert_clock_started_at_the_first_delta(&out, "content only");
    assert_eq!(out.text, "2+2=4");
    assert_eq!(out.reasoning, "");
    assert_eq!(out.completion_tokens, 2);
}

/// 2026-10-14: A server that mirrors one reasoning token under both names
/// sent one token, not two.
#[test]
fn reasoning_mirrored_under_both_names_counts_once() {
    let mut out = ChatOutcome::default();
    assert!(apply_chunk(
        &sse(r#"{"choices":[{"delta":{"reasoning_content":"x","reasoning":"x"}}]}"#),
        &mut out
    ));
    assert_eq!(out.reasoning, "x");
    assert_eq!(out.completion_tokens, 1);
}

/// 2026-10-14: A null or empty `reasoning` is no token, so a role chunk or a
/// null-carrying delta cannot start the clock.
#[test]
fn a_null_or_empty_vllm_reasoning_carries_nothing() {
    let mut out = ChatOutcome::default();
    for chunk in [
        r#"{"choices":[{"delta":{"role":"assistant","content":"","reasoning":null}}]}"#,
        r#"{"choices":[{"delta":{"reasoning":""}}]}"#,
    ] {
        assert!(!apply_chunk(&sse(chunk), &mut out), "{chunk}");
    }
    assert_eq!(out.completion_tokens, 0);
    assert_eq!(out.reasoning, "");
}
