// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `VllmDeltaShape` against the deltas vLLM's parser engine
//! streams for the same token sequence: unclosed, closed and absent think,
//! whitespace hold-back, and the drop before tool calls.
//!
//! Owner: server tests.
//! Invariants: none beyond the types.

use super::vllm_shape::VllmDeltaShape;
use crate::ir::{FinishReason, StreamDelta, Usage};

fn r(text: &str) -> StreamDelta {
    StreamDelta::Reasoning {
        text: text.into(),
        token_ids: Vec::new(),
    }
}

fn c(text: &str) -> StreamDelta {
    StreamDelta::Content {
        text: text.into(),
        token_ids: Vec::new(),
    }
}

fn finish() -> StreamDelta {
    StreamDelta::Finish {
        reason: FinishReason::Stop,
        usage: Usage::default(),
        token_ids: Vec::new(),
    }
}

fn tool_start() -> StreamDelta {
    StreamDelta::ToolCallStart {
        index: 0,
        id: "call_0".into(),
        name: "bash".into(),
    }
}

/// 2026-10-14: Feed one delta per token batch, as `handle_token` does, and
/// collect what goes out.
fn run(batches: Vec<StreamDelta>) -> Vec<StreamDelta> {
    let mut shape = VllmDeltaShape::default();
    batches
        .into_iter()
        .flat_map(|d| shape.apply(vec![d]))
        .collect()
}

#[test]
fn unclosed_think_streams_right_trimmed_reasoning_and_no_content() {
    let out = run(vec![r("The answer"), r(" is 4.\n"), r("\n"), finish()]);
    assert_eq!(out, vec![r("The answer"), r(" is 4."), finish()]);
}

#[test]
fn held_reasoning_whitespace_goes_out_when_more_reasoning_follows() {
    let out = run(vec![r("a\n"), r("\n"), r("b"), finish()]);
    assert_eq!(out, vec![r("a"), r("\n\nb"), finish()]);
}

#[test]
fn closed_think_drops_trailing_reasoning_whitespace_and_keeps_content_leading_newlines() {
    // 2026-10-14: `Plan.\n</think>\n\nHello`: vLLM streams reasoning
    // "Plan." and content "\n\nHello", the newlines in front of the first
    // non-whitespace content.
    let out = run(vec![r("Plan.\n"), c("\n\n"), c("Hello"), finish()]);
    assert_eq!(out, vec![r("Plan."), c("\n\nHello"), finish()]);
}

#[test]
fn absent_think_content_passes_through_once_it_has_text() {
    let out = run(vec![c("Hi"), c(" "), c("\n"), c("there"), finish()]);
    assert_eq!(out, vec![c("Hi"), c(" "), c("\n"), c("there"), finish()]);
}

#[test]
fn whitespace_only_content_goes_out_at_the_finish_without_tool_calls() {
    let out = run(vec![r("x"), c("\n\n"), finish()]);
    assert_eq!(out, vec![r("x"), c("\n\n"), finish()]);
}

#[test]
fn whitespace_only_content_is_dropped_once_a_tool_call_starts() {
    let out = run(vec![r("x"), c("\n"), tool_start(), c("\n"), finish()]);
    assert_eq!(out, vec![r("x"), tool_start(), finish()]);
    // 2026-10-14: Negative control: text after a call still goes out.
    let out = run(vec![tool_start(), c("done")]);
    assert_eq!(out, vec![tool_start(), c("done")]);
}

#[test]
fn ids_of_held_deltas_ride_on_the_next_delta_that_goes_out() {
    let mut shape = VllmDeltaShape::default();
    let held = shape.apply(vec![StreamDelta::Reasoning {
        text: "\n".into(),
        token_ids: vec![7],
    }]);
    assert!(held.is_empty(), "a whitespace-only reasoning delta is held");
    let out = shape.apply(vec![StreamDelta::Content {
        text: " ".into(),
        token_ids: vec![8],
    }]);
    assert!(out.is_empty(), "a whitespace-only content delta is held");
    let out = shape.apply(vec![StreamDelta::Finish {
        reason: FinishReason::Stop,
        usage: Usage::default(),
        token_ids: vec![9],
    }]);
    assert_eq!(
        out,
        vec![
            StreamDelta::Content {
                text: " ".into(),
                token_ids: vec![7, 8],
            },
            StreamDelta::Finish {
                reason: FinishReason::Stop,
                usage: Usage::default(),
                token_ids: vec![9],
            },
        ]
    );
}
