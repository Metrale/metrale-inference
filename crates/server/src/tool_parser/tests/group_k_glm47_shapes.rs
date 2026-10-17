// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The call shapes GLM-5.3 writes for "What's the weather in Paris?"
//! with one offered `get_weather(city)`, as the decoded output reaches the server
//! (special tokens such as `<|observation|>` are dropped by the decode): after a
//! `</think>` or straight from the reasoning the prompt opened, compact or with
//! the newlines the GLM-4.7 template itself writes between the name and the
//! pairs. Each must come out as one accepted `get_weather {"city": "Paris"}`,
//! through the reasoning split and both the blocking parse and the streaming
//! detector.
//!
//! Owner: server (tool parser) tests.
//! Invariants: none beyond the types.

use super::super::*;
use crate::reasoning_parser::ReasoningFormat;

fn tools() -> Vec<ToolDefinition> {
    vec![ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: "get_weather".into(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            })),
        },
    }]
}

const PREFIX: &str = "I'll check the weather in Paris for you.";
const CALLS: [&str; 3] = [
    "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
    "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>",
    "\n<tool_call>get_weather\n<arg_key>city</arg_key><arg_value>Paris</arg_value>\n</tool_call>\n",
];

/// 2026-10-09: The outputs: the call after a closed reasoning block, and the call
/// written straight from the reasoning (no `</think>`).
fn outputs() -> Vec<String> {
    CALLS
        .iter()
        .flat_map(|call| {
            [
                format!("Need the weather tool.</think>{PREFIX}{call}"),
                format!("{PREFIX}{call}"),
            ]
        })
        .collect()
}

fn expected() -> (String, serde_json::Value) {
    ("get_weather".into(), serde_json::json!({"city": "Paris"}))
}

#[test]
fn every_shape_is_one_accepted_call_on_the_blocking_path() {
    for output in outputs() {
        let (_, answer) = ReasoningFormat::Glm47
            .into_parser()
            .extract_thinking(&output, true);
        let (_, verdicts) = parse_glm47_answer(&answer, &tools());
        let calls: Vec<_> = verdicts
            .into_iter()
            .map(|v| match v {
                Verdict::Accept { name, arguments } => (
                    name,
                    serde_json::from_str::<serde_json::Value>(&arguments).unwrap(),
                ),
                other => panic!("refused: {other:#?}\n{output:?}"),
            })
            .collect();
        assert_eq!(calls, vec![expected()], "{output:?}");
    }
}

#[test]
fn every_shape_is_one_accepted_call_on_the_streaming_path() {
    for output in outputs() {
        let (_, answer) = ReasoningFormat::Glm47
            .into_parser()
            .extract_thinking(&output, true);
        for chunk in [1, 5, 64] {
            let mut d = StreamingToolDetector::new_with_tools(tools());
            d.set_promote_bare_names(true);
            d.set_fail_closed(true);
            let mut outs = Vec::new();
            let bytes: Vec<char> = answer.chars().collect();
            for piece in bytes.chunks(chunk) {
                outs.extend(d.process(&piece.iter().collect::<String>()));
            }
            outs.extend(d.flush());
            let calls: Vec<_> = outs
                .into_iter()
                .filter_map(|o| match o {
                    DetectorOutput::CheckedToolCall { call, refused, .. } => {
                        assert!(!refused, "{output:?}");
                        Some((
                            call.function.name,
                            serde_json::from_str::<serde_json::Value>(&call.function.arguments)
                                .unwrap(),
                        ))
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(calls, vec![expected()], "chunk {chunk}: {output:?}");
        }
    }
}
