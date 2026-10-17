// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The streaming detector under the fail-closed GLM-4.7 policy
//! (`set_fail_closed`): the verdict for every fixture equals the blocking one
//! whatever the delta boundaries, the header goes out early only for an offered
//! name, arguments are held to the close, and only `<tool_call>` is a call.
//!
//! Owner: server (tool parser) tests.
//! Invariants: none beyond the types.

use super::super::glm47::{Verdict, judge};
use super::super::*;

fn tools() -> Vec<ToolDefinition> {
    let tool = |name: &str, props: serde_json::Value| ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: name.into(),
            description: None,
            parameters: Some(serde_json::json!({"type": "object", "properties": props})),
        },
    };
    vec![
        tool("bash", serde_json::json!({"command": {"type": "string"}})),
        tool(
            "web_search",
            serde_json::json!({"query": {"type": "string"}, "limit": {"type": "integer"}}),
        ),
        tool(
            "write_file",
            serde_json::json!({"path": {"type": "string"}, "content": {"type": "string"}}),
        ),
    ]
}

fn detector() -> StreamingToolDetector {
    let mut d = StreamingToolDetector::new_with_tools(tools());
    d.set_promote_bare_names(true);
    d.set_fail_closed(true);
    d
}

/// 2026-10-08: Feed `text` in `chunk`-byte deltas (on char boundaries), then
/// flush; every output in order.
fn stream(d: &mut StreamingToolDetector, text: &str, chunk: usize) -> Vec<DetectorOutput> {
    let mut out = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let mut cut = chunk.min(rest.len());
        while !rest.is_char_boundary(cut) {
            cut += 1;
        }
        out.extend(d.process(&rest[..cut]));
        rest = &rest[cut..];
    }
    out.extend(d.flush());
    out
}

/// 2026-10-08: Content and `(name, arguments, header_sent, refused)` per call.
type Call = (String, String, bool, bool);

fn summarize(outputs: &[DetectorOutput]) -> (String, Vec<Call>) {
    let mut content = String::new();
    let mut calls = Vec::new();
    let mut started: Option<(String, String)> = None;
    for o in outputs {
        match o {
            DetectorOutput::Content(t) => content.push_str(t),
            DetectorOutput::ToolCallStart { id, name, .. } => {
                assert!(started.is_none(), "one header per call");
                started = Some((id.clone(), name.clone()));
            }
            DetectorOutput::CheckedToolCall {
                call,
                header_sent,
                refused,
                ..
            } => {
                if let Some((id, name)) = started.take() {
                    assert!(header_sent, "a header went out for this call");
                    assert_eq!(call.id, id, "the call keeps its header's id");
                    assert_eq!(call.function.name, name);
                } else {
                    assert!(!header_sent);
                }
                calls.push((
                    call.function.name.clone(),
                    call.function.arguments.clone(),
                    *header_sent,
                    *refused,
                ));
            }
            other => panic!(
                "fail-closed mode emits no other tool output: {}",
                kind(other)
            ),
        }
    }
    (content, calls)
}

fn kind(o: &DetectorOutput) -> &'static str {
    match o {
        DetectorOutput::ToolCall(..) => "ToolCall",
        DetectorOutput::ToolCallDelta { .. } => "ToolCallDelta",
        DetectorOutput::ToolCallArgsFragment { .. } => "ToolCallArgsFragment",
        DetectorOutput::ToolCallEnd { .. } => "ToolCallEnd",
        _ => "other",
    }
}

const FIXTURES: &[&str] = &[
    "bash<arg_key>command</arg_key><arg_value>ls -la</arg_value>",
    "bash</arg_key><arg_key>command</arg_key><arg_value>ls</arg_value>",
    "bash<arg_key>cmd</arg_key><arg_value>ls</arg_value>",
    "bsh<arg_key>command</arg_key><arg_value>ls</arg_value>",
    "bash<arg_key>command</arg_key><arg_value>ls\nhmm</think><tool_call>web_search\
     <arg_key>query</arg_key><arg_value>rust</arg_value>",
    "bash<arg_key>command</arg_key><arg_value># find it\nweb_search<arg_key>limit\
     </arg_key><arg_value>3</arg_value>",
    "write_file<arg_key>path</arg_key><arg_value>a.rs</arg_value><arg_key>content\
     </arg_key><arg_value>fn main() { println!(\"<arg_key>\"); }</arg_value>",
];

#[test]
fn streaming_verdicts_equal_blocking_verdicts_at_every_delta_size() {
    for body in FIXTURES {
        let expected = judge(body, &tools());
        for chunk in [1, 2, 3, 7, 64, 4096] {
            let mut d = detector();
            let text = format!("Before.<tool_call>{body}</tool_call>After.");
            let (content, calls) = summarize(&stream(&mut d, &text, chunk));
            assert_eq!(content, "Before.After.", "body {body:?}, chunk {chunk}");
            assert_eq!(calls.len(), 1, "body {body:?}, chunk {chunk}");
            let (name, args, _, refused) = &calls[0];
            assert_eq!(name, expected.name(), "body {body:?}, chunk {chunk}");
            assert_eq!(args, expected.arguments(), "body {body:?}, chunk {chunk}");
            assert_eq!(*refused, matches!(expected, Verdict::Refuse { .. }));
        }
    }
}

#[test]
fn the_header_goes_out_early_only_for_an_offered_name_and_arguments_are_held() {
    let mut d = detector();
    let mut out = d.process("<tool_call>bash<arg_key>command</arg_key><arg_value>sleep");
    assert!(
        matches!(out.as_slice(), [DetectorOutput::ToolCallStart { name, idx: 0, .. }] if name == "bash"),
        "header once the name is complete and offered"
    );
    assert_eq!(d.held_call_index(), Some(0));
    out = d.process(" 1</arg_value>");
    assert!(out.is_empty(), "arguments are held until the call closes");
    out = d.process("</tool_call>");
    assert!(matches!(
        out.as_slice(),
        [DetectorOutput::CheckedToolCall {
            header_sent: true,
            refused: false,
            ..
        }]
    ));
    assert_eq!(d.held_call_index(), None);

    let mut d = detector();
    let out = d.process("<tool_call>bsh<arg_key>command</arg_key><arg_value>ls");
    assert!(out.is_empty(), "no header for a name that was not offered");
    assert_eq!(
        d.held_call_index(),
        None,
        "no header, so nothing to keep alive"
    );
}

#[test]
fn a_call_the_stream_cut_short_is_judged_at_flush() {
    let mut d = detector();
    let (content, calls) = summarize(&stream(
        &mut d,
        "<tool_call>web_search<arg_key>query</arg_key><arg_value>rust</arg_value>\
         <arg_key>limit</arg_key><arg_value>3",
        5,
    ));
    assert_eq!(content, "");
    assert_eq!(
        calls,
        vec![(
            "web_search".into(),
            r#"{"query":"rust"}"#.into(),
            true,
            false
        )]
    );
}

#[test]
fn a_split_opener_never_leaks_into_content() {
    let mut d = detector();
    let mut out = d.process("Sure.<tool_");
    assert!(
        matches!(out.as_slice(), [DetectorOutput::Content(t)] if t == "Sure."),
        "the opener prefix waits"
    );
    out = d.process("call>bash<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>");
    let (content, calls) = summarize(&out);
    assert_eq!(content, "");
    assert_eq!(calls.len(), 1);
}

#[test]
fn only_the_glm_envelope_is_a_call() {
    let text = "Qwen writes <function=bash><parameter=command>ls</parameter></function>.";
    let (content, calls) = summarize(&stream(&mut detector(), text, 4));
    assert_eq!(content, text);
    assert!(calls.is_empty());
    // 2026-10-08: Negative control: the shared detector makes a call of it.
    let mut shared = StreamingToolDetector::new_with_tools(tools());
    let mut out = shared.process(text);
    out.extend(shared.flush());
    assert!(out.iter().any(|o| !matches!(o, DetectorOutput::Content(_))));
}

#[test]
fn reset_keeps_the_policy() {
    let mut d = detector();
    d.reset();
    let (_, calls) = summarize(&stream(&mut d, "<tool_call>bsh</tool_call>", 3));
    assert_eq!(calls.len(), 1);
    assert!(calls[0].3, "still judged fail-closed after reset");
}
