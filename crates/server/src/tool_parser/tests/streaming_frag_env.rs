// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Streaming-detector tests: `buffer_args` (`METRALE_BUFFER_TOOL_ARGS`) turns off
//! live argument fragments, and a bare `<function=…>` call streams live
//! without also being emitted whole. Fixtures come from `streaming_frag`.
//!
//! Owner: server (tool parser) tests.
//! Invariants: none beyond the types.

use super::super::*;
use super::streaming_frag::{args_from_outputs, collect_fragments, write_and_bash_tools};

#[test]
fn kill_switch_buffers_full_args_no_fragments() {
    // 2026-09-26: With `buffer_args`, the arguments come as one `ToolCallDelta` at the
    // close and no `ToolCallArgsFragment` is emitted.
    let mut det = StreamingToolDetector::new_with_tools(write_and_bash_tools(), true);
    let chunks = [
        "<tool_call>",
        "<function=Write>",
        "<parameter=file_path>",
        "/tmp/x.rs",
        "</parameter>",
        "<parameter=content>",
        "hello",
        "</parameter>",
        "</function>",
        "</tool_call>",
    ];
    let mut outputs = Vec::new();
    for c in chunks {
        outputs.extend(det.process(c));
    }
    let frag_count = outputs
        .iter()
        .filter(|o| matches!(o, DetectorOutput::ToolCallArgsFragment { .. }))
        .count();
    let delta_count = outputs
        .iter()
        .filter(|o| matches!(o, DetectorOutput::ToolCallDelta { .. }))
        .count();
    assert_eq!(frag_count, 0, "kill-switch must emit NO live fragments");
    assert_eq!(
        delta_count, 1,
        "kill-switch must emit exactly one buffered ToolCallDelta"
    );
    let args: serde_json::Value = serde_json::from_str(&args_from_outputs(&outputs)).unwrap();
    assert_eq!(args["file_path"], "/tmp/x.rs");
    assert_eq!(args["content"], "hello");
}

#[test]
fn bare_function_streams_incrementally_and_emits_call_once() {
    // 2026-09-26: A bare `<function=…>` with no envelope streams its header
    // and each completed parameter (`process_bare_function`), and a call
    // streamed that way is closed with `ToolCallEnd`, not emitted again whole.
    let mut det = StreamingToolDetector::new_with_tools(write_and_bash_tools(), false);
    let full = "<function=Write>\n\
                <parameter=file_path>\n/tmp/x.rs\n</parameter>\n\
                <parameter=content>\nhello\n</parameter>\n\
                </function>";
    let bytes = full.as_bytes();
    let mut outputs = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let end = (i + 5).min(bytes.len());
        outputs.extend(det.process(&full[i..end]));
        i = end;
    }
    outputs.extend(det.flush());

    let start_pos = outputs
        .iter()
        .position(|o| matches!(o, DetectorOutput::ToolCallStart { .. }))
        .expect("bare <function=> must emit ToolCallStart before the block closes");
    let frag_positions: Vec<usize> = outputs
        .iter()
        .enumerate()
        .filter(|(_, o)| matches!(o, DetectorOutput::ToolCallArgsFragment { .. }))
        .map(|(i, _)| i)
        .collect();
    assert!(
        frag_positions.len() >= 2,
        "expected MULTIPLE incremental fragments for a bare function, got {}",
        frag_positions.len()
    );
    assert!(
        frag_positions.iter().all(|&p| p > start_pos),
        "fragments must follow ToolCallStart"
    );

    let whole_calls = outputs
        .iter()
        .filter(|o| matches!(o, DetectorOutput::ToolCall(..)))
        .count();
    assert_eq!(
        whole_calls, 0,
        "a call streamed incrementally must not ALSO be emitted whole (duplicate delivery)"
    );

    let args: serde_json::Value = serde_json::from_str(&collect_fragments(&outputs)).unwrap();
    assert_eq!(
        args,
        serde_json::json!({"file_path": "/tmp/x.rs", "content": "hello"})
    );
}
