// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-19: Spinning suppression needs the short text-only run to be
//! loop-like, not merely short. `ccx-tp3` ratcheted: 5 short acks suppressed
//! `<tool_call>`, the suppressed turn produced another short ack, and the state
//! never lifted (13 consecutive firings in one minute).
//!
//! Owner: server (chat API) tests.
//! Invariants: every test names the behavior it pins.

use super::*;

fn text_msg(role: Role, text: &str) -> Message {
    Message {
        role,
        content: vec![crate::ir::ContentPart::Text(text.to_string())],
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
        reasoning: None,
        tool_error: false,
    }
}

/// The 2026-10-19 ratchet: five distinct short acks (a healthy session) must
/// NOT suppress. Proven to fail without the `short_run_is_loop_like` gate:
/// the old length-only condition returned true here.
#[test]
fn five_distinct_short_acks_do_not_suppress() {
    let acks = [
        "Done.",
        "Fixed it.",
        "Sure.",
        "Okay, checking.",
        "Ran the tests.",
        "Looks fine now.",
        "Pushed the fix.",
        "Applied it.",
    ];
    let messages: Vec<Message> = acks
        .iter()
        .rev()
        .map(|t| text_msg(Role::Assistant, t))
        .collect();
    let out = check_loops(&messages, true, true);
    assert!(
        !out.suppress_tool_call,
        "distinct short acks must not suppress the tool call"
    );
}

/// The original target: five identical short acks are a real spin and still
/// suppress. Positive control for the test above.
#[test]
fn five_identical_short_acks_still_suppress() {
    let ack = "Done.";
    let messages: Vec<Message> = (0..6).map(|_| text_msg(Role::Assistant, ack)).collect();
    let out = check_loops(&messages, true, true);
    assert!(
        out.suppress_tool_call,
        "an identical short-ack run is a spin and must suppress"
    );
}

/// A tool call in the run resets `recent_short`, as before.
#[test]
fn tool_call_ended_run_never_spins() {
    let mut messages: Vec<Message> = (0..6).map(|_| text_msg(Role::Assistant, "Done.")).collect();
    let mut called = text_msg(Role::Assistant, "On it.");
    called.tool_calls.push(crate::ir::message::ToolCall {
        id: "t1".into(),
        name: "Bash".into(),
        arguments: serde_json::json!({"command": "ls"}),
    });
    messages.push(called);
    let out = check_loops(&messages, true, true);
    assert!(!out.suppress_tool_call);
}

/// The lever kill-switch still wins even on a real spin.
#[test]
fn loop_suppress_off_disables_the_mask() {
    let messages: Vec<Message> = (0..6).map(|_| text_msg(Role::Assistant, "Done.")).collect();
    let out = check_loops(&messages, true, false);
    assert!(
        !out.suppress_tool_call,
        "METRALE_LOOP_NO_SUPPRESS=1 must keep the tool call available"
    );
}
