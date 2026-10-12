// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `compact::stream_error_frame`, the frame a `StreamEvent::Error`
//! becomes on the chat and `/v1/completions` streams. The stream has already answered
//! HTTP 200, so this frame is where the client reads the reason: it must be valid JSON
//! that carries the message unchanged, and (2026-10-11) the request and trace ids.
//!
//! Owner: server API.
//! Invariants: none beyond the types.

use metrale_qci_dump::contract::ids::{RequestId, SpanId, TraceId};
use metrale_qci_dump::trace::{IdSource, RequestTrace};

use crate::api::compact::stream_error_frame;

/// 2026-09-26: Error texts shaped like the scheduler's: `send_error_to_sink` callers pass
/// `{e:#}` anyhow chains (for example `scheduler/lifecycle.rs` "swap-in failed"), which
/// can hold quotes, newlines and backslashes.
const HOSTILE: &[&str] = &[
    r#"swap-in failed: open "/var/spill/swap_7.bin": No such file"#,
    "prefill failed: CUDA error\nlaunch failed",
    r"grammar compile failed near \x00",
    r#"tool "bash" rejected: unbalanced """#,
];

pub(super) fn trace() -> RequestTrace {
    RequestTrace {
        request_id: RequestId([0x0a; 16]),
        request_id_source: IdSource::Header,
        trace_id: TraceId([0x4b; 16]),
        trace_id_source: IdSource::Header,
        parent_id: None,
        span_id: SpanId([0x01; 8]),
        sampled: false,
    }
}

#[test]
fn an_error_message_survives_the_frame_as_parseable_json() {
    for msg in HOSTILE {
        let frame = stream_error_frame(msg, &trace());
        let v: serde_json::Value = serde_json::from_str(&frame).unwrap_or_else(|e| {
            panic!("frame for {msg:?} must be JSON a client can parse, got {frame:?}: {e}")
        });
        assert_eq!(
            v["error"]["message"].as_str(),
            Some(*msg),
            "frame must round-trip the reason verbatim: {frame}"
        );
    }
}

#[test]
fn the_frame_is_the_error_envelope_with_both_ids() {
    let v: serde_json::Value =
        serde_json::from_str(&stream_error_frame("Scheduler queue closed", &trace())).unwrap();
    assert_eq!(
        v,
        serde_json::json!({"error": {
            "message": "Scheduler queue closed",
            "type": "server_error",
            "code": 500,
            "request_id": "0a".repeat(16),
            "trace_id": "4b".repeat(16),
        }})
    );
}
