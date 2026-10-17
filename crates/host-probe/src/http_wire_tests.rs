// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the HTTP response reader: framing and error bodies.
//! 2026-10-17: Moved out of the bench's `http_tests.rs` with the reader;
//! the streaming-client tests stay there.
//!
//! Owner: host-probe (HTTP wire).
//! Invariants: none beyond the types.

use super::*;

#[test]
fn chunked_body_split_mid_line_still_yields_one_intact_line() {
    // 2026-09-26: A chunk boundary in the middle of a `data:` line must still
    // yield one intact line.
    let mut r = Reader::default();
    let head = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: Chunked\r\n\r\n";
    assert!(r.push(head).unwrap().is_empty());

    let first = "data: {\"choices\":[{\"de";
    let second = "lta\":{\"content\":\"hi\"}}]}\n";
    let mut wire = Vec::new();
    wire.extend_from_slice(format!("{:x}\r\n{first}\r\n", first.len()).as_bytes());
    let lines = r.push(&wire).unwrap();
    assert!(lines.is_empty(), "no complete line yet, got {lines:?}");

    let mut wire2 = Vec::new();
    wire2.extend_from_slice(format!("{:x}\r\n{second}\r\n", second.len()).as_bytes());
    let lines = r.push(&wire2).unwrap();
    assert_eq!(lines, [r#"data: {"choices":[{"delta":{"content":"hi"}}]}"#]);
}

#[test]
fn identity_body_is_read_straight_through() {
    let mut r = Reader::default();
    let lines = r
        .push(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\ndata: a\ndata: b\n")
        .unwrap();
    assert_eq!(lines, vec!["data: a", "data: b"]);
}

#[test]
fn a_non_200_status_is_an_error_not_an_empty_stream() {
    // 2026-09-26: A non-200 never reads as a successful empty stream. The
    // reader waits for the body; with no Content-Length and no body, as here,
    // the error is raised at EOF by `finish`.
    let mut r = Reader::default();
    let lines = r.push(b"HTTP/1.1 404 Not Found\r\n\r\n").unwrap();
    assert!(lines.is_empty(), "a failed response yields no data lines");
    let err = r.finish().unwrap_err().to_string();
    assert!(err.contains("404"), "{err}");

    let mut lookalike = Reader::default();
    lookalike
        .push(b"HTTP/1.1 2000 Not-A-Status\r\nContent-Length: 0\r\n\r\n")
        .expect_err("only the exact 200 status is successful");
}

/// 2026-09-26: A 503 with the server's JSON error body and a
/// `Content-Length`.
fn error_response(body: &str) -> Vec<u8> {
    format!(
        "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}

const NO_MODEL_BODY: &str = r#"{"error":{"message":"no model is loaded — Open the Library (press 4), choose a model and a recipe, and start it; then retry this request.","type":"model_not_loaded"}}"#;

#[test]
fn an_error_body_reaches_the_caller_instead_of_just_the_status_line() {
    let mut r = Reader::default();
    let err = r.push(&error_response(NO_MODEL_BODY)).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("503"), "keeps the status: {msg}");
    assert!(msg.contains("Library"), "and carries the hint: {msg}");
}

#[test]
fn an_error_body_arriving_after_its_headers_is_still_reported() {
    // 2026-09-26: Headers can land in a read of their own; the body that
    // follows must still be reported.
    let whole = error_response(NO_MODEL_BODY);
    let split = find(&whole, b"\r\n\r\n").unwrap() + 4;

    let mut r = Reader::default();
    assert!(
        r.push(&whole[..split]).is_ok(),
        "headers alone are not yet a verdict — the body is still coming"
    );
    let msg = format!("{}", r.push(&whole[split..]).unwrap_err());
    assert!(msg.contains("Library"), "hint survives the split: {msg}");
}

#[test]
fn a_chunked_error_body_is_decoded_before_it_is_parsed() {
    // 2026-09-26: A chunked error body with no Content-Length; collected
    // raw, it would not be JSON.
    let body = NO_MODEL_BODY;
    let raw = format!(
        "HTTP/1.1 503 Service Unavailable\r\ncontent-type: application/json\r\n\
         transfer-encoding: chunked\r\n\r\n{:X}\r\n{body}\r\n0\r\n\r\n",
        body.len()
    );
    let mut r = Reader::default();
    let msg = format!("{}", r.push(raw.as_bytes()).unwrap_err());
    assert!(msg.contains("503"), "{msg}");
    assert!(msg.contains("Library"), "chunked hint must survive: {msg}");
}

#[test]
fn a_chunked_error_split_across_reads_is_still_decoded() {
    let body = NO_MODEL_BODY;
    let raw = format!(
        "HTTP/1.1 503 Service Unavailable\r\ntransfer-encoding: chunked\r\n\r\n\
         {:X}\r\n{body}\r\n0\r\n\r\n",
        body.len()
    );
    let bytes = raw.as_bytes();
    // 2026-09-26: Split inside the chunk data.
    let cut = bytes.len() - 40;
    let mut r = Reader::default();
    assert!(r.push(&bytes[..cut]).is_ok(), "partial chunk: keep waiting");
    let msg = format!("{}", r.push(&bytes[cut..]).unwrap_err());
    assert!(msg.contains("Library"), "{msg}");
}

#[test]
fn an_error_without_content_length_is_reported_at_eof() {
    // 2026-09-26: No length and no chunking means the body ends at close, so
    // the error is reported at EOF.
    let mut r = Reader::default();
    let raw = format!("HTTP/1.1 500 Internal Server Error\r\n\r\n{NO_MODEL_BODY}");
    assert!(r.push(raw.as_bytes()).is_ok(), "still open, still waiting");
    let msg = format!("{}", r.finish().unwrap_err());
    assert!(msg.contains("500"), "{msg}");
    assert!(msg.contains("Library"), "{msg}");
}

#[test]
fn an_unparseable_error_body_still_reports_the_status() {
    // 2026-09-26: A body that is not an OpenAI error still reports the
    // status.
    let mut r = Reader::default();
    let raw = "HTTP/1.1 502 Bad Gateway\r\nContent-Length: 9\r\n\r\n<html></".to_string();
    let _ = r.push(raw.as_bytes());
    let msg = format!("{}", r.finish().unwrap_err());
    assert!(msg.contains("502"), "{msg}");
}

#[test]
fn a_200_response_is_completely_unaffected_by_the_error_path() {
    let mut r = Reader::default();
    let raw = "HTTP/1.1 200 OK\r\n\r\ndata: {\"a\":1}\n";
    let lines = r.push(raw.as_bytes()).unwrap();
    assert_eq!(lines, vec!["data: {\"a\":1}".to_string()]);
    assert!(r.finish().is_ok(), "no pending error on a good response");
}

#[test]
fn a_huge_error_body_is_capped_rather_than_buffered_without_limit() {
    let mut r = Reader::default();
    let big = "x".repeat(MAX_ERROR_BODY + 4096);
    let raw = format!("HTTP/1.1 503 Service Unavailable\r\n\r\n{big}");
    // 2026-09-26: Fails at the cap rather than growing until the sender
    // stops.
    assert!(r.push(raw.as_bytes()).is_err());
    assert_eq!(r.body.len(), MAX_ERROR_BODY, "the retained body is capped");

    let mut chunked = Reader::default();
    let wire = format!(
        "HTTP/1.1 503 Service Unavailable\r\nTransfer-Encoding: chunked\r\n\r\n\
         {:X}\r\n{big}\r\n0\r\n\r\n",
        big.len()
    );
    assert!(chunked.push(wire.as_bytes()).is_err());
    assert_eq!(
        chunked.body.len(),
        MAX_ERROR_BODY,
        "chunk framing cannot bypass the same cap"
    );
}

#[test]
fn message_from_body_ignores_bodies_that_are_not_openai_shaped() {
    assert_eq!(message_from_body(""), None);
    assert_eq!(message_from_body("{"), None);
    assert_eq!(message_from_body(r#"{"error":"str"}"#), None);
    assert_eq!(message_from_body(r#"{"error":{"message":"  "}}"#), None);
    assert_eq!(
        message_from_body(r#"{"error":{"message":"boom"}}"#).as_deref(),
        Some("boom")
    );
}

#[test]
fn a_whole_chunked_error_response_yields_its_message_in_one_shot() {
    // 2026-09-26: The TUI chat pane holds the entire response, not a stream,
    // and gets the same de-chunked answer as the streaming reader.
    let body = NO_MODEL_BODY;
    let raw = format!(
        "HTTP/1.1 503 Service Unavailable\r\ntransfer-encoding: chunked\r\n\r\n\
         {:X}\r\n{body}\r\n0\r\n\r\n",
        body.len()
    );
    let msg = error_message_from_response(raw.as_bytes()).expect("a message is there");
    assert!(msg.contains("Library"), "{msg}");
}

#[test]
fn a_successful_response_has_no_error_message_to_extract() {
    let raw = "HTTP/1.1 200 OK\r\n\r\ndata: {\"a\":1}\n";
    assert_eq!(error_message_from_response(raw.as_bytes()), None);
}

#[test]
fn an_error_response_with_an_unreadable_body_yields_nothing_rather_than_junk() {
    let raw = "HTTP/1.1 502 Bad Gateway\r\n\r\n<html>nope</html>";
    assert_eq!(error_message_from_response(raw.as_bytes()), None);
}
