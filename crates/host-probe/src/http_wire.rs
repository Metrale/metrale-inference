// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The HTTP/1.1 response wire shared by the bench client and the
//! TUI chat: the incremental [`Reader`] (chunked or raw framing, non-200
//! error-body collection), the request-failure taxonomy in [`failure`], and
//! the OpenAI error-body decoders. Moved out of `metrale-bench::http` so the
//! serve path reads error bodies without depending on the bench (OWAY).
//!
//! Owner: host-probe (HTTP wire).
//! Invariants:
//! - A non-200 response never reads as a successful empty stream.
//! - An error body is kept to at most [`MAX_ERROR_BODY`] bytes.

pub mod failure;
mod reader;
pub use reader::Reader;

/// 2026-09-26: The trimmed `error.message` of an OpenAI-shaped error body.
/// `None` when the body is not JSON with a non-empty string there, so callers
/// can fall back to the status line.
pub fn message_from_body(body: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let msg = v.get("error")?.get("message")?.as_str()?.trim();
    (!msg.is_empty()).then(|| msg.to_string())
}

/// 2026-09-26: The message from a complete non-200 HTTP response (status
/// line, headers and body), decoded by the same [`Reader`] as streams,
/// chunked bodies included. `None` unless the response is a non-200 whose
/// body [`message_from_body`] can read.
pub fn error_message_from_response(raw: &[u8]) -> Option<String> {
    let mut r = Reader::default();
    // 2026-09-26: Both return the error being reported; only the decoded body
    // is needed.
    let _ = r.push(raw);
    let _ = r.finish();
    r.error_status.as_ref()?;
    message_from_body(String::from_utf8_lossy(&r.body).trim())
}

/// 2026-09-26: Cap on how much of a non-200 body to buffer before reporting
/// it.
pub const MAX_ERROR_BODY: usize = 64 * 1024;

/// 2026-09-26: Whether a header block declares chunked transfer-encoding.
pub fn is_chunked(head: &str) -> bool {
    head.lines().any(|line| {
        let line = line.to_ascii_lowercase();
        line.starts_with("transfer-encoding:") && line.contains("chunked")
    })
}

/// 2026-09-26: Whether a status line is exactly status 200.
pub fn status_is_success(status_line: &str) -> bool {
    status_line.split_whitespace().nth(1) == Some("200")
}

/// 2026-09-26: `Content-Length` from a header block, if declared and
/// parseable.
pub fn content_length(head: &str) -> Option<usize> {
    head.lines()
        .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
        .and_then(|l| l.split_once(':'))
        .and_then(|(_, v)| v.trim().parse().ok())
}

/// 2026-09-26: The first index of `needle` in `hay`.
pub fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
#[path = "http_wire_tests.rs"]
mod tests;
