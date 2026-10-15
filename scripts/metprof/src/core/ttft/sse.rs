// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: one streamed chat response, as the socket delivered it, to its
//! client TTFT and usage (pure).
//!
//! The adapter records every socket read with its time since the request was
//! written. This module decodes the HTTP head, the chunked transfer encoding
//! (or a plain body) and the SSE `data:` frames. A frame counts at the time
//! of the read that completed it and its HTTP chunk (the server writes one
//! frame per chunk), so a frame split across reads is timed when its last
//! byte arrived.
//!
//! TTFT_client is the time of the first frame whose first choice's delta
//! carries non-empty `content` or `reasoning_content`, or a tool call (the
//! bench gate's rule). Errors: a non-200 status, an in-band `error` frame, a
//! `data:` frame that is not JSON, a stream with neither `[DONE]` nor a
//! `finish_reason`, no token-carrying frame, and no `usage.prompt_tokens`.

use serde_json::Value;

use crate::core::error::Result;
use crate::core::repr::quote;
use crate::fail;

/// The bytes of one socket read, `at_ns` after the request was written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Read {
    pub at_ns: u64,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: Option<u64>,
    /// The server's own `time_to_first_token_ms` (a join key, not TTFT).
    pub ttft_ms: Option<f64>,
    pub cached_tokens: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Streamed {
    pub ttft_client_ms: f64,
    /// `content`, `reasoning` or `tool_calls`: what the first frame carried.
    pub first_delta: &'static str,
    pub e2e_ms: f64,
    pub usage: Usage,
    pub finish_reason: String,
}

const BODY_IN_ERRORS: usize = 512;

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn snippet(b: &[u8]) -> String {
    let s = String::from_utf8_lossy(&b[..b.len().min(BODY_IN_ERRORS)]);
    quote(s.trim())
}

/// Move every complete chunk of `raw` into `body`; `Ok(true)` once the last
/// (zero-size) chunk is seen.
fn drain_chunks(raw: &mut Vec<u8>, body: &mut Vec<u8>) -> Result<bool> {
    loop {
        let Some(eol) = find(raw, b"\r\n") else {
            return Ok(false);
        };
        let line = String::from_utf8_lossy(&raw[..eol]).to_string();
        let hex = line.split(';').next().unwrap_or("").trim();
        let Ok(size) = usize::from_str_radix(hex, 16) else {
            fail!("bad chunk size line {}", quote(&line));
        };
        if size == 0 {
            raw.clear();
            return Ok(true);
        }
        let end = eol + 2 + size;
        if raw.len() < end + 2 {
            return Ok(false);
        }
        if &raw[end..end + 2] != b"\r\n" {
            fail!("a chunk of {size} bytes is not followed by CRLF");
        }
        body.extend_from_slice(&raw[eol + 2..end]);
        raw.drain(..end + 2);
    }
}

fn carried(chunk: &Value) -> Option<&'static str> {
    let delta = chunk.get("choices")?.get(0)?.get("delta")?;
    let text = |k: &str| {
        delta
            .get(k)
            .and_then(Value::as_str)
            .is_some_and(|s| !s.is_empty())
    };
    if text("reasoning_content") {
        Some("reasoning")
    } else if text("content") {
        Some("content")
    } else if delta
        .get("tool_calls")
        .and_then(Value::as_array)
        .is_some_and(|a| !a.is_empty())
    {
        Some("tool_calls")
    } else {
        None
    }
}

fn usage(u: &Value) -> Option<Usage> {
    Some(Usage {
        prompt_tokens: u.get("prompt_tokens")?.as_u64()?,
        completion_tokens: u.get("completion_tokens").and_then(Value::as_u64),
        ttft_ms: u.get("time_to_first_token_ms").and_then(Value::as_f64),
        cached_tokens: u
            .get("prompt_tokens_details")
            .and_then(|d| d.get("cached_tokens"))
            .and_then(Value::as_u64),
    })
}

#[derive(Default)]
struct Frames {
    first: Option<(u64, &'static str)>,
    usage: Option<Usage>,
    finish: Option<String>,
    done: bool,
}

impl Frames {
    fn line(&mut self, line: &[u8], at_ns: u64) -> Result<()> {
        let line = String::from_utf8_lossy(line);
        let Some(data) = line.trim_end_matches('\r').strip_prefix("data:") else {
            return Ok(());
        };
        let data = data.trim();
        if data == "[DONE]" {
            self.done = true;
            return Ok(());
        }
        let Ok(chunk) = serde_json::from_str::<Value>(data) else {
            fail!(
                "a data frame that is not JSON: {}",
                snippet(data.as_bytes())
            );
        };
        if let Some(e) = chunk.get("error").filter(|e| !e.is_null()) {
            fail!("the server reported an error mid-stream: {e}");
        }
        if let Some(u) = chunk.get("usage").filter(|u| u.is_object()) {
            match usage(u) {
                Some(u) => self.usage = Some(u),
                None => fail!("a usage block without prompt_tokens: {u}"),
            }
        }
        if let Some(r) = chunk
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("finish_reason"))
            .and_then(Value::as_str)
        {
            self.finish = Some(r.to_string());
        }
        if self.first.is_none() {
            self.first = carried(&chunk).map(|k| (at_ns, k));
        }
        Ok(())
    }
}

fn ms(ns: u64) -> f64 {
    ns as f64 / 1e6
}

/// Decode one exchange (every read, in order, up to EOF).
pub fn decode_stream(reads: &[Read]) -> Result<Streamed> {
    let mut raw: Vec<u8> = Vec::new();
    let mut head: Option<bool> = None;
    let mut body: Vec<u8> = Vec::new();
    let mut ended = false;
    let mut frames = Frames::default();
    for r in reads {
        raw.extend_from_slice(&r.bytes);
        if head.is_none() {
            let Some(end) = find(&raw, b"\r\n\r\n") else {
                continue;
            };
            let text = String::from_utf8_lossy(&raw[..end]).to_string();
            let status = text.lines().next().unwrap_or("");
            if status.split_whitespace().nth(1) != Some("200") {
                fail!("HTTP {}: {}", quote(status), snippet(&raw[end + 4..]));
            }
            let chunked = text.lines().any(|l| {
                let l = l.to_ascii_lowercase();
                l.starts_with("transfer-encoding:") && l.contains("chunked")
            });
            head = Some(chunked);
            raw.drain(..end + 4);
        }
        if head == Some(true) {
            if !ended {
                ended = drain_chunks(&mut raw, &mut body)?;
            }
        } else {
            body.append(&mut raw);
        }
        while let Some(eol) = body.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = body.drain(..=eol).collect();
            frames.line(&line[..eol], r.at_ns)?;
        }
    }
    let Some(chunked) = head else {
        fail!("the connection closed before a complete HTTP head");
    };
    if chunked && !ended {
        fail!("the chunked body ended without its last chunk");
    }
    if !body.is_empty() {
        let at = reads.last().map_or(0, |r| r.at_ns);
        frames.line(&std::mem::take(&mut body), at)?;
    }
    if !frames.done && frames.finish.is_none() {
        fail!("the stream ended with neither [DONE] nor a finish_reason: truncated");
    }
    let Some((first_ns, first_delta)) = frames.first else {
        fail!("the stream carried no content, reasoning or tool-call delta");
    };
    let Some(usage) = frames.usage else {
        fail!("the stream carried no usage block (send stream_options.include_usage)");
    };
    Ok(Streamed {
        ttft_client_ms: ms(first_ns),
        first_delta,
        e2e_ms: ms(reads.last().map_or(0, |r| r.at_ns)),
        usage,
        finish_reason: frames.finish.unwrap_or_default(),
    })
}
