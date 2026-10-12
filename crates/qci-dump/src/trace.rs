// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: Request and trace identity at the runtime's front edge (investor-mvp #49).
//!
//! The Gateway's agent sends its request id as `x-request-id` and the trace as a W3C
//! `traceparent`. [`resolve`] keeps a valid incoming id unchanged and mints one only when
//! the header is absent or invalid, and records which happened for each id. The ids then
//! travel in response headers, error bodies and the per-request usage record, never as a
//! metric label.
//!
//! Owner: qci-dump (runtime identity).
//! Invariants:
//! - A resolved id is never all zero.
//! - A header id is used only when it parses; nothing is truncated or repaired.

use serde_json::Value;

use crate::contract::ids::{RequestId, SpanId, TraceId};

pub const REQUEST_ID_HEADER: &str = "x-request-id";
pub const TRACEPARENT_HEADER: &str = "traceparent";
/// 2026-10-11: W3C Trace Context Level 2 response header: `00-<trace>-<runtime span>-<flags>`.
pub const TRACERESPONSE_HEADER: &str = "traceresponse";
/// 2026-10-11: Says, for each id, whether it came from the request or was minted here.
pub const ID_SOURCE_HEADER: &str = "x-metrale-id-source";

/// 2026-10-11: One request header as the HTTP layer read it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeaderRead<'a> {
    Absent,
    One(&'a str),
    /// 2026-10-11: Repeated, or not visible ASCII.
    Unreadable,
}

/// 2026-10-11: Where an id came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdSource {
    Header,
    GeneratedAbsent,
    GeneratedInvalid,
}

impl IdSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Header => "header",
            Self::GeneratedAbsent => "generated-absent",
            Self::GeneratedInvalid => "generated-invalid",
        }
    }
}

/// 2026-10-11: Random bytes the caller read for this request; used only for ids it mints.
#[derive(Debug, Clone, Copy)]
pub struct Entropy {
    pub request_id: [u8; 16],
    pub trace_id: [u8; 16],
    pub span_id: [u8; 8],
}

/// 2026-10-11: The resolved identity of one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestTrace {
    pub request_id: RequestId,
    pub request_id_source: IdSource,
    pub trace_id: TraceId,
    pub trace_id_source: IdSource,
    /// 2026-10-11: The caller's span, from `traceparent`; `None` when the trace was minted.
    pub parent_id: Option<SpanId>,
    /// 2026-10-11: This runtime's span for the request.
    pub span_id: SpanId,
    /// 2026-10-11: The W3C `sampled` flag, carried through; `false` for a minted trace.
    pub sampled: bool,
}

/// 2026-10-11: The minted entropy was all zero in a field it had to fill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZeroEntropy;

/// 2026-10-11: A parsed W3C `traceparent`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceParent {
    pub trace_id: TraceId,
    pub parent_id: SpanId,
    pub flags: u8,
}

fn lower_hex(s: &str) -> bool {
    s.bytes()
        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// 2026-10-11: W3C Trace Context Level 1. Version `00` is exactly four lower-case hex
/// fields; a later version may append `-`-separated fields, which are ignored. Version
/// `ff`, an all-zero trace or parent id, and upper-case hex are invalid.
pub fn parse_traceparent(value: &str) -> Option<TraceParent> {
    let mut parts = value.split('-');
    let (version, trace, parent, flags) =
        (parts.next()?, parts.next()?, parts.next()?, parts.next()?);
    let lengths_ok =
        version.len() == 2 && trace.len() == 32 && parent.len() == 16 && flags.len() == 2;
    if !lengths_ok || ![version, trace, parent, flags].iter().all(|p| lower_hex(p)) {
        return None;
    }
    if version == "ff" || (version == "00" && parts.next().is_some()) {
        return None;
    }
    let trace_id = TraceId::from_hex(trace).ok().filter(|t| !t.is_zero())?;
    let parent_id = SpanId::from_hex(parent).ok().filter(|s| !s.is_zero())?;
    let flags = u8::from_str_radix(flags, 16).ok()?;
    Some(TraceParent {
        trace_id,
        parent_id,
        flags,
    })
}

/// 2026-10-11: A request id is 16 bytes: 32 hex digits, or a UUID in 8-4-4-4-12 form.
/// Either case; all zero is invalid.
pub fn parse_request_id(value: &str) -> Option<RequestId> {
    let groups: Vec<&str> = value.split('-').collect();
    let compact = match groups.as_slice() {
        [whole] => (*whole).to_string(),
        [a, b, c, d, e] if [a.len(), b.len(), c.len(), d.len(), e.len()] == [8, 4, 4, 4, 12] => {
            groups.concat()
        }
        _ => return None,
    };
    RequestId::from_hex(&compact)
        .ok()
        .filter(|id| !id.is_zero())
}

/// 2026-10-11: The identity of one request from its two headers. Errors only when an id
/// that had to be minted came out all zero.
pub fn resolve(
    request_id: HeaderRead<'_>,
    traceparent: HeaderRead<'_>,
    fresh: Entropy,
) -> Result<RequestTrace, ZeroEntropy> {
    let span_id = Some(SpanId(fresh.span_id))
        .filter(|s| !s.is_zero())
        .ok_or(ZeroEntropy)?;
    let (request_id, request_id_source) = match request_id {
        HeaderRead::One(v) => match parse_request_id(v) {
            Some(id) => (id, IdSource::Header),
            None => (mint_request(fresh)?, IdSource::GeneratedInvalid),
        },
        HeaderRead::Unreadable => (mint_request(fresh)?, IdSource::GeneratedInvalid),
        HeaderRead::Absent => (mint_request(fresh)?, IdSource::GeneratedAbsent),
    };
    let parsed = match traceparent {
        HeaderRead::One(v) => parse_traceparent(v).ok_or(IdSource::GeneratedInvalid),
        HeaderRead::Unreadable => Err(IdSource::GeneratedInvalid),
        HeaderRead::Absent => Err(IdSource::GeneratedAbsent),
    };
    let trace = match parsed {
        Ok(tp) => RequestTrace {
            request_id,
            request_id_source,
            trace_id: tp.trace_id,
            trace_id_source: IdSource::Header,
            parent_id: Some(tp.parent_id),
            span_id,
            sampled: tp.flags & 1 == 1,
        },
        Err(source) => RequestTrace {
            request_id,
            request_id_source,
            trace_id: Some(TraceId(fresh.trace_id))
                .filter(|t| !t.is_zero())
                .ok_or(ZeroEntropy)?,
            trace_id_source: source,
            parent_id: None,
            span_id,
            sampled: false,
        },
    };
    Ok(trace)
}

fn mint_request(fresh: Entropy) -> Result<RequestId, ZeroEntropy> {
    Some(RequestId(fresh.request_id))
        .filter(|id| !id.is_zero())
        .ok_or(ZeroEntropy)
}

impl RequestTrace {
    /// 2026-10-11: The response headers that carry this identity, in a fixed order.
    pub fn response_headers(&self) -> [(&'static str, String); 3] {
        let flags = if self.sampled { "01" } else { "00" };
        [
            (REQUEST_ID_HEADER, self.request_id.to_hex()),
            (
                TRACERESPONSE_HEADER,
                format!("00-{}-{}-{flags}", self.trace_id, self.span_id),
            ),
            (
                ID_SOURCE_HEADER,
                format!(
                    "request_id={}, trace_id={}",
                    self.request_id_source.as_str(),
                    self.trace_id_source.as_str()
                ),
            ),
        ]
    }

    /// 2026-10-11: Add `request_id` and `trace_id` to the `error` object of an OpenAI or
    /// Anthropic error envelope. `false`, and `value` unchanged, when there is none.
    pub fn annotate_error(&self, value: &mut Value) -> bool {
        let Some(error) = value.get_mut("error").and_then(Value::as_object_mut) else {
            return false;
        };
        error.insert("request_id".into(), Value::String(self.request_id.to_hex()));
        error.insert("trace_id".into(), Value::String(self.trace_id.to_hex()));
        true
    }

    /// 2026-10-11: [`Self::annotate_error`] over a serialized body. `None` when the bytes
    /// are not JSON or carry no error object, so the caller sends the original bytes.
    pub fn annotate_error_body(&self, body: &[u8]) -> Option<Vec<u8>> {
        let mut value: Value = serde_json::from_slice(body).ok()?;
        if !self.annotate_error(&mut value) {
            return None;
        }
        serde_json::to_vec(&value).ok()
    }
}

#[cfg(test)]
#[path = "trace_tests.rs"]
mod tests;
