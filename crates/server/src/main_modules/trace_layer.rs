// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The HTTP half of request identity (investor-mvp #49): read the
//! `x-request-id` and `traceparent` headers, mint what is missing from `/dev/urandom`,
//! write the identity back as response headers, and add both ids to every JSON error body.
//! The decisions themselves are `metrale_qci_dump::trace` (pure, unit-tested there).
//!
//! Owner: server (HTTP layer).
//! Invariants: an error body is rewritten only when it is JSON, bounded and carries an
//! `error` object; any other body is sent byte for byte.

use std::time::Instant;

use axum::body::Body;
use axum::http::header::{CONTENT_LENGTH, CONTENT_TYPE};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use http_body::Body as _;
use metrale_qci_dump::trace::{
    self, Entropy, HeaderRead, REQUEST_ID_HEADER, RequestTrace, TRACEPARENT_HEADER,
};

use crate::request_stamp::RequestStamp;

/// 2026-10-11: Largest error body this layer reads to add the ids. Error envelopes are a
/// few hundred bytes; a larger body passes through unchanged.
pub(crate) const MAX_ERROR_BODY: usize = 64 * 1024;

/// 2026-10-11: One header as `trace::resolve` wants it: repeated or non-ASCII values are
/// unreadable, never guessed at.
pub(crate) fn header_read<'a>(headers: &'a HeaderMap, name: &str) -> HeaderRead<'a> {
    let mut values = headers.get_all(name).iter();
    match (values.next(), values.next()) {
        (None, _) => HeaderRead::Absent,
        (Some(v), None) => v.to_str().map_or(HeaderRead::Unreadable, HeaderRead::One),
        (Some(_), Some(_)) => HeaderRead::Unreadable,
    }
}

/// 2026-10-11: No random bytes could be read for an id that had to be minted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EntropyUnavailable;

/// 2026-10-11: The stamp of a request received at `received_at`.
pub(crate) fn stamp(
    headers: &HeaderMap,
    received_at: Instant,
) -> Result<RequestStamp, EntropyUnavailable> {
    let mut bytes = [0u8; 40];
    crate::ids::getrandom(&mut bytes).map_err(|()| EntropyUnavailable)?;
    let mut fresh = Entropy {
        request_id: [0; 16],
        trace_id: [0; 16],
        span_id: [0; 8],
    };
    fresh.request_id.copy_from_slice(&bytes[..16]);
    fresh.trace_id.copy_from_slice(&bytes[16..32]);
    fresh.span_id.copy_from_slice(&bytes[32..]);
    let trace = trace::resolve(
        header_read(headers, REQUEST_ID_HEADER),
        header_read(headers, TRACEPARENT_HEADER),
        fresh,
    )
    .map_err(|_| EntropyUnavailable)?;
    Ok(RequestStamp {
        trace,
        received_at,
        received_unix_ms: crate::ids::unix_millis(),
    })
}

/// 2026-10-11: The 500 sent when [`stamp`] fails.
pub(crate) fn entropy_unavailable() -> Response {
    let body = serde_json::json!({
        "error": {
            "message": "the server could not read random bytes to mint a request id",
            "type": "server_error",
            "param": null,
            "code": "entropy_unavailable",
        }
    });
    (StatusCode::INTERNAL_SERVER_ERROR, axum::Json(body)).into_response()
}

/// 2026-10-11: Write `x-request-id`, `traceresponse` and `x-metrale-id-source`.
pub(crate) fn set_headers(headers: &mut HeaderMap, trace: &RequestTrace) {
    for (name, value) in trace.response_headers() {
        // 2026-10-11: Hex ids and fixed source words are visible ASCII, so the value is
        // always a valid header value.
        if let Ok(v) = HeaderValue::from_str(&value) {
            headers.insert(HeaderName::from_static(name), v);
        }
    }
}

/// 2026-10-11: `resp` with `request_id` and `trace_id` added to its error object, when it
/// is a 4xx/5xx JSON body of at most [`MAX_ERROR_BODY`] bytes with an exact length.
pub(crate) async fn annotate_error(resp: Response, trace: &RequestTrace) -> Response {
    let status = resp.status();
    let json = resp
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.starts_with("application/json"));
    let bounded = resp
        .body()
        .size_hint()
        .exact()
        .is_some_and(|n| n <= MAX_ERROR_BODY as u64);
    if !(status.is_client_error() || status.is_server_error()) || !json || !bounded {
        return resp;
    }
    let (mut parts, body) = resp.into_parts();
    let bytes = match axum::body::to_bytes(body, MAX_ERROR_BODY).await {
        Ok(bytes) => bytes,
        Err(e) => {
            tracing::error!("error body of a {status} response could not be read: {e}");
            parts.headers.remove(CONTENT_LENGTH);
            return Response::from_parts(parts, Body::empty());
        }
    };
    let out = trace
        .annotate_error_body(&bytes)
        .map_or(bytes, axum::body::Bytes::from);
    parts.headers.remove(CONTENT_LENGTH);
    Response::from_parts(parts, Body::from(out))
}

#[cfg(test)]
#[path = "trace_layer_tests.rs"]
mod tests;
