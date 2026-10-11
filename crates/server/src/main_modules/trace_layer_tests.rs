// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: Request identity through the real observability middleware: headers in,
//! extension to the handler, headers and annotated error bodies out.

use axum::Router;
use axum::body::Body;
use axum::extract::Extension;
use axum::http::{HeaderMap, HeaderValue, Request, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::get;
use tower::ServiceExt;

use super::*;
use crate::main_modules::middleware::openai_observability_middleware;

const RID: &str = "0af7651916cd43dd8448eb211c80319c";
const TP: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

async fn echo_ids(Extension(stamp): Extension<RequestStamp>) -> Response {
    Json(serde_json::json!({
        "request_id": stamp.trace.request_id.to_hex(),
        "trace_id": stamp.trace.trace_id.to_hex(),
    }))
    .into_response()
}

async fn refuse() -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({"error": {"message": "bad", "type": "invalid_request_error"}})),
    )
        .into_response()
}

async fn plain_error() -> Response {
    (StatusCode::BAD_GATEWAY, "upstream text").into_response()
}

fn app() -> Router {
    Router::new()
        .route("/v1/ok", get(echo_ids))
        .route("/v1/bad", get(refuse))
        .route("/v1/plain", get(plain_error))
        .route("/health", get(|| async { "ok" }))
        .layer(axum::middleware::from_fn(openai_observability_middleware))
}

async fn call(path: &str, headers: &[(&str, &str)]) -> (StatusCode, HeaderMap, Vec<u8>) {
    let mut req = Request::builder().uri(path);
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let resp = app()
        .oneshot(req.body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap()
        .to_vec();
    (status, headers, body)
}

#[tokio::test]
async fn incoming_ids_reach_the_handler_and_come_back_as_headers() {
    let (status, headers, body) =
        call("/v1/ok", &[("x-request-id", RID), ("traceparent", TP)]).await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["request_id"], RID);
    assert_eq!(v["trace_id"], "4bf92f3577b34da6a3ce929d0e0e4736");
    assert_eq!(headers["x-request-id"], RID);
    let tr = headers["traceresponse"].to_str().unwrap();
    assert!(
        tr.starts_with("00-4bf92f3577b34da6a3ce929d0e0e4736-"),
        "{tr}"
    );
    assert!(tr.ends_with("-01"), "{tr}");
    assert_eq!(
        headers["x-metrale-id-source"],
        "request_id=header, trace_id=header"
    );
}

#[tokio::test]
async fn missing_ids_are_minted_and_the_handler_sees_the_minted_ones() {
    let (_, headers, body) = call("/v1/ok", &[]).await;
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let rid = headers["x-request-id"].to_str().unwrap();
    assert_eq!(v["request_id"], rid);
    assert_eq!(rid.len(), 32);
    assert_eq!(
        headers["x-metrale-id-source"],
        "request_id=generated-absent, trace_id=generated-absent"
    );
    let (_, again, _) = call("/v1/ok", &[]).await;
    assert_ne!(again["x-request-id"], headers["x-request-id"]);
}

#[tokio::test]
async fn an_unusable_client_id_is_replaced_and_labelled() {
    let (_, headers, _) = call("/v1/ok", &[("x-request-id", "abc")]).await;
    assert_ne!(headers["x-request-id"], "abc");
    assert!(
        headers["x-metrale-id-source"]
            .to_str()
            .unwrap()
            .starts_with("request_id=generated-invalid")
    );
}

#[tokio::test]
async fn json_error_bodies_carry_both_ids_with_a_correct_length() {
    let (status, headers, body) =
        call("/v1/bad", &[("x-request-id", RID), ("traceparent", TP)]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["error"]["message"], "bad");
    assert_eq!(v["error"]["request_id"], RID);
    assert_eq!(v["error"]["trace_id"], "4bf92f3577b34da6a3ce929d0e0e4736");
    if let Some(len) = headers.get("content-length") {
        assert_eq!(len.to_str().unwrap(), body.len().to_string());
    }
}

#[tokio::test]
async fn non_json_errors_and_paths_outside_v1_are_untouched() {
    let (status, headers, body) = call("/v1/plain", &[("x-request-id", RID)]).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert_eq!(body, b"upstream text");
    assert_eq!(headers["x-request-id"], RID);
    let (_, headers, body) = call("/health", &[("x-request-id", RID)]).await;
    assert_eq!(body, b"ok");
    assert!(headers.get("x-request-id").is_none());
}

#[test]
fn repeated_or_non_ascii_headers_are_unreadable() {
    let mut h = HeaderMap::new();
    assert_eq!(header_read(&h, "x-request-id"), HeaderRead::Absent);
    h.insert("x-request-id", HeaderValue::from_static(RID));
    assert_eq!(header_read(&h, "x-request-id"), HeaderRead::One(RID));
    h.append("x-request-id", HeaderValue::from_static(RID));
    assert_eq!(header_read(&h, "x-request-id"), HeaderRead::Unreadable);
    let mut h = HeaderMap::new();
    h.insert("traceparent", HeaderValue::from_bytes(b"\xff00").unwrap());
    assert_eq!(header_read(&h, "traceparent"), HeaderRead::Unreadable);
}
