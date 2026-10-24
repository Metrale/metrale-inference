// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: Tests for the `met embed` stdio transport.
//!
//! Owner: server (embed alias).
//! Invariants: none beyond the types.
//!
//! Mock: the router is a stand-in `/v1/embeddings` handler, because the real one needs a loaded
//! GPU model. Everything else is the production transport over in-memory pipes; the handler
//! itself is tested in `api/embeddings_tests.rs` and end to end on the GPU (LEDGER m1/m3 rows).

use super::*;
use axum::Json;
use axum::routing::post;
use std::time::Duration;

const SHA: &str = "ab";

fn cfg(idle: Option<Duration>) -> EmbedStdio {
    EmbedStdio {
        blueprint_sha256: SHA.into(),
        input_type: "passage".into(),
        idle_exit: idle,
    }
}

/// 2026-10-24: Row `i` of input `t` is `[len(t) + i, -0.0, 1e-40]`: bit patterns a decimal
/// round trip could change.
fn row(t: &str, i: usize) -> [f32; 3] {
    [(t.len() + i) as f32 + 0.1, -0.0, 1e-40]
}

/// 2026-10-24: The stand-in handler: it requires base64, answers like the real one, and
/// refuses an input `"bad"` with a 400.
fn fake_router() -> Router {
    Router::new().route(
        "/v1/embeddings",
        post(|Json(b): Json<Value>| async move {
            assert_eq!(b["encoding_format"], "base64", "the transport must ask for base64");
            let inputs: Vec<String> = match &b["input"] {
                Value::String(s) => vec![s.clone()],
                v => serde_json::from_value(v.clone()).unwrap(),
            };
            if inputs.iter().any(|t| t == "bad") {
                let e = serde_json::json!({"error": {"message": "[0x0203 EmbedInputInvalid] embedding input 0: bad"}});
                return (axum::http::StatusCode::BAD_REQUEST, Json(e));
            }
            let data: Vec<Value> = inputs
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let bytes: Vec<u8> = row(t, i).iter().flat_map(|x| x.to_le_bytes()).collect();
                    serde_json::json!({"object": "embedding", "index": i,
                        "embedding": base64::engine::general_purpose::STANDARD.encode(bytes)})
                })
                .collect();
            let body = serde_json::json!({"object": "list", "data": data, "model": b["model"],
                "prefix": "passage: ", "usage": {"prompt_tokens": 3, "total_tokens": 3}});
            (axum::http::StatusCode::OK, Json(body))
        }),
    )
}

fn frame(payload: &[u8]) -> Vec<u8> {
    let mut out = u32::try_from(payload.len()).unwrap().to_le_bytes().to_vec();
    out.extend_from_slice(payload);
    out
}

fn request(sha: &str, input_type: &str, encoding: &str, inputs: &[&str]) -> Vec<u8> {
    let h = serde_json::json!({"model": "m", "blueprint_sha256": sha, "input_type": input_type,
        "count": inputs.len(), "encoding": encoding});
    let b = serde_json::json!({"model": "m", "input_type": input_type, "input": inputs});
    let mut out = frame(&serde_json::to_vec(&h).unwrap());
    out.extend(frame(&serde_json::to_vec(&b).unwrap()));
    out
}

async fn read_all_frames(mut r: tokio::io::DuplexStream) -> Vec<Vec<u8>> {
    let mut bytes = Vec::new();
    r.read_to_end(&mut bytes).await.unwrap();
    let mut frames = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let n = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        frames.push(bytes[at + 4..at + 4 + n].to_vec());
        at += 4 + n;
    }
    frames
}

/// 2026-10-24: Run the transport over `input` (the write side closed after it unless `hold`),
/// returning its result and every frame it wrote.
async fn run_with(c: EmbedStdio, input: Vec<u8>, hold: bool) -> (Result<()>, Vec<Vec<u8>>) {
    let (mut client_w, mut server_r) = tokio::io::duplex(1 << 20);
    let (mut server_w, client_r) = tokio::io::duplex(1 << 20);
    client_w.write_all(&input).await.unwrap();
    let keep = if hold {
        Some(client_w)
    } else {
        drop(client_w);
        None
    };
    let res = serve_frames(fake_router(), &c, 1 << 20, &mut server_r, &mut server_w).await;
    drop(server_w);
    drop(keep);
    (res, read_all_frames(client_r).await)
}

fn code(e: &anyhow::Error) -> ErrorCode {
    e.downcast_ref::<metrale_error::Error>()
        .expect("a coded error")
        .code()
}

/// 2026-10-24: Two requests on one process: each answer is the header without vectors (with the
/// blueprint sha, model and prefix) and the exact f32 bits in order; end of input exits.
#[tokio::test]
async fn answers_until_the_input_ends_bit_exact() {
    let mut input = request(SHA, "passage", "f32le", &["aa", "b"]);
    input.extend(request(SHA, "passage", "f32le", &["cccc"]));
    let (res, frames) = run_with(cfg(Some(Duration::from_secs(30))), input, false).await;
    res.unwrap();
    assert_eq!(frames.len(), 4);
    for (k, texts) in [(0usize, vec!["aa", "b"]), (2, vec!["cccc"])] {
        let h: Value = serde_json::from_slice(&frames[k]).unwrap();
        assert_eq!(h["blueprint_sha256"], SHA);
        assert_eq!(h["model"], "m");
        assert_eq!(h["prefix"], "passage: ");
        let data = h["data"].as_array().unwrap();
        assert_eq!(data.len(), texts.len());
        assert!(data.iter().all(|d| d.get("embedding").is_none()), "{h}");
        let want: Vec<u32> = texts
            .iter()
            .enumerate()
            .flat_map(|(i, t)| row(t, i).map(f32::to_bits))
            .collect();
        let got: Vec<u32> = frames[k + 1]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(
            got, want,
            "response {k} vectors must be bit-exact and in order"
        );
    }
}

/// 2026-10-24: `--idle-exit-ms 0` answers exactly one request, even with a second one waiting.
#[tokio::test]
async fn idle_zero_answers_one_request() {
    let mut input = request(SHA, "passage", "f32le", &["a"]);
    input.extend(request(SHA, "passage", "f32le", &["b"]));
    let (res, frames) = run_with(cfg(None), input, true).await;
    res.unwrap();
    assert_eq!(frames.len(), 2);
}

/// 2026-10-24: With the input held open and no further request, the process exits after the
/// idle time, not before and not never.
#[tokio::test]
async fn an_idle_input_exits_after_the_idle_time() {
    let t0 = std::time::Instant::now();
    let input = request(SHA, "passage", "f32le", &["a"]);
    let run = run_with(cfg(Some(Duration::from_millis(300))), input, true);
    let (res, frames) = tokio::time::timeout(Duration::from_secs(10), run)
        .await
        .expect("an idle process must exit on its own");
    res.unwrap();
    assert_eq!(frames.len(), 2);
    let took = t0.elapsed();
    assert!(took >= Duration::from_millis(300), "{took:?}");
}

/// 2026-10-24: A request for another space, an unknown encoding or a malformed frame is a coded
/// error and no frame is written.
#[tokio::test]
async fn refusals_are_coded_and_write_nothing() {
    let cases: Vec<(Vec<u8>, ErrorCode)> = vec![
        (
            request("cd", "passage", "f32le", &["a"]),
            ErrorCode::EmbedSpaceMismatch,
        ),
        (
            request(SHA, "query", "f32le", &["a"]),
            ErrorCode::EmbedSpaceMismatch,
        ),
        (
            request(SHA, "passage", "f16", &["a"]),
            ErrorCode::EmbedFrameInvalid,
        ),
        (
            request(SHA, "passage", "f32le", &["bad"]),
            ErrorCode::EmbedRequestRefused,
        ),
        (frame(b"{not json"), ErrorCode::EmbedFrameInvalid),
        // 2026-10-24: Only a length prefix over the bound: refused without waiting for bytes.
        (
            (HEADER_MAX + 1).to_le_bytes().to_vec(),
            ErrorCode::EmbedFrameInvalid,
        ),
        // 2026-10-24: Two of four length bytes, then the input ends.
        (vec![5, 0], ErrorCode::EmbedFrameInvalid),
        // 2026-10-24: A header and a body frame that ends early.
        (
            {
                let mut v = request(SHA, "passage", "f32le", &["a"]);
                v.truncate(v.len() - 3);
                v
            },
            ErrorCode::EmbedFrameInvalid,
        ),
    ];
    for (i, (input, want)) in cases.into_iter().enumerate() {
        let (res, frames) = run_with(cfg(None), input, false).await;
        let e = res.expect_err(&format!("case {i} must be refused"));
        assert_eq!(code(&e), want, "case {i}: {e:#}");
        assert!(frames.is_empty(), "case {i} wrote {} frames", frames.len());
    }
}

/// 2026-10-24: The body checks: the header's count, model and input type must be the body's,
/// base64 is added when absent and anything else refused.
#[test]
fn prepare_checks_the_body_against_the_header() {
    let c = cfg(None);
    let h = br#"{"model":"m","blueprint_sha256":"ab","input_type":"passage","count":2,"encoding":"f32le"}"#;
    let (hdr, body) = prepare(
        &c,
        h,
        br#"{"model":"m","input_type":"passage","input":["x","y"]}"#,
    )
    .unwrap();
    assert_eq!(hdr.count, 2);
    let b: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(b["encoding_format"], "base64");
    assert_eq!(b["input"], serde_json::json!(["x", "y"]));
    for bad in [
        br#"{"model":"m","input_type":"passage","input":["x"]}"#.as_slice(),
        br#"{"model":"n","input_type":"passage","input":["x","y"]}"#,
        br#"{"model":"m","input_type":"query","input":["x","y"]}"#,
        br#"{"model":"m","input_type":"passage","input":["x","y"],"encoding_format":"float"}"#,
        br#"{"model":"m","input_type":"passage","input":7}"#,
    ] {
        let e = prepare(&c, h, bad).unwrap_err();
        assert_eq!(
            code(&e),
            ErrorCode::EmbedFrameInvalid,
            "{}",
            String::from_utf8_lossy(bad)
        );
    }
}

/// 2026-10-24: The exit line leads with the registry code when the error carries one.
#[test]
fn the_exit_line_is_coded() {
    let e: anyhow::Error = error!(ErrorCode::EmbedSpaceMismatch, "a", "b").into();
    assert!(
        coded_text(&e).starts_with("[0x0205 EmbedSpaceMismatch] "),
        "{}",
        coded_text(&e)
    );
    assert_eq!(coded_text(&anyhow::anyhow!("plain")), "plain");
}
