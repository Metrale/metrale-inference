// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: The `met embed` transport: embedding requests as u32-LE length-prefixed frames
//! on stdin, answered on stdout by the serve router's own `/v1/embeddings` handler
//! (DESIGN-latlas-local-embedder.md §4, the codec latlas-core `metframe` speaks).
//!
//! Request: frame 1 is the JSON header `{model, blueprint_sha256, input_type, count,
//! encoding: "f32le"}`, frame 2 the `/v1/embeddings` body. Response: frame 1 is the handler's
//! JSON without the `embedding` arrays, plus `blueprint_sha256`; frame 2 the vectors as raw
//! little-endian f32 `[count][dim]` in request order, bit-exact (the handler is asked for
//! base64 and its bytes are passed through, never re-parsed as decimals).
//!
//! Owner: server (embed alias).
//! Invariants:
//! - Nothing but frames is written to the output (logs go to stderr; `main` points fd 1 there).
//! - A refused request is an error, not a frame: the process exits non-zero and the coded
//!   error is stderr's last line, which the host reports.
//! - A frame longer than its bound is refused from its length prefix, before it is buffered.

use anyhow::Result;
use axum::Router;
use base64::Engine as _;
use metrale_error::{ErrorCode, error};
use serde::Deserialize;
use serde_json::Value;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tower::ServiceExt as _;

use crate::cli::embed_args::EmbedStdio;

/// 2026-10-24: The request header's bound: a handful of short strings.
pub const HEADER_MAX: u32 = 64 * 1024;
/// 2026-10-24: The vector encoding of response frame 2.
pub const ENCODING: &str = "f32le";

/// 2026-10-24: Request frame 1.
#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RequestHeader {
    pub model: String,
    pub blueprint_sha256: String,
    pub input_type: String,
    pub count: usize,
    pub encoding: String,
}

fn frame_err(which: &str, why: impl Into<String>) -> anyhow::Error {
    error!(ErrorCode::EmbedFrameInvalid, which.to_string(), why.into()).into()
}

fn space_err(what: &str, got: &str, want: &str) -> anyhow::Error {
    error!(
        ErrorCode::EmbedSpaceMismatch,
        format!("{what} {got:?}"),
        format!("{what} {want:?}")
    )
    .into()
}

/// 2026-10-24: The next frame of at most `max` bytes. `Ok(None)` only for an input that ends
/// cleanly before a length prefix when `may_end` (between requests).
pub async fn read_frame<R: AsyncRead + Unpin>(
    r: &mut R,
    which: &str,
    max: u32,
    may_end: bool,
) -> Result<Option<Vec<u8>>> {
    let mut len = [0u8; 4];
    let mut got = 0;
    while got < 4 {
        let n = r.read(&mut len[got..]).await?;
        if n == 0 {
            if got == 0 && may_end {
                return Ok(None);
            }
            return Err(frame_err(
                which,
                format!("truncated: {got} of 4 length bytes"),
            ));
        }
        got += n;
    }
    let n = u32::from_le_bytes(len);
    if n > max {
        return Err(frame_err(
            which,
            format!("declares {n} bytes; the limit is {max}"),
        ));
    }
    let mut buf = vec![0u8; usize::try_from(n)?];
    r.read_exact(&mut buf).await.map_err(|e| {
        frame_err(
            which,
            format!("truncated body of {n} bytes ({:?})", e.kind()),
        )
    })?;
    Ok(Some(buf))
}

async fn write_frame<W: AsyncWrite + Unpin>(w: &mut W, which: &str, payload: &[u8]) -> Result<()> {
    let n = u32::try_from(payload.len())
        .map_err(|_| frame_err(which, format!("{} bytes exceed u32", payload.len())))?;
    w.write_all(&n.to_le_bytes()).await?;
    w.write_all(payload).await?;
    Ok(())
}

/// 2026-10-24: Check a request's two frames against this process, and the body the handler
/// gets: the client's, with `encoding_format: "base64"` added when it names none.
pub fn prepare(cfg: &EmbedStdio, header: &[u8], body: &[u8]) -> Result<(RequestHeader, Vec<u8>)> {
    let h: RequestHeader =
        serde_json::from_slice(header).map_err(|e| frame_err("request header", e.to_string()))?;
    if h.blueprint_sha256 != cfg.blueprint_sha256 {
        return Err(space_err(
            "blueprint_sha256",
            &h.blueprint_sha256,
            &cfg.blueprint_sha256,
        ));
    }
    if h.input_type != cfg.input_type {
        return Err(space_err("input_type", &h.input_type, &cfg.input_type));
    }
    if h.encoding != ENCODING {
        return Err(frame_err(
            "request header",
            format!("encoding {:?} is not {ENCODING}", h.encoding),
        ));
    }
    let mut b: serde_json::Map<String, Value> =
        serde_json::from_slice(body).map_err(|e| frame_err("request body", e.to_string()))?;
    let field = |k: &str| b.get(k).and_then(Value::as_str).map(str::to_string);
    if field("model").as_deref() != Some(h.model.as_str()) {
        return Err(frame_err(
            "request body",
            format!("`model` is not the header's {:?}", h.model),
        ));
    }
    if field("input_type").as_deref() != Some(h.input_type.as_str()) {
        return Err(frame_err(
            "request body",
            format!("`input_type` is not the header's {:?}", h.input_type),
        ));
    }
    let count = match b.get("input") {
        Some(Value::String(_)) => 1,
        Some(Value::Array(a)) => a.len(),
        _ => {
            return Err(frame_err(
                "request body",
                "`input` is not a string or an array",
            ));
        }
    };
    if count != h.count {
        return Err(frame_err(
            "request body",
            format!("{count} inputs; the header says {}", h.count),
        ));
    }
    match b.get("encoding_format") {
        None => {
            b.insert("encoding_format".into(), Value::String("base64".into()));
        }
        Some(Value::String(s)) if s == "base64" => {}
        Some(other) => {
            return Err(frame_err(
                "request body",
                format!("`encoding_format` {other} is not base64 (frames carry {ENCODING})"),
            ));
        }
    }
    Ok((h, serde_json::to_vec(&b)?))
}

/// 2026-10-24: The handler's answer as the two response frames: the JSON without the vectors
/// plus `blueprint_sha256`, and the vectors' bytes in index order. Any status but 200 is the
/// handler's refusal, returned as `EmbedRequestRefused`.
pub fn split_response(
    cfg: &EmbedStdio,
    count: usize,
    status: u16,
    body: &[u8],
) -> Result<(Vec<u8>, Vec<u8>)> {
    let mut v: Value = serde_json::from_slice(body)
        .map_err(|e| frame_err("handler response", format!("HTTP {status}: {e}")))?;
    if status != 200 {
        let msg = v
            .pointer("/error/message")
            .and_then(Value::as_str)
            .unwrap_or("(no message)");
        return Err(error!(ErrorCode::EmbedRequestRefused, status, msg.to_string()).into());
    }
    let data = v
        .get_mut("data")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| frame_err("handler response", "no `data` array"))?;
    if data.len() != count {
        return Err(frame_err(
            "handler response",
            format!("{} vectors for {count} inputs", data.len()),
        ));
    }
    let mut vectors = Vec::new();
    let mut width = None;
    for (i, item) in data.iter_mut().enumerate() {
        if item.get("index").and_then(Value::as_u64) != Some(u64::try_from(i)?) {
            return Err(frame_err(
                "handler response",
                format!("data[{i}] is not index {i}"),
            ));
        }
        let emb = item
            .as_object_mut()
            .and_then(|o| o.remove("embedding"))
            .ok_or_else(|| frame_err("handler response", format!("data[{i}] has no embedding")))?;
        let bytes = emb
            .as_str()
            .map(|s| base64::engine::general_purpose::STANDARD.decode(s))
            .ok_or_else(|| frame_err("handler response", format!("data[{i}] is not base64")))?
            .map_err(|e| frame_err("handler response", format!("data[{i}]: {e}")))?;
        if *width.get_or_insert(bytes.len()) != bytes.len() || bytes.len() % 4 != 0 {
            return Err(frame_err(
                "handler response",
                format!("data[{i}] has {} bytes", bytes.len()),
            ));
        }
        vectors.extend_from_slice(&bytes);
    }
    v.as_object_mut()
        .ok_or_else(|| frame_err("handler response", "not an object"))?
        .insert(
            "blueprint_sha256".into(),
            Value::String(cfg.blueprint_sha256.clone()),
        );
    Ok((serde_json::to_vec(&v)?, vectors))
}

/// 2026-10-24: One request through `router`: status and body.
async fn dispatch(router: &Router, body: Vec<u8>, body_max: usize) -> Result<(u16, Vec<u8>)> {
    let req = axum::http::Request::post("/v1/embeddings")
        .header(axum::http::header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(body))?;
    let resp = router.clone().oneshot(req).await?;
    let status = resp.status().as_u16();
    let bytes = axum::body::to_bytes(resp.into_body(), body_max).await?;
    Ok((status, bytes.to_vec()))
}

/// 2026-10-24: Answer requests from `r` on `w` until the input ends, the first response when
/// `cfg.idle_exit` is `None`, or `cfg.idle_exit` passes with no request.
pub async fn serve_frames<R, W>(
    router: Router,
    cfg: &EmbedStdio,
    body_max: u32,
    r: &mut R,
    w: &mut W,
) -> Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let response_max = usize::MAX;
    let mut served = 0usize;
    loop {
        let header = if served == 0 {
            read_frame(r, "request header", HEADER_MAX, true).await?
        } else if let Some(idle) = cfg.idle_exit {
            match tokio::time::timeout(idle, read_frame(r, "request header", HEADER_MAX, true))
                .await
            {
                Ok(frame) => frame?,
                Err(_) => {
                    tracing::info!("met embed: no request for {} ms, exiting", idle.as_millis());
                    break;
                }
            }
        } else {
            break;
        };
        let Some(header) = header else {
            tracing::info!("met embed: input closed after {served} request(s)");
            break;
        };
        let body = read_frame(r, "request body", body_max, false)
            .await?
            .ok_or_else(|| frame_err("request body", "missing"))?;
        let (h, body) = prepare(cfg, &header, &body)?;
        let (status, resp) = dispatch(&router, body, response_max).await?;
        let (head, vectors) = split_response(cfg, h.count, status, &resp)?;
        write_frame(w, "response header", &head).await?;
        write_frame(w, "response vectors", &vectors).await?;
        w.flush().await?;
        served += 1;
    }
    Ok(())
}

/// 2026-10-24: Take fd 1 for the frames and point fd 1 at stderr, so no print anywhere in the
/// process can land inside a frame. Called by `main` before anything is written.
pub fn claim_stdout() -> Result<std::fs::File> {
    use std::os::fd::FromRawFd as _;
    // SAFETY: dup/dup2 on the process's own standard descriptors; the duplicate is owned by
    // the returned File alone, and fd 2 stays open for the process lifetime.
    let fd = unsafe { libc::dup(1) };
    anyhow::ensure!(
        fd >= 0,
        "met embed: dup(stdout) failed: {}",
        std::io::Error::last_os_error()
    );
    // SAFETY: as above.
    let rc = unsafe { libc::dup2(2, 1) };
    anyhow::ensure!(
        rc >= 0,
        "met embed: dup2(stderr, stdout) failed: {}",
        std::io::Error::last_os_error()
    );
    // SAFETY: `fd` is a fresh descriptor this function owns and hands over exactly once.
    Ok(unsafe { std::fs::File::from_raw_fd(fd) })
}

/// 2026-10-24: The `met embed` tail of `serve`: the loaded model must embed; then the router
/// answers frames on stdin and the claimed stdout until [`serve_frames`] returns.
pub async fn run(
    host: std::sync::Arc<crate::main_modules::model_host::ModelHost>,
    body_limit: usize,
    cfg: EmbedStdio,
    out: std::fs::File,
) -> Result<()> {
    let state = host
        .current()
        .ok_or_else(|| anyhow::anyhow!("met embed: no model was loaded"))?;
    if state.embedding.is_none() {
        let e = error!(ErrorCode::EmbedNotSupported, state.model_name.clone());
        anyhow::bail!(crate::scheduler::coded(&e));
    }
    drop(state);
    tracing::info!("met embed: ready for frames on stdin");
    let router = crate::main_modules::serve_router::router(host, body_limit);
    let mut out = tokio::fs::File::from_std(out);
    let served = serve_frames(
        router,
        &cfg,
        u32::try_from(body_limit)?,
        &mut tokio::io::stdin(),
        &mut out,
    )
    .await;
    if let Err(e) = served {
        // 2026-10-24: The coded error must be stderr's last line (the host reports it). The
        // engine's threads keep logging their teardown through `exit`'s handlers, so this
        // ends the process at once (`_exit`); the device memory and descriptors go back with
        // it. `eprintln!` is unbuffered, so the line is written before.
        eprintln!("Error: {}", coded_text(&e));
        // SAFETY: `_exit` takes no pointers and never returns; skipping destructors and atexit
        // handlers is the point (the kernel reclaims every resource of the process).
        unsafe { libc::_exit(1) }
    }
    Ok(())
}

/// 2026-10-24: `[0xNNNN Name] text` for a registry error, else the error's own text.
pub fn coded_text(e: &anyhow::Error) -> String {
    match e.downcast_ref::<metrale_error::Error>() {
        Some(x) => crate::scheduler::coded(x),
        None => format!("{e:#}"),
    }
}

#[cfg(test)]
#[path = "embed_stdio_tests.rs"]
mod tests;
