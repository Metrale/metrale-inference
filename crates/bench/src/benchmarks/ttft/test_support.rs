// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Test support shared by the TTFT source tests (`long_prompt_tests.rs`,
//! `token_prompt_tests.rs`): a loopback endpoint that scripts the server's replies, a driver
//! that runs a gate to its terminal frame, and readers of the user messages it sent. Moved
//! here from `long_prompt_tests.rs` unchanged.
//!
//! Owner: bench, ttft.
//! Invariants: the requests are real HTTP from `http::chat_stream` to a loopback socket.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::long_prompt::WARM_UP_TAG_PREFIX;
use crate::dynamic::DynBenchmark;
use crate::plugin::TargetEndpoint;
use crate::result::{BenchmarkResult, RunStatus};

/// 2026-09-27: A loopback endpoint that answers `/v1/models` and every chat
/// request with one streamed token, then a usage frame carrying
/// `prompt_tokens` (none when `None`). It keeps every request body.
pub(super) async fn endpoint(
    prompt_tokens: Option<u64>,
) -> (TargetEndpoint, Arc<Mutex<Vec<Value>>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback");
    let port = listener.local_addr().expect("address").port();
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let seen = bodies.clone();
    tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let seen = seen.clone();
            tokio::spawn(async move {
                let mut request = Vec::new();
                let mut buf = [0u8; 64 * 1024];
                let head_end = loop {
                    let n = socket.read(&mut buf).await.expect("read");
                    request.extend_from_slice(&buf[..n]);
                    if let Some(i) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                        break i + 4;
                    }
                    assert!(n > 0, "request ended inside its headers");
                };
                let head = String::from_utf8_lossy(&request[..head_end]).to_string();
                if head.starts_with("GET ") {
                    let _ = socket
                        .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
                        .await;
                    return;
                }
                let length: usize = head
                    .lines()
                    .find_map(|l| l.strip_prefix("Content-Length: "))
                    .expect("content length")
                    .trim()
                    .parse()
                    .expect("numeric content length");
                while request.len() < head_end + length {
                    let n = socket.read(&mut buf).await.expect("read body");
                    assert!(n > 0, "request ended inside its body");
                    request.extend_from_slice(&buf[..n]);
                }
                let body: Value =
                    serde_json::from_slice(&request[head_end..head_end + length]).expect("json");
                seen.lock().expect("bodies").push(body);
                let usage = match prompt_tokens {
                    Some(n) => format!(
                        "data: {{\"choices\":[],\"usage\":{{\"prompt_tokens\":{n},\"completion_tokens\":1}}}}\n\n"
                    ),
                    None => String::new(),
                };
                let reply = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n\
                     data: {{\"choices\":[{{\"delta\":{{\"content\":\"Ish\"}},\"finish_reason\":null}}]}}\n\n\
                     data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"length\"}}]}}\n\n\
                     {usage}data: [DONE]\n\n"
                );
                let _ = socket.write_all(reply.as_bytes()).await;
            });
        }
    });
    (TargetEndpoint::local(port, "test-model"), bodies)
}

pub(super) async fn run(g: &mut Box<dyn DynBenchmark>) -> Result<BenchmarkResult> {
    loop {
        let frame = g.next().await?;
        if frame.status != RunStatus::Running {
            return Ok(frame);
        }
    }
}

/// 2026-09-27: The user messages sent, after checking that the first is the
/// run's one unmeasured warm-up and that no other request carries its tag.
pub(super) fn measured_contents(bodies: &Mutex<Vec<Value>>) -> Vec<String> {
    let sent = contents(bodies);
    let (first, rest) = sent.split_first().expect("a warm-up request");
    assert!(
        first.starts_with(&format!("[{WARM_UP_TAG_PREFIX}")),
        "{first}"
    );
    assert!(
        first.len() < 1024,
        "the warm-up is short: {} bytes",
        first.len()
    );
    assert!(
        rest.iter().all(|c| !c.contains(WARM_UP_TAG_PREFIX)),
        "a measured request carried the warm-up tag"
    );
    rest.to_vec()
}

pub(super) fn contents(bodies: &Mutex<Vec<Value>>) -> Vec<String> {
    bodies
        .lock()
        .expect("bodies")
        .iter()
        .map(|b| {
            b["messages"][0]["content"]
                .as_str()
                .expect("content")
                .to_string()
        })
        .collect()
}
