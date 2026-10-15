// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: builders for synthetic serve logs, ttft configs and streamed
//! HTTP responses, in the exact line and wire formats the engine emits.

use metprof::core::ttft::sse::Read;

const TS: &str = "2026-01-01T00:00:00.000000Z";

pub fn start(prompt: u64, max_tokens: u64) -> String {
    format!(
        "{TS}  INFO met::scheduler::prefill_a_step: Chunked prefill start: {prompt} prompt tokens, chunk_size={prompt}, max_tokens={max_tokens}"
    )
}

/// The prefill-path cache-hit format.
pub fn hit(skipped: u64) -> String {
    format!(
        "{TS}  INFO met::scheduler::prefill_a: Marconi SSM cache hit: {skipped} tokens skipped (4 blocks), snapshot 1"
    )
}

/// The prefix-lookup cache-hit format.
pub fn hit_lookup(skipped: u64, to: u64) -> String {
    format!(
        "{TS}  INFO m::prefix_lookup: Marconi SSM cache hit: {skipped} tokens skipped (4 blocks), snapshot 2, replaying {} SSM tokens to reach {to}",
        to - skipped
    )
}

pub fn first(token: u64) -> String {
    format!("{TS}  INFO met::scheduler::prefill_a_step: Prefill first token: {token}")
}

pub fn done(tokens: u64, ttft_ms: f64) -> String {
    format!(
        "{TS}  INFO met::scheduler::mtp_accept_debug: Done: {tokens} tokens (length) 36.2 tok/s, TTFT={ttft_ms:.1}ms, serial=1.00 mtp=0.00"
    )
}

/// One whole solo request: start, optional hit, first token, Done.
pub fn req(prompt: u64, max_tokens: u64, warm: bool, token: u64, ttft_ms: f64) -> Vec<String> {
    let mut v = vec![start(prompt, max_tokens)];
    if warm {
        v.push(hit(prompt - 1));
    }
    v.push(first(token));
    v.push(done(max_tokens, ttft_ms));
    v
}

pub fn join_lines(parts: &[Vec<String>]) -> String {
    let mut s = parts.concat().join("\n");
    s.push('\n');
    s
}

/// Measured max_tokens 8, step token 7, cells a=100 and b=200 prompt tokens.
pub const CONFIG: &str = r#"schema = 1
[select]
max_tokens = 8
[class]
step_token_id = 7
[[cell]]
name = "a"
prompt_tokens = 100
[[cell]]
name = "b"
prompt_tokens = 200
"#;

pub const CLIENT: &str = r#"[client]
model = "m"
temperature = 0.0
top_p = 1.0
seed = 42
salt_digits = 4
timeout_s = 30
warmup_requests = 1
warmup_max_tokens = 32
"#;

pub fn config_with_client() -> String {
    format!("{CONFIG}{CLIENT}")
}

pub fn parse_config(text: &str) -> metprof::core::error::Result<metprof::core::ttft::TtftConfig> {
    let t = metprof::core::config::parse_toml(text.as_bytes()).expect("test config is TOML");
    metprof::core::ttft::ttft_config(&t)
}

/// A `data:` frame for one chunk whose first choice's delta is `delta`.
pub fn delta(delta: &str) -> String {
    format!("data: {{\"choices\":[{{\"index\":0,\"delta\":{delta},\"finish_reason\":null}}]}}\n\n")
}

pub fn finish(reason: &str) -> String {
    format!(
        "data: {{\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{reason}\"}}]}}\n\n"
    )
}

pub fn usage(prompt: u64, completion: u64, ttft_ms: f64, cached: u64) -> String {
    format!(
        "data: {{\"choices\":[],\"usage\":{{\"prompt_tokens\":{prompt},\"completion_tokens\":{completion},\"time_to_first_token_ms\":{ttft_ms},\"prompt_tokens_details\":{{\"cached_tokens\":{cached}}}}}}}\n\n"
    )
}

pub const DONE: &str = "data: [DONE]\n\n";

pub const HEAD_CHUNKED: &str =
    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n";

/// `body` as one HTTP chunk.
pub fn chunk(body: &str) -> String {
    format!("{:x}\r\n{body}\r\n", body.len())
}

pub const LAST_CHUNK: &str = "0\r\n\r\n";

/// A whole chunked response, one chunk per frame, delivered as one read per
/// frame `step_ns` apart (the head arrives with the first frame).
pub fn response(frames: &[String], step_ns: u64) -> Vec<Read> {
    let mut reads: Vec<Read> = frames
        .iter()
        .enumerate()
        .map(|(i, f)| Read {
            at_ns: (i as u64 + 1) * step_ns,
            bytes: chunk(f).into_bytes(),
        })
        .collect();
    reads[0].bytes = [HEAD_CHUNKED.as_bytes(), &reads[0].bytes].concat();
    reads.push(Read {
        at_ns: (frames.len() as u64 + 1) * step_ns,
        bytes: LAST_CHUNK.as_bytes().to_vec(),
    });
    reads
}

/// A normal reply: a role-only frame, a content frame, finish, usage, DONE.
pub fn ok_reply(prompt: u64, ttft_ms: f64, step_ns: u64) -> Vec<Read> {
    response(
        &[
            delta(r#"{"role":"assistant"}"#),
            delta(r#"{"content":"hi"}"#),
            finish("length"),
            usage(prompt, 8, ttft_ms, 0),
            DONE.to_string(),
        ],
        step_ns,
    )
}
