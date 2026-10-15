// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the client's pure half: the streamed-response decoder (chunk
//! and read boundaries anywhere), the endpoint, the salted prompt, the
//! request bytes, the JSONL record and the client-to-server join.

mod common;

use common::ttft::*;
use common::{assert_close, assert_err};
use metprof::core::ttft::cells::select;
use metprof::core::ttft::client::{
    Endpoint, Intent, join, parse_records, parse_url, record, records_jsonl, render_prompt,
    request, salt_text,
};
use metprof::core::ttft::log::parse_serve_log;
use metprof::core::ttft::sse::{Read, decode_stream};
use serde_json::{Value, json};

/// The whole response re-cut into reads at `cuts` (byte offsets), the k-th
/// read arriving at (k + 1) ms.
fn recut(reads: &[Read], cuts: &[usize]) -> Vec<Read> {
    let all: Vec<u8> = reads.iter().flat_map(|r| r.bytes.clone()).collect();
    let mut edges = vec![0];
    edges.extend_from_slice(cuts);
    edges.push(all.len());
    edges
        .windows(2)
        .enumerate()
        .map(|(k, w)| Read {
            at_ns: (k as u64 + 1) * 1_000_000,
            bytes: all[w[0]..w[1]].to_vec(),
        })
        .collect()
}

#[test]
fn a_streamed_reply_decodes_to_ttft_usage_and_finish() {
    let s = decode_stream(&ok_reply(100, 412.25, 2_000_000)).unwrap();
    assert_close(s.ttft_client_ms, 4.0);
    assert_eq!(
        s.first_delta, "content",
        "the role-only frame is not a token"
    );
    assert_close(s.e2e_ms, 12.0);
    assert_eq!(
        (s.usage.prompt_tokens, s.usage.completion_tokens),
        (100, Some(8))
    );
    assert_eq!(
        (s.usage.ttft_ms, s.usage.cached_tokens),
        (Some(412.25), Some(0))
    );
    assert_eq!(s.finish_reason, "length");
}

#[test]
fn a_frame_split_across_reads_is_timed_when_its_last_byte_arrives() {
    let whole = ok_reply(100, 1.0, 1);
    let bytes: Vec<u8> = whole.iter().flat_map(|r| r.bytes.clone()).collect();
    let at = |needle: &str| {
        bytes
            .windows(needle.len())
            .position(|w| w == needle.as_bytes())
            .unwrap()
    };
    let hi = at("\"hi\"");
    for cut in [
        hi + 1,
        hi - 30,
        at("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\"") + 3,
    ] {
        let s = decode_stream(&recut(&whole, &[cut])).unwrap();
        assert_close(s.ttft_client_ms, 2.0);
    }
    let byte_by_byte: Vec<usize> = (1..bytes.len()).collect();
    let s = decode_stream(&recut(&whole, &byte_by_byte)).unwrap();
    let chunk_last = hi
        + bytes[hi..]
            .windows(4)
            .position(|w| w == b"\n\n\r\n")
            .unwrap()
        + 3;
    assert_close(s.ttft_client_ms, (chunk_last + 1) as f64);
    assert_eq!(s.usage.ttft_ms, Some(1.0));
}

#[test]
fn reasoning_or_a_tool_call_is_a_first_token_and_empty_text_is_not() {
    let frames = |d: &str| {
        vec![
            delta(r#"{"content":""}"#),
            delta(d),
            finish("stop"),
            usage(5, 1, 1.0, 0),
            DONE.to_string(),
        ]
    };
    let s = decode_stream(&response(
        &frames(r#"{"reasoning_content":"r"}"#),
        1_000_000,
    ))
    .unwrap();
    assert_eq!((s.first_delta, s.ttft_client_ms), ("reasoning", 2.0));
    let tool = r#"{"tool_calls":[{"index":0,"function":{"name":"f"}}]}"#;
    let s = decode_stream(&response(&frames(tool), 1_000_000)).unwrap();
    assert_eq!(s.first_delta, "tool_calls");
    let none = response(&frames(r#"{"tool_calls":[]}"#), 1);
    assert_err(
        decode_stream(&none),
        "no content, reasoning or tool-call delta",
    );
}

#[test]
fn a_plain_body_without_chunking_decodes_too() {
    let body = [
        delta(r#"{"content":"x"}"#),
        usage(5, 1, 1.0, 0),
        DONE.to_string(),
    ]
    .concat();
    let head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n";
    let reads = vec![
        Read {
            at_ns: 3_000_000,
            bytes: head.as_bytes().to_vec(),
        },
        Read {
            at_ns: 5_000_000,
            bytes: body.into_bytes(),
        },
    ];
    let s = decode_stream(&reads).unwrap();
    assert_eq!((s.ttft_client_ms, s.finish_reason.as_str()), (5.0, ""));
}

#[test]
fn errors_truncation_and_missing_usage_are_refused() {
    let err_frame = "data: {\"error\":{\"message\":\"boom\"}}\n\n".to_string();
    assert_err(
        decode_stream(&response(&[err_frame], 1)),
        "error mid-stream: .*boom",
    );
    let mut reads = response(&[delta(r#"{"content":"x"}"#), usage(5, 1, 1.0, 0)], 1);
    assert_err(
        decode_stream(&reads),
        "neither \\[DONE\\] nor a finish_reason",
    );
    reads.pop();
    assert_err(decode_stream(&reads), "ended without its last chunk");
    let no_usage = response(&[delta(r#"{"content":"x"}"#), DONE.to_string()], 1);
    assert_err(decode_stream(&no_usage), "no usage block");
    let bad_usage = response(
        &["data: {\"usage\":{\"completion_tokens\":1}}\n\n".to_string()],
        1,
    );
    assert_err(
        decode_stream(&bad_usage),
        "usage block without prompt_tokens",
    );
    let not_json = response(&["data: {oops\n\n".to_string()], 1);
    assert_err(decode_stream(&not_json), "not JSON: '\\{oops'");
    let status = vec![Read {
        at_ns: 1,
        bytes: b"HTTP/1.1 400 Bad Request\r\ncontent-length: 9\r\n\r\nbad input".to_vec(),
    }];
    assert_err(
        decode_stream(&status),
        "^HTTP 'HTTP/1.1 400 Bad Request': 'bad input'",
    );
    let no_head = vec![Read {
        at_ns: 1,
        bytes: b"HTTP/1.1 200 OK\r\n".to_vec(),
    }];
    assert_err(decode_stream(&no_head), "before a complete HTTP head");
    let bad_size = vec![Read {
        at_ns: 1,
        bytes: format!("{HEAD_CHUNKED}zz\r\n").into_bytes(),
    }];
    assert_err(decode_stream(&bad_size), "bad chunk size line 'zz'");
    let bad_crlf = vec![Read {
        at_ns: 1,
        bytes: format!("{HEAD_CHUNKED}2\r\nabXY").into_bytes(),
    }];
    assert_err(
        decode_stream(&bad_crlf),
        "chunk of 2 bytes is not followed by CRLF",
    );
}

#[test]
fn the_endpoint_is_http_host_and_port_only() {
    let ep = parse_url("http://box-a:8888/").unwrap();
    assert_eq!(
        ep,
        Endpoint {
            host: "box-a".to_string(),
            port: 8888
        }
    );
    assert_eq!(parse_url("http://127.0.0.1:1").unwrap().port, 1);
    assert_err(parse_url("https://h:1"), "must start with http://");
    assert_err(parse_url("http://h"), "must name a port");
    assert_err(parse_url("http://h:0"), "bad port '0'");
    assert_err(parse_url("http://h:1/v1"), "bad port '1/v1'");
    assert_err(parse_url("http://u@h:1"), "bad host");
}

#[test]
fn the_salt_is_fixed_width_and_replaces_the_one_placeholder() {
    assert_eq!(salt_text(42, 6), "000042");
    assert_eq!(salt_text(1_234_567, 4), "4567");
    assert_eq!(
        salt_text(u64::MAX, 19),
        format!("{:019}", u64::MAX % 10u64.pow(19))
    );
    assert_eq!(render_prompt("x {salt} y", "0042").unwrap(), "x 0042 y");
    assert_err(render_prompt("x y", "1"), "exactly once, found 0");
    assert_err(render_prompt("{salt}{salt}", "1"), "exactly once, found 2");
}

#[test]
fn the_request_is_one_streamed_completion_with_usage() {
    let cfg = parse_config(&config_with_client()).unwrap();
    let ep = Endpoint {
        host: "h".to_string(),
        port: 9,
    };
    let bytes = request(&ep, cfg.client().unwrap(), 8, "say \"hi\"");
    let text = String::from_utf8(bytes).unwrap();
    let (head, body) = text.split_once("\r\n\r\n").unwrap();
    assert!(head.starts_with("POST /v1/chat/completions HTTP/1.1\r\nHost: h:9\r\n"));
    assert!(head.contains(&format!("\r\nContent-Length: {}\r\n", body.len())));
    assert!(head.ends_with("Connection: close"));
    let v: Value = serde_json::from_str(body).unwrap();
    assert_eq!(
        v,
        json!({"model": "m", "messages": [{"role": "user", "content": "say \"hi\""}], "stream": true,
               "stream_options": {"include_usage": true}, "max_tokens": 8, "temperature": 0.0,
               "top_p": 1.0, "seed": 42})
    );
}

fn rec(prompt: u64, intent: Intent, usage_ttft: f64) -> metprof::core::ttft::client::ClientRecord {
    let cfg = parse_config(CONFIG).unwrap();
    let cell = cfg
        .cells
        .iter()
        .find(|c| c.prompt_tokens == prompt)
        .unwrap();
    let s = decode_stream(&ok_reply(prompt, usage_ttft, 1_000_000)).unwrap();
    record(cell, 0, intent, "0001", &s).unwrap()
}

#[test]
fn records_refuse_a_foreign_prompt_and_round_trip_through_jsonl() {
    let cfg = parse_config(CONFIG).unwrap();
    let s = decode_stream(&ok_reply(101, 1.0, 1)).unwrap();
    assert_err(
        record(&cfg.cells[0], 3, Intent::Cold, "1", &s),
        "a cold rep 3: the server counted 101 prompt tokens, the cell is 100",
    );
    let rs = vec![rec(100, Intent::Warmup, 1.0), rec(200, Intent::Warm, 2.5)];
    let text = records_jsonl(&rs);
    assert_eq!(text.lines().count(), 2);
    assert_eq!(parse_records(&format!("\n{text}\n")).unwrap(), rs);
    let line = text.lines().next().unwrap();
    assert_err(
        parse_records(&line.replace("\"rep\":0,", "")),
        "line 1: missing key 'rep'",
    );
    assert_err(
        parse_records(&line.replace("\"rep\":0,", "\"rep\":0,\"x\":1,")),
        "unknown key 'x'",
    );
    assert_err(
        parse_records(&line.replace("\"warmup\"", "\"hot\"")),
        "mode must be warmup, cold or warm",
    );
    assert_err(parse_records("{"), "line 1: not JSON");
}

#[test]
fn records_join_in_order_within_the_done_line_rounding() {
    let cfg = parse_config(CONFIG).unwrap();
    let text = join_lines(&[
        req(100, 8, false, 7, 450.0),
        req(200, 8, false, 9, 700.0),
        req(100, 8, false, 9, 460.0),
        req(100, 8, true, 9, 90.0),
    ]);
    let log = parse_serve_log(&text).unwrap();
    let sel = select(&log, &cfg).unwrap();
    let recs = vec![
        rec(100, Intent::Warmup, 1.0),
        rec(100, Intent::Cold, 449.95),
        rec(100, Intent::Cold, 460.04),
        rec(100, Intent::Warm, 90.0),
    ];
    assert_eq!(
        join(&recs, &sel).unwrap(),
        vec![(0, 1), (2, 2), (3, 3)],
        "the c200 request is skipped"
    );
    let off = vec![rec(100, Intent::Cold, 450.06)];
    assert_err(
        join(&off, &sel),
        "client record 1 .a cold rep 0.: no later measured server request with 100 prompt tokens and TTFT 450.060",
    );
    let order = vec![rec(100, Intent::Cold, 460.0), rec(100, Intent::Cold, 450.0)];
    assert_err(join(&order, &sel), "client record 2");
    let intent = vec![rec(100, Intent::Warm, 450.0)];
    assert_err(
        join(&intent, &sel),
        "line 1 was cold .cache-hit marker., not warm",
    );
    let mut no_key = rec(100, Intent::Cold, 450.0);
    no_key.usage_ttft_ms = None;
    assert_err(join(&[no_key], &sel), "no usage_ttft_ms");
}
