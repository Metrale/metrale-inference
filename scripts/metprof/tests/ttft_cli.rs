// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `metprof ttft log|compare|client` through the real CLI over
//! in-memory ports, and the real socket adapter against an in-test server.

mod common;

use std::io::{Read as _, Write as _};
use std::net::TcpListener;
use std::thread;
use std::time::Duration;

use common::Mem;
use common::ttft::*;
use metprof::adapters::chat::{ClockSalt, TcpChat};
use metprof::core::ttft::client::{Endpoint, parse_records};
use metprof::core::ttft::sse::decode_stream;
use metprof::ports::{ChatTransport, SaltSource};
use serde_json::{Value, json};

fn log_a() -> String {
    join_lines(&[
        req(100, 32, false, 7, 900.0),
        req(100, 8, false, 7, 450.0),
        req(100, 8, false, 9, 470.0),
        req(100, 8, true, 9, 90.0),
    ])
}

#[test]
fn ttft_log_reports_counts_cells_and_requests() {
    let m = Mem::new().file("/log", &log_a()).file("/t.toml", CONFIG);
    let v = m.json(&["ttft", "log", "--log", "/log", "--config", "/t.toml"]);
    assert_eq!(
        v["counts"],
        json!({"requests": 4, "concurrent": 0, "unfinished": 0, "measured": 3, "unassigned": 0, "other_max_tokens": 1})
    );
    let server = v["server"].as_array().unwrap();
    assert_eq!(
        (server[0]["mode"].clone(), server[0]["median"].clone()),
        (json!("cold"), json!(460.0))
    );
    assert_eq!(server[0]["step"], json!({"n": 1, "median": 450.0}));
    assert_eq!(v["requests"][2]["mode"], json!("warm"));
    assert_eq!(v.get("client"), None);
    let text = m.run(&["ttft", "log", "--log", "/log", "--config", "/t.toml"]);
    assert_eq!(text.code, 0, "{}", text.stderr);
    assert!(text.stdout.contains("ttft_server_ms"));
}

#[test]
fn ttft_log_refuses_a_bad_log_naming_the_file() {
    let bad = format!("{}\n{}\n", start(100, 8), "Marconi SSM cache hit: 1 tokens");
    let m = Mem::new().file("/log", &bad).file("/t.toml", CONFIG);
    let err = m.refused(&["ttft", "log", "--log", "/log", "--config", "/t.toml"]);
    assert!(
        err.contains("/log: line 2: a cache-hit line in neither known format"),
        "{err}"
    );
    let err = Mem::new()
        .file("/t.toml", CONFIG)
        .refused(&["ttft", "log", "--log", "/x", "--config", "/t.toml"]);
    assert!(err.contains("cannot read /x"), "{err}");
}

#[test]
fn ttft_client_runs_warmups_cold_reps_and_warm_resends() {
    let m = Mem::new()
        .file("/t.toml", &config_with_client())
        .file("/p.txt", "id {salt} then text")
        .reply(ok_reply(100, 900.0, 1_000_000))
        .reply(ok_reply(100, 450.0, 3_000_000))
        .reply(ok_reply(100, 470.0, 5_000_000))
        .reply(ok_reply(100, 90.0, 1_000_000));
    let argv = [
        "--json",
        "ttft",
        "client",
        "--url",
        "http://h:9",
        "--config",
        "/t.toml",
        "--cell",
        "a",
        "--prompt",
        "/p.txt",
        "--cold",
        "2",
        "--warm",
        "1",
        "--out",
        "/c.jsonl",
    ];
    let out = m.run(&argv);
    assert_eq!(out.code, 0, "{}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(
        (v["cold"]["n"].clone(), v["cold"]["median"].clone()),
        (json!(2), json!(8.0))
    );
    assert_eq!(v["warm"]["median"], json!(2.0));
    let sent = m.sent.borrow();
    let bodies: Vec<Value> = sent
        .iter()
        .map(|(ep, req, t)| {
            assert_eq!(
                (ep, *t),
                (
                    &Endpoint {
                        host: "h".to_string(),
                        port: 9
                    },
                    30.0
                )
            );
            let text = String::from_utf8(req.clone()).unwrap();
            serde_json::from_str(text.split_once("\r\n\r\n").unwrap().1).unwrap()
        })
        .collect();
    let prompt = |i: usize| {
        bodies[i]["messages"][0]["content"]
            .as_str()
            .unwrap()
            .to_string()
    };
    let max = |i: usize| bodies[i]["max_tokens"].as_u64().unwrap();
    assert_eq!((0..4).map(max).collect::<Vec<_>>(), vec![32, 8, 8, 8]);
    assert_eq!(
        (0..4).map(prompt).collect::<Vec<_>>(),
        vec![
            "id 0001 then text",
            "id 0002 then text",
            "id 0003 then text",
            "id 0003 then text",
        ]
    );
    assert_eq!(out.files.len(), 1);
    assert_eq!(out.files[0].0, "/c.jsonl");
    let recs = parse_records(std::str::from_utf8(&out.files[0].1).unwrap()).unwrap();
    let modes: Vec<(&str, u64, &str)> = recs
        .iter()
        .map(|r| (r.intent.name(), r.rep, r.salt.as_str()))
        .collect();
    assert_eq!(
        modes,
        vec![
            ("warmup", 0, "0001"),
            ("cold", 0, "0002"),
            ("cold", 1, "0003"),
            ("warm", 0, "0003")
        ]
    );

    let joined = Mem::new()
        .file("/log", &log_a())
        .file("/t.toml", CONFIG)
        .bytes("/c.jsonl", &out.files[0].1);
    let v = joined.json(&[
        "ttft", "log", "--log", "/log", "--config", "/t.toml", "--client", "/c.jsonl",
    ]);
    assert_eq!(v["joined"], json!(3));
    assert_eq!(v["client"][0]["step"], json!({"n": 1, "median": 6.0}));
    assert_eq!(v["requests"][0]["ttft_client_ms"], json!(6.0));
}

#[test]
fn ttft_client_refuses_a_foreign_prompt_a_bad_template_and_transport_errors() {
    let base = || {
        Mem::new()
            .file("/t.toml", &config_with_client())
            .file("/p.txt", "{salt}")
    };
    let argv = |cold: &'static str| {
        vec![
            "ttft",
            "client",
            "--url",
            "http://h:9",
            "--config",
            "/t.toml",
            "--cell",
            "a",
            "--prompt",
            "/p.txt",
            "--cold",
            cold,
            "--warm",
            "0",
            "--out",
            "/c.jsonl",
        ]
    };
    let err = base().reply(ok_reply(101, 1.0, 1)).refused(&argv("1"));
    assert!(
        err.contains("cell a warmup rep 0: the server counted 101 prompt tokens"),
        "{err}"
    );
    let err = base().reply_err("connection refused").refused(&argv("1"));
    assert!(
        err.contains("cell a warmup rep 0: connection refused"),
        "{err}"
    );
    let err = base()
        .reply(ok_reply(100, 1.0, 1))
        .reply(vec![])
        .refused(&argv("1"));
    assert!(
        err.contains("cell a cold rep 0: the connection closed before a complete HTTP head"),
        "{err}"
    );
    let err = base().refused(&argv("0"));
    assert!(err.contains("--cold must be at least 1"), "{err}");
    let m = Mem::new()
        .file("/t.toml", &config_with_client())
        .file("/p.txt", "no placeholder");
    assert!(
        m.refused(&argv("1"))
            .contains("/p.txt: the prompt must contain {salt} exactly once")
    );
    assert!(
        m.sent.borrow().is_empty(),
        "nothing is sent for a bad template"
    );
    let m = Mem::new().file("/t.toml", CONFIG).file("/p.txt", "{salt}");
    assert!(m.refused(&argv("1")).contains("no [client] table"));
}

#[test]
fn ttft_compare_gives_a_verdict_per_cell() {
    let slower = join_lines(&[
        req(100, 8, false, 7, 480.0),
        req(100, 8, false, 9, 490.0),
        req(100, 8, true, 9, 90.0),
    ]);
    let m = Mem::new()
        .file("/a", &log_a())
        .file("/b", &slower)
        .file("/t.toml", CONFIG);
    let v = m.json(&[
        "ttft",
        "compare",
        "--a",
        "/a",
        "--b",
        "/b",
        "--config",
        "/t.toml",
        "--min-reps",
        "2",
    ]);
    let cells = v["cells"].as_array().unwrap();
    assert_eq!(
        (cells[0]["mode"].clone(), cells[0]["verdict"].clone()),
        (json!("cold"), json!("loss"))
    );
    assert_eq!(cells[0]["delta"], json!(25.0));
    assert_eq!(cells[1]["verdict"], json!("too-few"));
    let err = m.refused(&[
        "ttft",
        "compare",
        "--a",
        "/a",
        "--b",
        "/b",
        "--config",
        "/t.toml",
        "--min-reps",
        "0",
    ]);
    assert!(err.contains("--min-reps must be at least 1"), "{err}");
    let err = m.refused(&[
        "ttft", "compare", "--a", "/a", "--b", "/b", "--config", "/t.toml",
    ]);
    assert!(err.contains("--min-reps"), "{err}");
}

/// A one-connection server: reads the request head and body, then writes
/// `parts` with `pause` before each and closes (stopping early once the
/// client has gone).
fn serve_once(parts: Vec<Vec<u8>>, pause: Duration) -> (u16, thread::JoinHandle<Vec<u8>>) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    let h = thread::spawn(move || {
        let (mut s, _) = l.accept().unwrap();
        let mut got = Vec::new();
        let mut buf = [0u8; 4096];
        while !got.ends_with(b"}") {
            let n = s.read(&mut buf).unwrap();
            assert!(n > 0, "the client closed before sending its body");
            got.extend_from_slice(&buf[..n]);
        }
        for p in parts {
            thread::sleep(pause);
            if s.write_all(&p).and_then(|()| s.flush()).is_err() {
                break;
            }
        }
        got
    });
    (port, h)
}

#[test]
fn the_socket_adapter_times_reads_from_the_write() {
    let reads = ok_reply(100, 5.0, 1);
    let parts: Vec<Vec<u8>> = reads.into_iter().map(|r| r.bytes).collect();
    let (port, h) = serve_once(parts, Duration::from_millis(30));
    let ep = Endpoint {
        host: "127.0.0.1".to_string(),
        port,
    };
    let req = b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}".to_vec();
    let got = TcpChat.exchange(&ep, &req, 10.0).unwrap();
    assert_eq!(h.join().unwrap(), req, "the request arrives whole");
    let s = decode_stream(&got).unwrap();
    assert!(
        s.ttft_client_ms >= 60.0,
        "two pauses precede the content frame: {}",
        s.ttft_client_ms
    );
    assert!(s.e2e_ms >= s.ttft_client_ms + 60.0, "{s:?}");
    assert!(got.windows(2).all(|w| w[0].at_ns <= w[1].at_ns));
}

#[test]
fn the_socket_adapter_enforces_the_deadline_and_reports_refusal() {
    let (port, h) = serve_once(
        vec![HEAD_CHUNKED.as_bytes().to_vec(), vec![b'0']],
        Duration::from_millis(400),
    );
    let ep = Endpoint {
        host: "127.0.0.1".to_string(),
        port,
    };
    let err = TcpChat.exchange(&ep, b"{}", 0.2).unwrap_err();
    assert!(err.0.contains("no complete response within 0.2 s"), "{err}");
    drop(h);
    let closed = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = closed.local_addr().unwrap().port();
    drop(closed);
    let err = TcpChat
        .exchange(
            &Endpoint {
                host: "127.0.0.1".to_string(),
                port,
            },
            b"{}",
            1.0,
        )
        .unwrap_err();
    assert!(
        err.0
            .starts_with(&format!("cannot connect to 127.0.0.1:{port}")),
        "{err}"
    );
}

#[test]
fn clock_salts_do_not_repeat() {
    let mut seen: Vec<u64> = (0..1000).map(|_| ClockSalt.fresh()).collect();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), 1000);
}
