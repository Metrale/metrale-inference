// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-27: Tests for the high-ISL TTFT source: the committed fixture, the
//! prompt layout and tags, and the prompt-token check, driven through the
//! production client against a loopback endpoint.
//!
//! Owner: bench, ttft.
//! Invariants: the requests are real HTTP from `http::chat_stream` to a
//! loopback socket; only the server's replies are scripted.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use serde_json::Value;
use sha2::Digest;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::*;
use crate::artifacts::ArtifactStore;
use crate::benchmarks::baseline;
use crate::benchmarks::ttft::{
    COLD_DESCRIPTOR, HIGH_ISL_COLD_DESCRIPTOR, HIGH_ISL_COLD_MOE_DESCRIPTOR,
    HIGH_ISL_WARM_DESCRIPTOR, HIGH_ISL_WARM_MOE_DESCRIPTOR, WARM_DESCRIPTOR,
};
use crate::dynamic::DynBenchmark;
use crate::plugin::{PluginHandle, TargetEndpoint};
use crate::result::{BenchmarkResult, RunStatus};

/// 2026-09-27: `prompts/NOTICE.md` records the same digests and length;
/// `scripts/make_long_prompt.py --check` rebuilds the file from the download.
const FIXTURE_SHA256: &str = "4f961eabd9433e18166e795052b59239760d4e3cc7fa78c258d0bd1b623a1993";
const FIXTURE_BYTES: usize = 126_709;
/// 2026-09-27: The script's `warm_content_sha256`: the warm user message it
/// counted, so the layout it mirrors and `content` cannot drift apart.
const WARM_CONTENT_SHA256: &str =
    "c943228f01a771c1ddce0cf5c0cb0eb6f6bd8a495b19a19b82f2fd400fc31b26";

fn sha256(text: &str) -> String {
    metrale_closure::hex_lower(&sha2::Sha256::digest(text.as_bytes()))
}

/// 2026-09-27: A loopback endpoint that answers `/v1/models` and every chat
/// request with one streamed token, then a usage frame carrying
/// `prompt_tokens` (none when `None`). It keeps every request body.
async fn endpoint(prompt_tokens: Option<u64>) -> (TargetEndpoint, Arc<Mutex<Vec<Value>>>) {
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

/// 2026-09-27: The gate as the registry would build it, through its
/// descriptor's `ctor`, loaded against `target`. `repeats` overrides the
/// default when given.
async fn gate(
    descriptor: &'static BenchmarkDescriptor,
    target: TargetEndpoint,
    root: &str,
    repeats: Option<i64>,
) -> (Box<dyn DynBenchmark>, ArtifactStore) {
    let mut g = descriptor.build();
    let (tx, rx) = std::sync::mpsc::channel();
    std::mem::forget(rx);
    let dir = std::env::temp_dir().join(format!("metrale-hisl-{root}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let store = ArtifactStore::with_root(dir);
    let mut values = ParamValues::defaults(&g.parameters());
    if let Some(n) = repeats {
        values.set("repeats", ParamValue::Int(n));
    }
    g.configure(&values).expect("defaults configure");
    let handle = PluginHandle::new(
        1,
        target,
        store.clone(),
        tx,
        Arc::new(AtomicBool::new(false)),
    );
    g.load(handle).await.expect("load");
    (g, store)
}

async fn run(g: &mut Box<dyn DynBenchmark>) -> Result<BenchmarkResult> {
    loop {
        let frame = g.next().await?;
        if frame.status != RunStatus::Running {
            return Ok(frame);
        }
    }
}

/// 2026-09-27: The user messages sent, after checking that the first is the
/// run's one unmeasured warm-up and that no other request carries its tag.
fn measured_contents(bodies: &Mutex<Vec<Value>>) -> Vec<String> {
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

fn contents(bodies: &Mutex<Vec<Value>>) -> Vec<String> {
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

#[test]
fn the_fixture_is_the_committed_moby_dick_cut() {
    assert_eq!(LONG_32K_TEXT.len(), FIXTURE_BYTES);
    assert_eq!(sha256(LONG_32K_TEXT), FIXTURE_SHA256);
    assert!(LONG_32K_TEXT.starts_with("MOBY-DICK;\n"));
    assert!(LONG_32K_TEXT.ends_with("world of woe\n"));
    assert!(!LONG_32K_TEXT.contains('\r'));
    assert!(!LONG_32K_TEXT.contains("Gutenberg"));
}

#[test]
fn a_prompt_is_the_tag_then_the_fixture_then_the_task_line() {
    let long = LongPrompt {
        text: LONG_32K_TEXT,
        min_prompt_tokens: LONG_32K_TOKENS,
        salt: 42,
        prompt_tokens: None,
    };
    let cold = long.prompt(Mode::Cold, 7);
    assert_eq!(
        cold,
        format!("[{COLD_TAG_PREFIX}0000000000042007] {LONG_32K_TEXT}\n{TASK_LINE}")
    );
    let warm = long.prompt(Mode::Warm, 7);
    assert_eq!(warm, format!("[{WARM_TAG}] {LONG_32K_TEXT}\n{TASK_LINE}"));
    assert_eq!(sha256(&warm), WARM_CONTENT_SHA256);
}

#[test]
fn cold_tags_are_unique_per_sample_and_share_the_warm_tags_shape() {
    let digits = |tag: &str, prefix: &str| {
        let nonce = tag.strip_prefix(prefix).expect("prefix");
        assert_eq!(nonce.len(), NONCE_DIGITS, "{tag}");
        assert!(nonce.bytes().all(|b| b.is_ascii_digit()), "{tag}");
    };
    digits(WARM_TAG, "warm-32k-");
    assert_eq!(WARM_TAG.len(), COLD_TAG_PREFIX.len() + NONCE_DIGITS);
    let salt = fresh_salt();
    let mut seen = std::collections::BTreeSet::new();
    for sample in 0..200 {
        let t = tag(Mode::Cold, salt, sample);
        digits(&t, COLD_TAG_PREFIX);
        assert!(seen.insert(t), "sample {sample} repeats a tag");
        assert_eq!(tag(Mode::Warm, salt, sample), WARM_TAG);
    }
    assert_ne!(tag(Mode::Cold, 1, 0), tag(Mode::Cold, 2, 0));
}

#[test]
fn the_metric_is_the_smallest_count_seen() {
    let mut long = LongPrompt {
        text: LONG_32K_TEXT,
        min_prompt_tokens: 32_768,
        salt: 0,
        prompt_tokens: None,
    };
    for n in [32_770, 32_768, 32_769] {
        long.admit(0, n).expect("above the minimum");
    }
    assert_eq!(long.prompt_tokens, Some(32_768));
}

#[tokio::test]
async fn warm_resends_one_byte_identical_prompt_and_records_the_servers_count() {
    let (target, bodies) = endpoint(Some(32_768)).await;
    // 2026-09-27: Three samples exercise the per-sample tag and check; the
    // one-shot default would not repeat them.
    let (mut g, _) = gate(&HIGH_ISL_WARM_MOE_DESCRIPTOR, target, "warm", Some(3)).await;
    let done = run(&mut g).await.expect("the run completes");
    assert_eq!(done.status, RunStatus::Completed);
    assert_eq!(done.metrics["prompt_tokens"], 32_768.0);
    assert_eq!(done.metrics["samples"], 3.0);

    let sent = measured_contents(&bodies);
    // 2026-09-27: A priming request and a measured one per sample.
    assert_eq!(sent.len(), 6);
    let expected = content(LONG_32K_TEXT, WARM_TAG);
    assert!(sent.iter().all(|c| *c == expected));
    for body in bodies.lock().expect("bodies").iter() {
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
        assert_eq!(body["max_tokens"], 8);
        assert_eq!(body["temperature"], 0.0);
        assert_eq!(body["stream"], true);
    }
}

#[tokio::test]
async fn cold_sends_a_new_tag_at_the_start_of_every_sample() {
    let (target, bodies) = endpoint(Some(32_768)).await;
    let (mut g, _) = gate(&HIGH_ISL_COLD_DESCRIPTOR, target, "cold", Some(3)).await;
    let done = run(&mut g).await.expect("the run completes");
    assert_eq!(done.status, RunStatus::Completed);

    let sent = measured_contents(&bodies);
    assert_eq!(sent.len(), 3);
    let unique: std::collections::BTreeSet<_> = sent.iter().collect();
    assert_eq!(unique.len(), 3, "a cold tag repeated");
    for c in &sent {
        let (tag, _) = c[1..].split_once("] ").expect("a leading tag");
        assert!(c.starts_with(&format!("[{COLD_TAG_PREFIX}")), "{tag}");
        assert_eq!(*c, content(LONG_32K_TEXT, tag));
    }
}

#[tokio::test]
async fn a_short_server_count_makes_the_run_invalid_not_a_pass() {
    let (target, _) = endpoint(Some(32_767)).await;
    let (mut g, store) = gate(&HIGH_ISL_COLD_MOE_DESCRIPTOR, target, "short", None).await;
    let err = run(&mut g)
        .await
        .expect_err("a short prompt is not a measurement");
    assert!(
        format!("{err:#}").contains(
            "sample 0: the server counted 32767 prompt tokens, below min_prompt_tokens 32768"
        ),
        "{err:#}"
    );
    assert!(
        baseline::load_for(&store, HIGH_ISL_COLD_MOE_DESCRIPTOR.id, Some("test-model")).is_none()
    );
}

#[tokio::test]
async fn a_response_without_usage_is_an_error() {
    let (target, _) = endpoint(None).await;
    let (mut g, store) = gate(&HIGH_ISL_WARM_DESCRIPTOR, target, "nousage", None).await;
    let err = run(&mut g)
        .await
        .expect_err("an unverified prompt size is not a measurement");
    assert!(
        format!("{err:#}").contains("sample 0: the response carried no usage.prompt_tokens"),
        "{err:#}"
    );
    assert!(baseline::load_for(&store, HIGH_ISL_WARM_DESCRIPTOR.id, Some("test-model")).is_none());
}

#[test]
fn each_high_isl_gate_has_its_own_id_and_the_long_prompt_parameters() {
    let ids: Vec<_> = [
        &HIGH_ISL_COLD_DESCRIPTOR,
        &HIGH_ISL_WARM_DESCRIPTOR,
        &HIGH_ISL_COLD_MOE_DESCRIPTOR,
        &HIGH_ISL_WARM_MOE_DESCRIPTOR,
    ]
    .into_iter()
    .map(|d| {
        let b = d.build();
        assert_eq!(b.descriptor().id, d.id);
        assert!(d.expected_secs > 0, "{}", d.id);
        let specs = b.parameters();
        let values = ParamValues::defaults(&specs);
        values.validate_against(&specs).expect("defaults validate");
        assert_eq!(values.text("prompt").unwrap(), LONG_32K);
        assert_eq!(values.usize("min_prompt_tokens").unwrap(), LONG_32K_TOKENS);
        assert_eq!(values.usize("repeats").unwrap(), 1, "{}", d.id);
        assert!(!specs.iter().any(|s| s.key == "prompt_lengths"), "{}", d.id);
        d.id
    })
    .collect();
    assert_eq!(
        ids,
        [
            "high-isl-ttft-cold",
            "high-isl-ttft-warm",
            "high-isl-ttft-cold-moe",
            "high-isl-ttft-warm-moe"
        ]
    );
    // 2026-09-27: The synthetic gates keep their parameters unchanged.
    for d in [&WARM_DESCRIPTOR, &COLD_DESCRIPTOR] {
        let specs = d.build().parameters();
        assert!(specs.iter().any(|s| s.key == "prompt_lengths"), "{}", d.id);
        assert!(!specs.iter().any(|s| s.key == "prompt"), "{}", d.id);
        let values = ParamValues::defaults(&specs);
        assert_eq!(values.usize("repeats").unwrap(), 12, "{}", d.id);
        assert_eq!(values.float("median_limit_pct").unwrap(), 3.0, "{}", d.id);
        assert_eq!(values.float("p90_limit_pct").unwrap(), 5.0, "{}", d.id);
    }
}

/// 2026-09-27: The owner's one-shot instrument at the defaults: a warm-up,
/// then exactly one fresh cold request; its TTFT is both statistics.
#[tokio::test]
async fn cold_one_shot_is_a_warm_up_then_one_fresh_request() {
    let (target, bodies) = endpoint(Some(32_772)).await;
    let (mut g, store) = gate(&HIGH_ISL_COLD_MOE_DESCRIPTOR, target, "oneshot-cold", None).await;
    let done = run(&mut g).await.expect("the run completes");
    assert_eq!(done.status, RunStatus::Completed);
    assert_eq!(done.metrics["samples"], 1.0);
    assert_eq!(done.metrics["median_ms"], done.metrics["p90_ms"]);
    assert_eq!(done.metrics["prompt_tokens"], 32_772.0);
    let sent = measured_contents(&bodies);
    assert_eq!(sent.len(), 1);
    assert!(
        sent[0].starts_with(&format!("[{COLD_TAG_PREFIX}")),
        "{}",
        sent[0]
    );
    // 2026-09-27: The first run on a box has no baseline and stores this one.
    let stored = baseline::load_for(&store, HIGH_ISL_COLD_MOE_DESCRIPTOR.id, Some("test-model"))
        .expect("the first run is stored as the baseline");
    assert_eq!(stored.get("samples"), Some(1.0));
}

/// 2026-09-27: Warm at the defaults: a warm-up, then one priming request and
/// one measured byte-identical re-send.
#[tokio::test]
async fn warm_one_shot_is_a_warm_up_a_prime_and_one_measured_resend() {
    let (target, bodies) = endpoint(Some(32_772)).await;
    let (mut g, _) = gate(&HIGH_ISL_WARM_DESCRIPTOR, target, "oneshot-warm", None).await;
    let done = run(&mut g).await.expect("the run completes");
    assert_eq!(done.metrics["samples"], 1.0);
    let sent = measured_contents(&bodies);
    assert_eq!(sent, vec![content(LONG_32K_TEXT, WARM_TAG); 2]);
}

/// 2026-09-27: Each mode's one-shot limits: one sample makes the median and the
/// p90 one number, so both limits are equal.
#[test]
fn one_shot_limits_are_one_bound_per_mode() {
    for (d, limit) in [
        (&HIGH_ISL_COLD_DESCRIPTOR, 5.0),
        (&HIGH_ISL_COLD_MOE_DESCRIPTOR, 5.0),
        (&HIGH_ISL_WARM_DESCRIPTOR, 25.0),
        (&HIGH_ISL_WARM_MOE_DESCRIPTOR, 25.0),
    ] {
        let values = ParamValues::defaults(&d.build().parameters());
        assert_eq!(values.float("median_limit_pct").unwrap(), limit, "{}", d.id);
        assert_eq!(values.float("p90_limit_pct").unwrap(), limit, "{}", d.id);
    }
}

#[test]
fn a_warm_up_tag_has_the_measured_tags_shape_and_prefixes_no_measured_prompt() {
    let long = LongPrompt {
        text: LONG_32K_TEXT,
        min_prompt_tokens: LONG_32K_TOKENS,
        salt: 42,
        prompt_tokens: None,
    };
    let warm_up = warm_up_content(42);
    assert!(warm_up.starts_with(&format!("[{WARM_UP_TAG_PREFIX}0000000000000042] ")));
    for mode in [Mode::Cold, Mode::Warm] {
        assert!(!long.prompt(mode, 0).starts_with(&warm_up[..12]));
    }
    assert_ne!(warm_up_content(1), warm_up_content(2));
}
