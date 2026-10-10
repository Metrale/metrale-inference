// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Tests for token-exact TTFT prompts, over a real byte-level BPE
//! tokenizer trained here on the committed fixture and loaded from its
//! `tokenizer.json` through the production loader, and through the gate
//! against a loopback endpoint.
//!
//! Owner: bench, ttft.
//! Invariants: no test reads a tokenizer from outside the repository.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, OnceLock};

use tokenizers::models::TrainerWrapper;
use tokenizers::models::bpe::{BPE, BpeTrainerBuilder};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;

use super::*;
use crate::artifacts::ArtifactStore;
use crate::benchmark::BenchmarkDescriptor;
use crate::benchmarks::ttft::long_prompt::tests::{contents, endpoint, measured_contents, run};
use crate::benchmarks::ttft::long_prompt::{LONG_32K, TASK_LINE, fixture_text};
use crate::benchmarks::ttft::{COLD_DESCRIPTOR, HIGH_ISL_WARM_DESCRIPTOR, WARM_DESCRIPTOR};
use crate::dynamic::DynBenchmark;
use crate::params::{ParamValue, ParamValues};
use crate::plugin::PluginHandle;

fn fixture() -> &'static str {
    fixture_text(LONG_32K).expect("the committed fixture")
}

/// 2026-10-10: A 2000-token byte-level BPE (the GPT/Qwen/GLM family's scheme)
/// trained on the fixture, saved once per test process.
fn tokenizer_path() -> &'static PathBuf {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let mut tok = tokenizers::Tokenizer::new(BPE::default());
        tok.with_pre_tokenizer(Some(ByteLevel::default()));
        tok.with_decoder(Some(ByteLevel::default()));
        let mut trainer: TrainerWrapper = BpeTrainerBuilder::new()
            .vocab_size(2000)
            .min_frequency(2)
            .show_progress(false)
            .initial_alphabet(ByteLevel::alphabet().into_iter().collect())
            .build()
            .into();
        tok.train(&mut trainer, fixture().lines()).expect("train");
        let dir = std::env::temp_dir().join(format!("metrale-tokprompt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        tok.save(dir.join("tokenizer.json"), false).expect("save");
        dir
    })
}

fn codec() -> tokenizers::Tokenizer {
    load(&tokenizer_path().display().to_string()).expect("the production loader reads it")
}

fn count(prompt: &str) -> usize {
    Codec::encode(&codec(), prompt).expect("encode").len()
}

#[test]
fn every_target_is_met_exactly_and_keeps_the_layout() {
    let mut p = ExactPrompts::new(codec(), fixture(), 4096, 7).expect("source");
    let mut targets: Vec<usize> = (40..400).step_by(13).collect();
    targets.extend([1000, 1024, 2047, 4096]);
    for target in targets {
        for mode in [Mode::Cold, Mode::Warm] {
            let text = p.prompt(mode, target, 3).expect("exact");
            assert_eq!(count(&text), target, "{target} tokens, {}", &text[..40]);
            assert!(text.starts_with(&format!("[{}", tag(mode, 7, target, 3))));
            assert!(text.ends_with(TASK_LINE));
        }
    }
}

/// 2026-10-10: A character codec with two merges: `y` + newline and space + `a` are one token
/// each. On a body of `xy` pairs, cuts ending in `y` merge with the newline after the body, so
/// neighbouring cuts skip every other count, as GLM-5.3's tokenizer did on the filler at 4096.
struct Skipping;

const Y_NL: u32 = 0x11_0000;
const SP_A: u32 = 0x11_0001;

impl Codec for Skipping {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let chars: Vec<char> = text.chars().collect();
        let mut out = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            match (chars[i], chars.get(i + 1)) {
                ('y', Some('\n')) => (out.push(Y_NL), i += 2),
                (' ', Some('a')) => (out.push(SP_A), i += 2),
                (c, _) => (out.push(c as u32), i += 1),
            };
        }
        Ok(out)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        Ok(ids
            .iter()
            .map(|&id| match id {
                Y_NL => "y\n".to_string(),
                SP_A => " a".to_string(),
                c => char::from_u32(c).expect("char").to_string(),
            })
            .collect())
    }
}

#[test]
fn a_count_every_cut_skips_is_reached_through_a_tag_suffix() {
    let body = "xy".repeat(400);
    let mut p = ExactPrompts::new(Skipping, &body, 600, 3).expect("source");
    let count = |cut: usize, target: usize| {
        let text = content(&body[..cut], &tag(Mode::Warm, 3, target, 0));
        Skipping.encode(&text).expect("encode").len()
    };
    let skipped = (200..400)
        .find(|&t| !(0..=600).any(|cut| count(cut, t) == t))
        .expect("the codec makes every cut skip some count");
    let text = p
        .prompt(Mode::Warm, skipped, 0)
        .expect("reached via a suffix");
    assert_eq!(Skipping.encode(&text).expect("encode").len(), skipped);
    assert!(
        text.starts_with(&format!("[{} a", tag(Mode::Warm, 3, skipped, 0))),
        "{}",
        &text[..40]
    );
}

#[test]
fn a_short_body_is_repeated_until_it_covers_the_target() {
    let short = "Call me Ishmael. Some years ago, never mind how long precisely.";
    assert!(count(short) < 50);
    let mut p = ExactPrompts::new(codec(), short, 700, 1).expect("source");
    let text = p.prompt(Mode::Cold, 700, 0).expect("exact");
    assert_eq!(count(&text), 700);
    assert!(text.matches("Call me Ishmael").count() > 10);
}

#[test]
fn cold_prompts_differ_from_their_first_tag_token() {
    let tok = codec();
    let mut p = ExactPrompts::new(codec(), fixture(), 512, 99).expect("source");
    let ids: Vec<Vec<u32>> = (0..40)
        .map(|i| {
            let text = p.prompt(Mode::Cold, 512, i).expect("exact");
            Codec::encode(&tok, &text).expect("encode")
        })
        .collect();
    let opener = Codec::encode(&tok, "[").expect("encode").len();
    for (a, x) in ids.iter().enumerate() {
        for y in &ids[a + 1..] {
            let first_diff = x.iter().zip(y).position(|(l, r)| l != r);
            assert!(
                first_diff.is_some_and(|d| d <= opener + 1),
                "two cold prompts share a prefix past the tag's first token: {first_diff:?}"
            );
        }
    }
    let other_run = ExactPrompts::new(codec(), fixture(), 512, 100)
        .expect("source")
        .prompt(Mode::Cold, 512, 0)
        .expect("exact");
    let first = &ids[0];
    let theirs = Codec::encode(&tok, &other_run).expect("encode");
    assert_ne!(
        first[..opener + 2],
        theirs[..opener + 2],
        "another salt shares sample 0's tag"
    );
}

#[test]
fn warm_resends_one_byte_identical_prompt_per_target() {
    let mut p = ExactPrompts::new(codec(), fixture(), 1024, 5).expect("source");
    let first = p.prompt(Mode::Warm, 1024, 0).expect("exact");
    for i in 1..6 {
        assert_eq!(p.prompt(Mode::Warm, 1024, i).expect("exact"), first);
    }
    let other = ExactPrompts::new(codec(), fixture(), 1024, 6)
        .expect("source")
        .prompt(Mode::Warm, 1024, 0)
        .expect("exact");
    assert_eq!(other, first, "the warm tag does not depend on the run salt");
    assert_ne!(p.prompt(Mode::Warm, 512, 0).expect("exact"), first);
}

#[test]
fn a_target_below_the_tag_and_task_line_is_an_error_not_a_longer_prompt() {
    let mut p = ExactPrompts::new(codec(), fixture(), 64, 0).expect("source");
    let floor = count(&content("", &tag(Mode::Cold, 0, 3, 0)));
    let e = p.prompt(Mode::Cold, 3, 0).expect_err("unreachable");
    assert!(e.to_string().contains("exactly 3 tokens"), "{e}");
    assert!(floor > 3);
}

#[test]
fn the_server_count_must_be_present_and_cover_the_message() {
    let mut p = ExactPrompts::new(codec(), fixture(), 64, 0).expect("source");
    assert!(
        p.admit(256, 0, 0)
            .unwrap_err()
            .to_string()
            .contains("no usage.prompt_tokens")
    );
    assert!(
        p.admit(256, 0, 255)
            .unwrap_err()
            .to_string()
            .contains("truncated")
    );
    p.admit(256, 1, 270).expect("covers");
    p.admit(256, 2, 262).expect("covers");
    p.admit(1024, 0, 1030).expect("covers");
    assert_eq!(p.reported(256), Some(262));
    assert_eq!(p.reported(512), None);
    assert_eq!(p.smallest_reported(), Some(262));
}

#[test]
fn a_tokenizer_value_names_the_file_or_a_directory_holding_it() {
    let dir = tokenizer_path();
    let file = dir.join("tokenizer.json");
    assert_eq!(tokenizer_file(&dir.display().to_string()).unwrap(), file);
    assert_eq!(tokenizer_file(&file.display().to_string()).unwrap(), file);
    let empty = dir.join("empty");
    std::fs::create_dir_all(&empty).expect("dir");
    let e = tokenizer_file(&empty.display().to_string()).unwrap_err();
    assert!(e.to_string().contains("no tokenizer.json"), "{e}");
    let e = tokenizer_file("/nonexistent/metrale/tok").unwrap_err();
    assert!(e.to_string().contains("no such file"), "{e}");
    assert!(configure(NO_TOKENIZER, fixture(), 64, 0).unwrap().is_none());
}

/// 2026-10-10: A gate built by its descriptor and configured with `tokenizer`.
async fn gate(
    descriptor: &'static BenchmarkDescriptor,
    target: crate::plugin::TargetEndpoint,
    set: &[(&str, ParamValue)],
) -> Box<dyn DynBenchmark> {
    let mut g = descriptor.build();
    let (tx, rx) = std::sync::mpsc::channel();
    std::mem::forget(rx);
    let dir = std::env::temp_dir().join(format!("metrale-tokgate-{}", std::process::id()));
    let mut values = ParamValues::defaults(&g.parameters());
    values.set(
        "tokenizer",
        ParamValue::Text(tokenizer_path().display().to_string()),
    );
    values.set("update_baseline", ParamValue::Bool(false));
    for (k, v) in set {
        values.set(*k, v.clone());
    }
    g.configure(&values).expect("configure");
    let handle = PluginHandle::new(
        1,
        target,
        ArtifactStore::with_root(dir),
        tx,
        Arc::new(AtomicBool::new(false)),
    );
    g.load(handle).await.expect("load");
    g
}

#[tokio::test]
async fn a_tokenized_cold_gate_sends_exact_unique_prompts_and_records_the_servers_count() {
    let (target, bodies) = endpoint(Some(310)).await;
    let set = [
        ("prompt_lengths", ParamValue::IntList(vec![300])),
        ("repeats", ParamValue::Int(3)),
    ];
    let mut g = gate(&COLD_DESCRIPTOR, target, &set).await;
    let done = run(&mut g).await.expect("completes");
    let sent = measured_contents(&bodies);
    assert_eq!(sent.len(), 3);
    assert!(sent.iter().all(|c| count(c) == 300));
    assert_eq!(
        sent.iter().collect::<std::collections::BTreeSet<_>>().len(),
        3
    );
    assert_eq!(done.metrics["prompt_tokens"], 310.0);
    assert_eq!(done.metrics["samples"], 3.0);
    assert!(done.metrics["min_ms"] <= done.metrics["median_ms"]);
    assert_eq!(done.metrics["cached_prompt_tokens"], 0.0);
    let body = &bodies.lock().expect("bodies")[1];
    assert_eq!(body["stream_options"]["include_usage"], true);
}

#[tokio::test]
async fn a_tokenized_warm_gate_primes_and_resends_the_same_exact_prompt() {
    let (target, bodies) = endpoint(Some(140)).await;
    let set = [
        ("prompt_lengths", ParamValue::IntList(vec![128])),
        ("repeats", ParamValue::Int(2)),
    ];
    let mut g = gate(&WARM_DESCRIPTOR, target, &set).await;
    run(&mut g).await.expect("completes");
    let sent = measured_contents(&bodies);
    assert_eq!(sent.len(), 4, "prime + measure per sample");
    assert!(sent.iter().all(|c| c == &sent[0] && count(c) == 128));
}

#[tokio::test]
async fn a_server_count_below_the_exact_message_invalidates_the_run() {
    let (target, bodies) = endpoint(Some(127)).await;
    let set = [
        ("prompt_lengths", ParamValue::IntList(vec![128])),
        ("repeats", ParamValue::Int(2)),
    ];
    let mut g = gate(&COLD_DESCRIPTOR, target, &set).await;
    let e = run(&mut g).await.expect_err("invalid");
    assert!(e.to_string().contains("truncated"), "{e}");
    assert_eq!(
        contents(&bodies).len(),
        2,
        "the warm-up, then the first sample stops the run"
    );
}

#[tokio::test]
async fn a_tokenized_high_isl_gate_cuts_the_fixture_to_min_prompt_tokens() {
    let (target, bodies) = endpoint(Some(2060)).await;
    let set = [("min_prompt_tokens", ParamValue::Int(2048))];
    let mut g = gate(&HIGH_ISL_WARM_DESCRIPTOR, target, &set).await;
    let done = run(&mut g).await.expect("completes");
    let sent = measured_contents(&bodies);
    assert_eq!(sent.len(), 2);
    assert!(
        sent.iter()
            .all(|c| count(c) == 2048 && c.contains("MOBY-DICK"))
    );
    assert_eq!(done.metrics["prompt_tokens"], 2060.0);
}
