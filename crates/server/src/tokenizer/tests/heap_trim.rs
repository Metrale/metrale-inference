// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `--tokenizer-trim-min-tokens` through `ChatTokenizer::encode`: the ids are
//! the same with the trim on and off, and the trim runs at the threshold and above, never
//! below it, and never at 0.
//!
//! Owner: server (tokenizer) tests.
//! Invariants: none beyond the types.
//!
//! The counting trimmer stands in for `malloc_trim` in the threshold controls: what the real
//! trim returns depends on every other allocation in the test process, so only a call count
//! states exactly when the trim ran. The real trim runs in
//! `the_glibc_trim_leaves_the_ids_unchanged`.

use super::super::heap_trim::HeapTrim;
use super::super::{ChatTokenizer, HeapTrimPolicy, TemplateSource};
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

const WORDS: [&str; 4] = ["alpha", "beta", "gamma", "delta"];

#[derive(Default)]
struct CountingTrim(AtomicUsize);

impl HeapTrim for CountingTrim {
    fn trim(&self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// 2026-10-17: A word-level tokenizer over `WORDS` (one token a word), loaded with `policy`.
fn tokenizer(policy: HeapTrimPolicy) -> (tempfile::TempDir, ChatTokenizer) {
    let dir = tempfile::tempdir().unwrap();
    let vocab: HashMap<String, u32> = ["[UNK]"]
        .into_iter()
        .chain(WORDS)
        .zip(0u32..)
        .map(|(w, id)| (w.to_string(), id))
        .collect();
    let model = WordLevel::builder()
        .vocab(vocab.into_iter().collect())
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut hf = tokenizers::Tokenizer::new(model);
    hf.with_pre_tokenizer(Some(WhitespaceSplit));
    hf.save(dir.path().join("tokenizer.json"), false).unwrap();
    let t = ChatTokenizer::from_model_dir(
        dir.path(),
        0,
        false,
        "heap_trim_fixture",
        None,
        TemplateSource::Checkpoint,
        policy,
    )
    .unwrap();
    (dir, t)
}

fn text(tokens: usize) -> String {
    (0..tokens)
        .map(|i| WORDS[(i * 7 + i / 5) % WORDS.len()])
        .collect::<Vec<_>>()
        .join(" ")
}

fn counting(min_tokens: usize) -> (Arc<CountingTrim>, HeapTrimPolicy) {
    let trim = Arc::new(CountingTrim::default());
    let policy = HeapTrimPolicy::with_trimmer(min_tokens, trim.clone());
    (trim, policy)
}

fn digest(ids: &[u32]) -> u64 {
    let mut h = DefaultHasher::new();
    ids.hash(&mut h);
    h.finish()
}

#[test]
fn the_trim_runs_at_and_above_the_threshold_only() {
    let (trim, policy) = counting(1000);
    let (_dir, t) = tokenizer(policy);
    assert_eq!(t.encode(&text(999)).unwrap().len(), 999);
    assert_eq!(trim.0.load(Ordering::SeqCst), 0, "below the threshold");
    assert_eq!(t.encode(&text(1000)).unwrap().len(), 1000);
    assert_eq!(trim.0.load(Ordering::SeqCst), 1, "at the threshold");
    t.encode(&text(4000)).unwrap();
    assert_eq!(trim.0.load(Ordering::SeqCst), 2, "once per encode above it");
    t.encode("alpha").unwrap();
    assert_eq!(
        trim.0.load(Ordering::SeqCst),
        2,
        "a short prompt after a long one"
    );
}

#[test]
fn a_zero_threshold_never_trims() {
    let (trim, policy) = counting(0);
    let (_dir, t) = tokenizer(policy);
    for n in [0, 1, 50_000] {
        t.encode(&text(n)).unwrap();
    }
    assert_eq!(trim.0.load(Ordering::SeqCst), 0);
    assert_eq!(HeapTrimPolicy::process_heap(0).unwrap().min_tokens(), 0);
}

#[test]
fn the_ids_are_the_same_with_the_trim_on_and_off() {
    let (_off_dir, off) = tokenizer(HeapTrimPolicy::off());
    let (trim, policy) = counting(1);
    let (_on_dir, on) = tokenizer(policy);
    for n in [1, 256, 1024, 32_768, 200_000] {
        let body = text(n);
        let (a, b) = (off.encode(&body).unwrap(), on.encode(&body).unwrap());
        assert_eq!(a.len(), n);
        assert_eq!(digest(&a), digest(&b), "{n} tokens");
        assert_eq!(a, b, "{n} tokens");
    }
    assert_eq!(
        trim.0.load(Ordering::SeqCst),
        5,
        "the on side trimmed every encode"
    );
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[test]
fn the_glibc_trim_leaves_the_ids_unchanged() {
    let (_off_dir, off) = tokenizer(HeapTrimPolicy::off());
    let policy = HeapTrimPolicy::process_heap(1).unwrap();
    assert_eq!(policy.min_tokens(), 1);
    let (_on_dir, on) = tokenizer(policy);
    let body = text(100_000);
    assert_eq!(off.encode(&body).unwrap(), on.encode(&body).unwrap());
}
