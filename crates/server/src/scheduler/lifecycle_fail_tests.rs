// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: A failed scheduler step reaches the client as an error that carries the CERF
//! code `SequenceFailed` (0x0200), on the blocking and the streaming sink, never as a finish;
//! and no error arm anywhere in the scheduler retires a sequence without `fail_sequence`
//! (the shape that sent "1 token, HTTP 200, finish_reason stop" when the GLM mHC exchange
//! failed at decode, mhc_site.rs:129).
//!
//! Owner: server scheduler tests.
//! Invariants: the scan reads the crate's sources at test time; nothing is written.

use std::path::{Path, PathBuf};

use super::*;
use crate::api::inference_types::StreamEvent;

const CAUSE: &str = "GLM mHC plan: the communicator has no one-shot exchange for 8192 bytes";
const CODE: &str = "[0x0200 SequenceFailed]";

fn finish(a: &mut ActiveSeq) {
    finish_sequence(
        &SchedIo::for_test_with(std::sync::Arc::new(StubModel)),
        a,
        MAX_SEQ_LEN,
    );
}

#[test]
fn a_blocking_client_gets_the_code_and_the_cause() {
    let (mut a, mut rx) = test_seq(vec![5], 500, None, 10);
    fail_sequence(&mut a, format!("bootstrap decode error: {CAUSE}"));
    finish(&mut a);
    let err = match rx.try_recv().expect("the caller must be answered") {
        Ok(_) => panic!("a failed step must answer with an error, not a completion"),
        Err(e) => format!("{e:#}"),
    };
    assert!(
        err.starts_with(CODE),
        "the frame must lead with the code: {err}"
    );
    assert!(err.contains(CAUSE), "the frame must carry the cause: {err}");
}

#[test]
fn a_streaming_client_gets_an_error_frame_and_no_done() {
    let (mut a, _blocking) = test_seq(vec![5], 500, None, 10);
    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    a.sink = ResponseSink::Streaming(tx);
    fail_sequence(&mut a, format!("bootstrap decode error: {CAUSE}"));
    finish(&mut a);
    match rx.try_recv().expect("the stream must get a terminal frame") {
        StreamEvent::Error(m) => {
            assert!(m.starts_with(CODE) && m.contains(CAUSE), "{m}");
        }
        StreamEvent::Done { .. } => panic!("a failed step must not end the stream as Done"),
        _ => panic!("the terminal frame must be the error"),
    }
    assert!(rx.try_recv().is_err(), "no Done may follow the error frame");
}

/// 2026-10-23: The lines (1-based) of `src` where `<x>.finished = true;` follows a
/// `tracing::error!` within the 5 lines before it: an error arm that retires the sequence as a
/// normal finish.
fn silent_finishes(src: &str) -> Vec<usize> {
    let lines: Vec<&str> = src.lines().collect();
    (0..lines.len())
        .filter(|&i| {
            let t = lines[i].trim();
            t.ends_with(".finished = true;")
                && t.split('.')
                    .next()
                    .is_some_and(|v| v.chars().all(|c| c.is_alphanumeric() || c == '_'))
                && lines[i.saturating_sub(5)..i]
                    .iter()
                    .any(|l| l.contains("tracing::error!"))
        })
        .map(|i| i + 1)
        .collect()
}

fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).expect("readable scheduler dir") {
        let p = e.expect("readable entry").path();
        let name = p
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();
        if p.is_dir() {
            sources(&p, out);
        } else if name.ends_with(".rs") && !name.contains("tests") {
            out.push(p);
        }
    }
}

#[test]
fn the_scan_flags_a_silent_error_arm_and_passes_fail_sequence() {
    let silent = "Err(e) => {\n    tracing::error!(\"decode: {e:#}\");\n    for a in batch.iter_mut() {\n        a.finished = true;\n    }\n}";
    assert_eq!(
        silent_finishes(silent),
        vec![4],
        "the scan must see through a loop"
    );
    let failed = "Err(e) => {\n    tracing::error!(\"decode: {e:#}\");\n    fail_sequence(a, format!(\"decode: {e:#}\"));\n}";
    assert!(silent_finishes(failed).is_empty());
}

#[test]
fn no_scheduler_error_arm_retires_a_sequence_as_a_normal_finish() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/scheduler");
    let mut files = Vec::new();
    sources(&dir, &mut files);
    assert!(
        files.len() >= 40,
        "the scan found only {} scheduler sources",
        files.len()
    );
    let mut hits = Vec::new();
    for f in &files {
        let src = std::fs::read_to_string(f).expect("readable source");
        for line in silent_finishes(&src) {
            hits.push(format!(
                "{}:{line}",
                f.strip_prefix(&dir).unwrap_or(f).display()
            ));
        }
    }
    assert!(
        hits.is_empty(),
        "error arms that retire a sequence without fail_sequence (HTTP 200 over a failed \
         generation): {hits:?}"
    );
}
