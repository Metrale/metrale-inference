// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: `met circuit diff --swap`: the sequence swap (`kv_swap_out` / `kv_swap_in`, the
//! scheduler's spill) under each forward. Each run prefills a prompt, decodes half the steps,
//! spills the sequence to a record, frees it, restores the record into a new sequence as the
//! scheduler's resume does (`scheduler/lifecycle.rs`), and decodes the rest. Every run is fed
//! the reference run's tokens.
//!
//! Compared with the legacy reference: each forward's record byte for byte, and every step's
//! logits before and after the spill. Two crossings prove the record format is shared: the
//! circuit restores legacy's record, and the controls change one byte of the record before a
//! restore (the logits after it must differ, or the comparison proves nothing).
//!
//! Owner: server CLI (FEATURES workstream).
//! Invariants:
//! - PASS needs every comparison equal and the control different; the exit status is non-zero
//!   otherwise.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use metrale_model_engine::traits::{ForwardSelect, Model};
use serde::Serialize;

use super::{argmax_bf16, logits};

/// 2026-10-03: One spilled run: the record, and each step's logits (before then after it).
struct SwapRun {
    record: Vec<u8>,
    steps: Vec<Vec<u8>>,
    tokens: Vec<u32>,
}

/// 2026-10-03: How the run's record is restored.
enum Restore<'a> {
    /// 2026-10-03: The record this run wrote.
    Own,
    /// 2026-10-03: These bytes instead (another forward's record, or a changed one).
    Bytes(&'a [u8]),
}

fn spilled(
    model: &dyn Model,
    prompt: &[u32],
    steps: usize,
    feed: Option<&[u32]>,
    restore: Restore<'_>,
) -> Result<SwapRun> {
    let mut seq = model.alloc_sequence()?;
    let first = logits(model, model.prefill(prompt, &mut seq, 0)?)?;
    let mut tokens = vec![feed.map_or_else(|| argmax_bf16(&first), |f| f[0])];
    let mut out = Vec::with_capacity(steps);
    let mut record = Vec::new();
    let result = (|| -> Result<()> {
        for step in 0..steps {
            if step == steps / 2 {
                model.save_sequence_state(&seq, &mut record)?;
                let blocks = seq.block_table.len();
                let (saved_tokens, saved_len) = (seq.tokens.clone(), seq.seq_len);
                model.free_sequence(&mut seq)?;
                seq = model.alloc_sequence()?;
                let bytes = match &restore {
                    Restore::Own => record.as_slice(),
                    Restore::Bytes(b) => *b,
                };
                model.restore_sequence_state(&mut seq, blocks, &mut &bytes[..])?;
                seq.tokens = saved_tokens;
                seq.seq_len = saved_len;
            }
            let l = logits(model, model.decode(tokens[step], &mut seq, 0)?)?;
            tokens.push(feed.map_or_else(|| argmax_bf16(&l), |f| f[step + 1]));
            out.push(l);
        }
        Ok(())
    })();
    model.free_sequence(&mut seq)?;
    result?;
    Ok(SwapRun {
        record,
        steps: out,
        tokens,
    })
}

/// 2026-10-03: One run against the reference.
#[derive(Debug, Clone, Serialize)]
struct SwapComparison {
    variant: String,
    record_equal: bool,
    record_bytes: usize,
    steps_before_equal: bool,
    steps_after_equal: bool,
    first_mismatch: Option<usize>,
}

fn compare(variant: &str, reference: &SwapRun, run: &SwapRun, steps: usize) -> SwapComparison {
    let eq = |r: std::ops::Range<usize>| r.clone().all(|i| reference.steps[i] == run.steps[i]);
    SwapComparison {
        variant: variant.to_string(),
        record_equal: reference.record == run.record,
        record_bytes: run.record.len(),
        steps_before_equal: eq(0..steps / 2),
        steps_after_equal: eq(steps / 2..steps),
        first_mismatch: (0..steps).find(|&i| reference.steps[i] != run.steps[i]),
    }
}

#[derive(Debug, Serialize)]
struct SwapReport {
    steps: usize,
    prompt_tokens: usize,
    comparisons: Vec<SwapComparison>,
    controls: Vec<SwapComparison>,
    verdict: &'static str,
    reasons: Vec<String>,
}

/// 2026-10-03: Why a swap report fails; empty for a pass. A control must see a difference
/// after the spill.
fn failures(comparisons: &[SwapComparison], controls: &[SwapComparison]) -> Vec<String> {
    let mut out: Vec<String> = comparisons
        .iter()
        .filter(|c| !(c.record_equal && c.steps_before_equal && c.steps_after_equal))
        .map(|c| format!("{c:?}"))
        .collect();
    out.extend(
        controls
            .iter()
            .filter(|c| c.steps_after_equal)
            .map(|c| format!("control `{}` saw no difference after the spill", c.variant)),
    );
    out
}

/// 2026-10-03: Run `met circuit diff --swap`.
pub(super) fn swap_report(
    model: &dyn Model,
    prompt: &[u32],
    steps: usize,
    forwards: &[(&'static str, ForwardSelect)],
    out: &Path,
) -> Result<()> {
    ensure!(
        steps >= 4,
        "--swap needs at least 4 steps (two each side of the spill)"
    );
    model.set_forward(&ForwardSelect::Legacy)?;
    let reference = spilled(model, prompt, steps, None, Restore::Own)?;
    let feed = Some(reference.tokens.as_slice());
    let mut comparisons = Vec::new();
    for (name, sel) in forwards {
        model.set_forward(sel)?;
        let r = spilled(model, prompt, steps, feed, Restore::Own)?;
        comparisons.push(compare(name, &reference, &r, steps));
    }
    let (_, circuit) = forwards
        .iter()
        .find(|(n, _)| *n == "circuit")
        .context("no circuit forward")?;
    model.set_forward(circuit)?;
    let crossed = spilled(
        model,
        prompt,
        steps,
        feed,
        Restore::Bytes(&reference.record),
    )?;
    comparisons.push(compare(
        "circuit, legacy's record",
        &reference,
        &crossed,
        steps,
    ));
    let mut changed = reference.record.clone();
    let last = changed.len().checked_sub(1).context("an empty record")?;
    changed[last] ^= 0x40;
    let mut controls = Vec::new();
    for (name, sel) in [("legacy", &ForwardSelect::Legacy), ("circuit", circuit)] {
        model.set_forward(sel)?;
        let c = spilled(model, prompt, steps, feed, Restore::Bytes(&changed))?;
        controls.push(compare(
            &format!("{name}, record changed"),
            &reference,
            &c,
            steps,
        ));
    }
    model.set_forward(&ForwardSelect::Legacy)?;
    let reasons = failures(&comparisons, &controls);
    let report = SwapReport {
        steps,
        prompt_tokens: prompt.len(),
        comparisons,
        controls,
        verdict: if reasons.is_empty() { "PASS" } else { "FAIL" },
        reasons: reasons.clone(),
    };
    std::fs::write(out, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("writing {}", out.display()))?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !reasons.is_empty() {
        bail!("circuit swap diff FAILED:\n  {}", reasons.join("\n  "));
    }
    Ok(())
}

#[cfg(test)]
#[path = "circuit_diff_swap_tests.rs"]
mod circuit_diff_swap_tests;
