// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `met circuit diff --batch`: the diff over multi-sequence decode. For each batch
//! width `n`, `n` sequences are prefilled one by one and then decoded together through
//! `decode_batch` (padded to the graph ladder), under the same forwards and controls as the
//! single-sequence diff; each step's logits are the `n` real rows, compared byte for byte.
//!
//! Owner: server CLI.
//! Invariants:
//! - Every run after the reference is fed the reference's per-row tokens, so all runs decode
//!   the same inputs at every row.
//! - The detection control changes one row's token halfway; its difference must be seen.

use anyhow::{Result, ensure};
use metrale_model_engine::traits::{ForwardSelect, Model, SequenceState};
use serde::Serialize;

use super::{Comparison, Run, Timing, argmax_bf16, compare, logits, timing};

/// 2026-09-28: The prompts of a batch of `n`: token ids from a 64-bit LCG over
/// `[64, vocab - 64)`, prompt `i` `12 + 7 * (i % 16)` tokens long, so rows sit at different
/// positions and block tables.
pub(crate) fn batch_prompts(n: usize, vocab: usize) -> Vec<Vec<u32>> {
    let mut x: u64 = 0xba7c_0928 ^ n as u64;
    let span = (vocab.saturating_sub(128)).max(1) as u64;
    (0..n)
        .map(|i| {
            (0..12 + 7 * (i % 16))
                .map(|_| {
                    x = x
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    64 + ((x >> 33) % span) as u32
                })
                .collect()
        })
        .collect()
}

/// 2026-09-28: How a batch run picks each step's input tokens.
pub(crate) enum BatchFeed<'a> {
    /// 2026-09-28: Each row's argmax of the previous logits.
    Greedy,
    /// 2026-09-28: These tokens per row, `steps + 1` each.
    Forced(&'a [Vec<u32>]),
}

/// 2026-09-28: The rows' tokens and the run, its prefill and step logits concatenated row by
/// row.
fn run_batch(
    model: &dyn Model,
    prompts: &[Vec<u32>],
    steps: usize,
    feed: BatchFeed<'_>,
) -> Result<(Vec<Vec<u32>>, Run)> {
    let n = prompts.len();
    let row_bytes = model.vocab_size() * 2;
    let mut seqs: Vec<SequenceState> = Vec::with_capacity(n);
    let result = (|| {
        let mut prefill = Vec::with_capacity(n * row_bytes);
        let mut tokens: Vec<Vec<u32>> = Vec::with_capacity(n);
        for (i, p) in prompts.iter().enumerate() {
            seqs.push(model.alloc_sequence()?);
            let l = logits(model, model.prefill(p, &mut seqs[i], 0)?)?;
            tokens.push(vec![match feed {
                BatchFeed::Greedy => argmax_bf16(&l),
                BatchFeed::Forced(t) => t[i][0],
            }]);
            prefill.extend_from_slice(&l);
        }
        let (mut out, mut enqueue_ms, mut step_ms) = (Vec::new(), Vec::new(), Vec::new());
        for step in 0..steps {
            let now: Vec<u32> = tokens.iter().map(|t| t[step]).collect();
            let mut refs: Vec<&mut SequenceState> = seqs.iter_mut().collect();
            let t0 = std::time::Instant::now();
            let ptr = model.decode_batch(&now, &mut refs, 0)?;
            let t1 = std::time::Instant::now();
            let mut l = vec![0u8; n * row_bytes];
            model.copy_logits_to_host(ptr, &mut l)?;
            enqueue_ms.push((t1 - t0).as_secs_f64() * 1e3);
            step_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            for (i, t) in tokens.iter_mut().enumerate() {
                t.push(match feed {
                    BatchFeed::Greedy => argmax_bf16(&l[i * row_bytes..(i + 1) * row_bytes]),
                    BatchFeed::Forced(f) => f[i][step + 1],
                });
            }
            out.push(l);
        }
        Ok((
            tokens,
            Run {
                prefill,
                steps: out,
                tokens: Vec::new(),
                enqueue_ms,
                step_ms,
            },
        ))
    })();
    for s in &mut seqs {
        model.free_sequence(s)?;
    }
    result
}

/// 2026-09-29: The rows whose prefill logits differ between two runs. The prefill is legacy
/// in every run, so such a row starts its decode from another state and its steps are not a
/// test of the forward.
pub(crate) fn prefill_rows_differ(reference: &Run, run: &Run, row_bytes: usize) -> Vec<usize> {
    reference
        .prefill
        .chunks(row_bytes)
        .zip(run.prefill.chunks(row_bytes))
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect()
}

/// 2026-09-29: How one row's prefill logits differ from the reference's.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct RowDelta {
    pub row: usize,
    /// 2026-09-29: Bytes that differ.
    pub bytes: usize,
    /// 2026-09-29: Largest absolute difference of a logit.
    pub max_abs: f32,
    /// 2026-09-29: Both sides pick the same greedy token.
    pub argmax_equal: bool,
}

/// 2026-09-29: [`RowDelta`] of every row in `rows`.
pub(crate) fn row_deltas(
    reference: &Run,
    run: &Run,
    row_bytes: usize,
    rows: &[usize],
) -> Vec<RowDelta> {
    let val = |c: &[u8]| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16);
    rows.iter()
        .map(|&i| {
            let a = &reference.prefill[i * row_bytes..(i + 1) * row_bytes];
            let b = &run.prefill[i * row_bytes..(i + 1) * row_bytes];
            RowDelta {
                row: i,
                bytes: a.iter().zip(b).filter(|(x, y)| x != y).count(),
                max_abs: a
                    .chunks_exact(2)
                    .zip(b.chunks_exact(2))
                    .map(|(x, y)| (val(x) - val(y)).abs())
                    .fold(0.0, f32::max),
                argmax_equal: argmax_bf16(a) == argmax_bf16(b),
            }
        })
        .collect()
}

/// 2026-09-29: `run` without the rows in `skip`, prefill and steps alike.
fn without_rows(run: &Run, row_bytes: usize, skip: &[usize]) -> Run {
    let keep = |bytes: &[u8]| -> Vec<u8> {
        bytes
            .chunks(row_bytes)
            .enumerate()
            .filter(|(i, _)| !skip.contains(i))
            .flat_map(|(_, r)| r.iter().copied())
            .collect()
    };
    Run {
        prefill: keep(&run.prefill),
        steps: run.steps.iter().map(|s| keep(s)).collect(),
        tokens: Vec::new(),
        enqueue_ms: Vec::new(),
        step_ms: Vec::new(),
    }
}

/// 2026-09-29: `run` against `reference` over the rows whose prefill agrees, and those that
/// do not.
pub(crate) fn compare_rows(
    variant: &str,
    reference: &Run,
    run: &Run,
    row_bytes: usize,
) -> (Comparison, Vec<usize>) {
    let skip = prefill_rows_differ(reference, run, row_bytes);
    let c = compare(
        variant,
        &without_rows(reference, row_bytes, &skip),
        &without_rows(run, row_bytes, &skip),
    );
    (c, skip)
}

/// 2026-09-28: One batch width's results.
#[derive(Debug, Serialize)]
pub(crate) struct WidthReport {
    pub rows: usize,
    /// 2026-09-29: The row whose token the detection control changed.
    pub changed_row: usize,
    pub comparisons: Vec<Comparison>,
    /// 2026-09-29: Per comparison, the rows left out because their prefill differs.
    pub prefill_rows_differ: Vec<Vec<usize>>,
    /// 2026-09-29: Per comparison, how each of those rows differs.
    pub prefill_row_deltas: Vec<Vec<RowDelta>>,
    pub timings: Vec<Timing>,
    pub detection_control: Comparison,
}

/// 2026-09-28: Run the reference and every forward in `forwards` at each width.
pub(crate) fn diff_widths(
    model: &dyn Model,
    widths: &[usize],
    steps: usize,
    forwards: &[(&'static str, ForwardSelect)],
) -> Result<Vec<WidthReport>> {
    ensure!(!forwards.is_empty(), "no forward to compare");
    let mut out = Vec::with_capacity(widths.len());
    for &n in widths {
        ensure!(n >= 2, "a batch width is at least 2 rows");
        let prompts = batch_prompts(n, model.vocab_size());
        let row_bytes = model.vocab_size() * 2;
        model.set_forward(&ForwardSelect::Legacy)?;
        let (ref_tokens, reference) = run_batch(model, &prompts, steps, BatchFeed::Greedy)?;
        let (mut comparisons, mut timings) = (Vec::new(), vec![timing("legacy", &reference)]);
        let (mut prefill_rows, mut deltas) = (Vec::new(), Vec::new());
        for (name, sel) in forwards {
            model.set_forward(sel)?;
            let (_, r) = run_batch(model, &prompts, steps, BatchFeed::Forced(&ref_tokens))?;
            let (c, skipped) = compare_rows(name, &reference, &r, row_bytes);
            deltas.push(row_deltas(&reference, &r, row_bytes, &skipped));
            prefill_rows.push(skipped);
            tracing::info!("circuit diff: {n} rows {name}: {c:?}");
            timings.push(timing(name, &r));
            comparisons.push(c);
        }
        // 2026-09-28: The control runs under the last forward, still selected.
        let mut changed = ref_tokens.clone();
        let (row, at) = (n / 2, steps / 2);
        changed[row][at] = (changed[row][at] + 1) % model.vocab_size() as u32;
        let (_, r) = run_batch(model, &prompts, steps, BatchFeed::Forced(&changed))?;
        let (control, control_skipped) = compare_rows(
            "last forward, one row's token changed",
            &reference,
            &r,
            row_bytes,
        );
        deltas.push(row_deltas(&reference, &r, row_bytes, &control_skipped));
        prefill_rows.push(control_skipped);
        out.push(WidthReport {
            rows: n,
            changed_row: row,
            comparisons,
            prefill_rows_differ: prefill_rows,
            prefill_row_deltas: deltas,
            timings,
            detection_control: control,
        });
    }
    model.set_forward(&ForwardSelect::Legacy)?;
    Ok(out)
}

/// 2026-09-29: Why a batch report fails; empty for a pass. A comparison leaving out more than
/// an eighth of the rows (prefill disagreement) fails as too thin to judge, and so does a
/// detection control whose changed row was left out.
pub(crate) fn batch_failures(widths: &[WidthReport]) -> Vec<String> {
    let mut out = Vec::new();
    for w in widths {
        for (i, skipped) in w.prefill_rows_differ.iter().enumerate() {
            if skipped.len() * 8 > w.rows {
                out.push(format!(
                    "{} rows, comparison {i}: {} rows left out for a differing prefill",
                    w.rows,
                    skipped.len()
                ));
            }
        }
        if w.prefill_rows_differ
            .last()
            .is_some_and(|s| s.contains(&w.changed_row))
        {
            out.push(format!(
                "{} rows: the detection control's changed row {} was left out",
                w.rows, w.changed_row
            ));
        }
        for c in &w.comparisons {
            if !c.prefill_equal || c.mismatched_steps > 0 {
                out.push(format!(
                    "{} rows, {}: prefill equal {}, {} of {} steps differ (first at {:?})",
                    w.rows,
                    c.variant,
                    c.prefill_equal,
                    c.mismatched_steps,
                    c.steps,
                    c.first_mismatch
                ));
            }
        }
        if w.detection_control.mismatched_steps == 0 {
            out.push(format!(
                "{} rows: the detection control saw no difference after changing a token",
                w.rows
            ));
        }
    }
    out
}
