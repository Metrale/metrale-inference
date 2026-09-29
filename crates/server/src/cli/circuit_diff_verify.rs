// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: `met circuit diff --verify`: the diff over the single-sequence MTP verify
//! (`decode_verify_graphed{,_k3,_k4}`). For each `K`, one prompt is prefilled and then
//! verified step by step with `[last, d1 .. d_{K-1}]`, the drafts taken from the model's own
//! greedy continuation with one of them corrupted on every odd step, so full accepts and
//! each partial accept occur. After each verify the accepted prefix is committed and the
//! rejected rows rewound, as the scheduler does (`verify_k2_step.rs`, `verify_k4_verdict.rs`).
//! Each step's `K` logits rows are compared byte for byte.
//!
//! Owner: server CLI.
//! Invariants:
//! - Every run after the reference replays the reference's tokens and accept counts, so all
//!   runs verify the same inputs and commit the same prefixes.
//! - The detection control changes one draft halfway; its difference must be seen.

use anyhow::{Result, bail, ensure};
use metrale_model_engine::traits::{ForwardSelect, Model, SequenceState};
use serde::Serialize;

use super::{Comparison, Run, Timing, argmax_bf16, compare, logits, timing};

/// 2026-09-29: One verify step's inputs and its accepted-draft count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct Step {
    pub tokens: Vec<u32>,
    pub accepted: usize,
}

/// 2026-09-29: How a verify run picks each step.
enum Feed<'a> {
    /// 2026-09-29: Drafts from `greedy` (the model's continuation), one corrupted on odd steps;
    /// the accept count from the run's own argmaxes.
    Reference { greedy: &'a [u32] },
    /// 2026-09-29: These steps exactly.
    Replay(&'a [Step]),
}

/// 2026-09-29: The drafts of step `s` at committed position `p`: `greedy[p + 1 ..]`, with
/// draft `1 + s % (k - 1)` bumped by one on an odd step.
pub(crate) fn drafts(greedy: &[u32], p: usize, s: usize, k: usize, vocab: u32) -> Vec<u32> {
    (1..k)
        .map(|j| {
            let t = greedy.get(p + j).copied().unwrap_or(0);
            if s % 2 == 1 && j == 1 + s % (k - 1) {
                (t + 1) % vocab
            } else {
                t
            }
        })
        .collect()
}

/// 2026-09-29: Leading drafts the verify accepted: draft `j` stands when row `j - 1`'s argmax
/// is it.
pub(crate) fn accepted(tokens: &[u32], outs: &[u32]) -> usize {
    (1..tokens.len())
        .take_while(|&j| outs[j - 1] == tokens[j])
        .count()
}

fn verify(model: &dyn Model, tokens: &[u32], seq: &mut SequenceState) -> Result<Vec<u32>> {
    Ok(match tokens.len() {
        2 => model
            .decode_verify_graphed(&[tokens[0], tokens[1]], seq, 0)?
            .to_vec(),
        3 => model
            .decode_verify_graphed_k3(&[tokens[0], tokens[1], tokens[2]], seq, 0)?
            .to_vec(),
        4 => model
            .decode_verify_graphed_k4(&[tokens[0], tokens[1], tokens[2], tokens[3]], seq, 0)?
            .to_vec(),
        k => bail!("no single-sequence verify for K={k}"),
    })
}

/// 2026-09-29: The model's greedy continuation of `prompt`, `n` tokens, by plain decode.
fn continuation(model: &dyn Model, prompt: &[u32], n: usize) -> Result<Vec<u32>> {
    let mut seq = model.alloc_sequence()?;
    let result = (|| {
        let mut out = vec![argmax_bf16(&logits(
            model,
            model.prefill(prompt, &mut seq, 0)?,
        )?)];
        while out.len() < n {
            let l = logits(model, model.decode(out[out.len() - 1], &mut seq, 0)?)?;
            out.push(argmax_bf16(&l));
        }
        Ok(out)
    })();
    model.free_sequence(&mut seq)?;
    result
}

fn run_verify(
    model: &dyn Model,
    prompt: &[u32],
    k: usize,
    steps: usize,
    feed: Feed<'_>,
) -> Result<(Vec<Step>, Run)> {
    let vocab = model.vocab_size();
    let mut seq = model.alloc_sequence()?;
    let result = (|| {
        let prefill = logits(model, model.prefill(prompt, &mut seq, 0)?)?;
        let mut last = argmax_bf16(&prefill);
        let (mut p, mut log, mut out) = (0usize, Vec::new(), Vec::new());
        let (mut enqueue_ms, mut step_ms) = (Vec::new(), Vec::new());
        for s in 0..steps {
            let tokens = match feed {
                Feed::Reference { greedy } => std::iter::once(last)
                    .chain(drafts(greedy, p, s, k, vocab as u32))
                    .collect(),
                Feed::Replay(r) => r[s].tokens.clone(),
            };
            model.sync_secondary()?;
            let t0 = std::time::Instant::now();
            let outs = verify(model, &tokens, &mut seq)?;
            let t1 = std::time::Instant::now();
            let mut l = vec![0u8; k * vocab * 2];
            model.copy_logits_to_host(model.logits_buffer_ptr(), &mut l)?;
            enqueue_ms.push((t1 - t0).as_secs_f64() * 1e3);
            step_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            let na = match feed {
                Feed::Reference { .. } => accepted(&tokens, &outs),
                Feed::Replay(r) => r[s].accepted,
            };
            // 2026-09-29: As the scheduler does: rewind the rejected rows, commit the rest.
            let rejected = k - 1 - na;
            seq.seq_len -= rejected;
            let keep = seq.tokens.len() - rejected;
            seq.tokens.truncate(keep);
            model.commit_accepted_prefix(&mut seq, na + 1, k)?;
            if let Feed::Reference { .. } = feed {
                (last, p) = (outs[na], p + na + 1);
            }
            log.push(Step {
                tokens,
                accepted: na,
            });
            out.push(l);
        }
        model.sync_secondary()?;
        Ok((
            log,
            Run {
                prefill,
                steps: out,
                tokens: Vec::new(),
                enqueue_ms,
                step_ms,
            },
        ))
    })();
    model.free_sequence(&mut seq)?;
    result
}

/// 2026-09-29: One verify width's results.
#[derive(Debug, Serialize)]
pub(crate) struct VerifyReport {
    pub k: usize,
    /// 2026-09-29: The reference's accept count per step.
    pub accepted: Vec<usize>,
    pub comparisons: Vec<Comparison>,
    pub timings: Vec<Timing>,
    pub detection_control: Comparison,
}

/// 2026-09-29: Run the reference and every forward in `forwards` at each `K`.
pub(crate) fn diff_verify(
    model: &dyn Model,
    prompt: &[u32],
    ks: &[usize],
    steps: usize,
    forwards: &[(&'static str, ForwardSelect)],
) -> Result<Vec<VerifyReport>> {
    ensure!(!forwards.is_empty(), "no forward to compare");
    let mut out = Vec::with_capacity(ks.len());
    for &k in ks {
        model.set_forward(&ForwardSelect::Legacy)?;
        let greedy = continuation(model, prompt, steps * k + 1)?;
        let (log, reference) =
            run_verify(model, prompt, k, steps, Feed::Reference { greedy: &greedy })?;
        let (mut comparisons, mut timings) = (Vec::new(), vec![timing("legacy", &reference)]);
        for (name, sel) in forwards {
            model.set_forward(sel)?;
            let (_, r) = run_verify(model, prompt, k, steps, Feed::Replay(&log))?;
            let c = compare(name, &reference, &r);
            tracing::info!("circuit diff: verify K={k} {name}: {c:?}");
            timings.push(timing(name, &r));
            comparisons.push(c);
        }
        // 2026-09-29: The control runs under the last forward, still selected.
        let mut changed = log.clone();
        let at = steps / 2;
        changed[at].tokens[k - 1] = (changed[at].tokens[k - 1] + 1) % model.vocab_size() as u32;
        let (_, r) = run_verify(model, prompt, k, steps, Feed::Replay(&changed))?;
        out.push(VerifyReport {
            k,
            accepted: log.iter().map(|s| s.accepted).collect(),
            comparisons,
            timings,
            detection_control: compare("last forward, one draft changed", &reference, &r),
        });
    }
    model.set_forward(&ForwardSelect::Legacy)?;
    Ok(out)
}

/// 2026-09-29: Why a verify report fails; empty for a pass. A reference whose steps never
/// accepted a draft, or never rejected one, did not exercise both commits and fails as thin.
pub(crate) fn verify_failures(reports: &[VerifyReport]) -> Vec<String> {
    let mut out = Vec::new();
    for r in reports {
        for c in &r.comparisons {
            if !c.prefill_equal || c.mismatched_steps > 0 {
                out.push(format!(
                    "verify K={}, {}: prefill equal {}, {} of {} steps differ (first at {:?})",
                    r.k, c.variant, c.prefill_equal, c.mismatched_steps, c.steps, c.first_mismatch
                ));
            }
        }
        if r.detection_control.mismatched_steps == 0 {
            out.push(format!(
                "verify K={}: the detection control saw no difference after changing a draft",
                r.k
            ));
        }
        let full = r.accepted.iter().filter(|&&a| a == r.k - 1).count();
        if full == 0 || full == r.accepted.len() {
            out.push(format!(
                "verify K={}: the reference accepted {:?}; it needs both full and partial accepts",
                r.k, r.accepted
            ));
        }
    }
    out
}
