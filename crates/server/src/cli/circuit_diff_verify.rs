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

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
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
    /// 2026-09-29: Drafts from the model's MTP draft head, as the scheduler's speculative loop
    /// runs it (`mtp_step`): a bootstrap decode, then after every verify the hidden save, the
    /// proposer trim and a propose.
    Drafter,
    /// 2026-09-29: These steps exactly (after this bootstrap token, with the draft head run
    /// and compared, when `mtp`).
    Replay {
        steps: &'a [Step],
        bootstrap: u32,
        mtp: bool,
    },
}

impl Feed<'_> {
    fn mtp(&self) -> bool {
        matches!(self, Feed::Drafter | Feed::Replay { mtp: true, .. })
    }
}

fn bytes_of(tokens: &[u32]) -> impl Iterator<Item = u8> + '_ {
    tokens.iter().flat_map(|t| t.to_le_bytes())
}

/// 2026-09-29: Propose `k - 1` drafts after `last`; appends the last draft's logits and the
/// drafts to `out`.
fn propose(
    model: &dyn Model,
    seq: &mut SequenceState,
    last: u32,
    k: usize,
    out: &mut Vec<u8>,
) -> Result<Vec<u32>> {
    let drafts = model.run_mtp_propose_multi(last, seq.seq_len, k - 1, seq, 0, None)?;
    ensure!(
        drafts.len() == k - 1,
        "the drafter proposed {} of {}",
        drafts.len(),
        k - 1
    );
    out.extend(logits(model, model.logits_buffer_ptr())?);
    out.extend(bytes_of(&drafts));
    Ok(drafts)
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
pub(crate) fn continuation(model: &dyn Model, prompt: &[u32], n: usize) -> Result<Vec<u32>> {
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

/// 2026-09-29: One run: its steps, the bytes compared (the prefill, and with the drafter the
/// bootstrap decode and first propose, in `Run::prefill`; each step's `K` verify rows, and with
/// the drafter the propose after it), and the bootstrap token.
fn run_verify(
    model: &dyn Model,
    prompt: &[u32],
    k: usize,
    steps: usize,
    feed: Feed<'_>,
) -> Result<(Vec<Step>, Run, u32)> {
    let vocab = model.vocab_size();
    let mut seq = model.alloc_sequence()?;
    let result = (|| {
        let mut prefill = logits(model, model.prefill(prompt, &mut seq, 0)?)?;
        let mut last = argmax_bf16(&prefill);
        let mut next = Vec::new();
        if feed.mtp() {
            let l = logits(model, model.decode(last, &mut seq, 0)?)?;
            last = match feed {
                Feed::Replay { bootstrap, .. } => bootstrap,
                _ => argmax_bf16(&l),
            };
            prefill.extend(l);
            model.save_hidden_for_mtp(0, 0)?;
            next = propose(model, &mut seq, last, k, &mut prefill)?;
        }
        let bootstrap = last;
        let (mut p, mut log, mut out) = (0usize, Vec::new(), Vec::new());
        let (mut enqueue_ms, mut step_ms) = (Vec::new(), Vec::new());
        for s in 0..steps {
            let tokens: Vec<u32> = match feed {
                Feed::Reference { greedy } => std::iter::once(last)
                    .chain(drafts(greedy, p, s, k, vocab as u32))
                    .collect(),
                Feed::Drafter => std::iter::once(last).chain(next.iter().copied()).collect(),
                Feed::Replay { steps, .. } => steps[s].tokens.clone(),
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
                Feed::Replay { steps, .. } => steps[s].accepted,
                _ => accepted(&tokens, &outs),
            };
            // 2026-09-29: As the scheduler does (`verify_k4_verdict.rs`): rewind the rejected
            // rows and trim the proposer, commit the rest, save the hidden the next propose
            // reads, and trim the proposer after a full accept.
            let rejected = k - 1 - na;
            seq.seq_len -= rejected;
            let keep = seq.tokens.len() - rejected;
            seq.tokens.truncate(keep);
            if feed.mtp() && rejected > 0 {
                model.trim_proposer_state(&mut seq, na, 0)?;
            }
            model.commit_accepted_prefix(&mut seq, na + 1, k)?;
            last = match feed {
                Feed::Replay { steps, .. } => steps.get(s + 1).map_or(outs[na], |n| n.tokens[0]),
                _ => outs[na],
            };
            p += na + 1;
            if feed.mtp() {
                model.save_hidden_for_mtp(na, 0)?;
                if rejected == 0 {
                    model.trim_proposer_state(&mut seq, na, 0)?;
                }
                next = propose(model, &mut seq, last, k, &mut l)?;
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
            bootstrap,
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

/// 2026-09-29: Run the reference and every forward in `forwards` at each `K`; with `mtp`, the
/// drafts come from the model's draft head, which every run then also runs and compares.
pub(crate) fn diff_verify(
    model: &dyn Model,
    prompt: &[u32],
    ks: &[usize],
    steps: usize,
    mtp: bool,
    forwards: &[(&'static str, ForwardSelect)],
) -> Result<Vec<VerifyReport>> {
    ensure!(!forwards.is_empty(), "no forward to compare");
    let mut out = Vec::with_capacity(ks.len());
    for &k in ks {
        model.set_forward(&ForwardSelect::Legacy)?;
        let greedy;
        let feed = if mtp {
            Feed::Drafter
        } else {
            greedy = continuation(model, prompt, steps * k + 1)?;
            Feed::Reference { greedy: &greedy }
        };
        // 2026-09-29: The first run of a process differs in its first propose from every later
        // run (legacy against legacy), so with the drafter one discarded run goes first.
        if mtp && out.is_empty() {
            run_verify(model, prompt, k, steps, Feed::Drafter)?;
        }
        let (log, reference, bootstrap) = run_verify(model, prompt, k, steps, feed)?;
        let replay = |steps| Feed::Replay {
            steps,
            bootstrap,
            mtp,
        };
        let (mut comparisons, mut timings) = (Vec::new(), vec![timing("legacy", &reference)]);
        for (name, sel) in forwards {
            model.set_forward(sel)?;
            let (_, r, _) = run_verify(model, prompt, k, steps, replay(&log))?;
            let c = compare(name, &reference, &r);
            tracing::info!("circuit diff: verify K={k} {name}: {c:?}");
            timings.push(timing(name, &r));
            comparisons.push(c);
        }
        // 2026-09-29: The control runs under the last forward, still selected.
        let mut changed = log.clone();
        let at = steps / 2;
        changed[at].tokens[k - 1] = (changed[at].tokens[k - 1] + 1) % model.vocab_size() as u32;
        let (_, r, _) = run_verify(model, prompt, k, steps, replay(&changed))?;
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

#[derive(Debug, Serialize)]
struct VerifyDiffReport {
    graphs: &'static str,
    mtp: bool,
    steps: usize,
    variants: Vec<super::Variant>,
    verify: Vec<VerifyReport>,
    verdict: &'static str,
    reasons: Vec<String>,
}

/// 2026-09-29: The `--verify` diff: every `K`, then the verdict.
pub(crate) fn verify_report(
    model: &dyn Model,
    prompt: &[u32],
    (ks, mtp): (&[usize], bool),
    steps: usize,
    forwards: &[(&'static str, ForwardSelect)],
    out: &Path,
) -> Result<()> {
    let variants = super::disclosed(model, forwards)?;
    let verify = diff_verify(model, prompt, ks, steps, mtp, forwards)?;
    let reasons = verify_failures(&verify);
    let report = VerifyDiffReport {
        graphs: if std::env::var("METRALE_DEBUG_NO_GRAPH").as_deref() == Ok("1") {
            "eager"
        } else {
            "graphed"
        },
        mtp,
        steps,
        variants,
        verify,
        verdict: if reasons.is_empty() { "PASS" } else { "FAIL" },
        reasons: reasons.clone(),
    };
    std::fs::write(out, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("writing {}", out.display()))?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !reasons.is_empty() {
        bail!("circuit diff FAILED:\n  {}", reasons.join("\n  "));
    }
    Ok(())
}
