// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: `met circuit diff --verify-batch`: the diff over the batched MTP verify
//! (`decode_verify_batched`), as the scheduler runs it (`verify_k4_batch_step.rs`). `n`
//! prompts are prefilled, then verified together step by step with write-on-accept, each
//! sequence's rows `[last, d1 .. d_{k-1}]`. The drafts come from each sequence's greedy
//! continuation with one corrupted on every odd step, so full and partial accepts both occur.
//! After each verify the GDN verdict is folded (`gdn_fold_accepted`), then each sequence's
//! rejected rows are rewound and its accepted prefix committed. Each step's `R = Σ k` logits
//! rows are compared byte for byte. 2026-09-30: With the drafter, each step also stashes the
//! accepted rows' hiddens and runs the batched propose over the batch, as the scheduler does
//! after a verdict, and compares its last position's logits, drafts and confidences; the
//! drafter's rows past the first are then trimmed, so the drafter's state is a function of the
//! replayed steps alone.
//!
//! Owner: server CLI.
//! Invariants:
//! - Every run after the reference replays the reference's tokens and accept counts, so all
//!   runs verify the same inputs and commit the same prefixes.
//! - Sequences are allocated in batch order, so their state slots are contiguous unless
//!   `fragment` holds a spare slot before sequence `n / 2`.
//! - The detection control changes one draft halfway; its difference must be seen.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use metrale_model_engine::traits::{ForwardSelect, Model, SequenceState, VerifyBatchedOpts};
use serde::Serialize;

use super::verify::{accepted, continuation, drafts};
use super::{Comparison, Run, Timing, argmax_bf16, compare, logits, timing};

/// 2026-09-30: One batched step's inputs and each sequence's accepted-draft count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct BatchStep {
    pub tokens: Vec<u32>,
    pub accepted: Vec<usize>,
}

/// 2026-09-30: How a run picks each step.
enum Feed<'a> {
    /// 2026-09-30: Each sequence's drafts from its greedy continuation, one corrupted on odd
    /// steps; the accept counts from the run's own argmaxes.
    Reference { greedy: &'a [Vec<u32>] },
    /// 2026-09-30: These steps exactly.
    Replay { steps: &'a [BatchStep] },
}

/// 2026-09-30: The widths a batch of `n` sequences verifies at: every `k` in `ks`, uniform,
/// then ragged ones that mix them deepest first, as `verify_batch_permutation` orders a batch.
pub(crate) fn batch_shapes(n: usize, ks: &[usize]) -> Vec<Vec<usize>> {
    let mut sorted: Vec<usize> = ks.to_vec();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    sorted.dedup();
    let mut out: Vec<Vec<usize>> = sorted.iter().map(|&k| vec![k; n]).collect();
    if sorted.len() > 1 && n >= sorted.len() {
        // 2026-09-30: `n` split over the widths as evenly as possible, deepest first.
        let per = n / sorted.len();
        let extra = n % sorted.len();
        out.push(
            sorted
                .iter()
                .enumerate()
                .flat_map(|(i, &k)| std::iter::repeat_n(k, per + usize::from(i < extra)))
                .collect(),
        );
    }
    out
}

/// 2026-09-30: The state slot sequence `i` of `n` takes: the pool hands out slots from 0, and
/// `fragment` holds one before sequence `n / 2`.
fn slot_of(i: usize, n: usize, fragment: bool) -> usize {
    i + usize::from(fragment && i >= n / 2)
}

/// 2026-09-30: `shape` with each sequence's width capped at its slot's verify capacity
/// (`ssm_reserve::verify_slot_drafts`, the K ladder's tiered pools), as the scheduler clamps a
/// step's drafts (`spec_capacity`).
pub(crate) fn capped(shape: &[usize], cap: impl Fn(usize) -> usize) -> Vec<usize> {
    shape
        .iter()
        .enumerate()
        .map(|(i, &k)| k.min(cap(i)))
        .collect()
}

/// 2026-09-30: `ks`' prefix sums: sequence `i`'s first row, then the total.
fn offsets(ks: &[usize]) -> Vec<usize> {
    let mut off = Vec::with_capacity(ks.len() + 1);
    let mut acc = 0;
    for &k in ks {
        off.push(acc);
        acc += k;
    }
    off.push(acc);
    off
}

/// 2026-09-30: The batched propose after a step (`verify_k4_batch_step.rs`): `nd` drafts for
/// every sequence from its stash slot, appending the last position's logits, the drafts and the
/// confidences to `out`; then each drafter is trimmed back as after a verify that accepted
/// nothing.
fn propose_all(
    model: &dyn Model,
    seqs: &mut [SequenceState],
    last: &[u32],
    nd: usize,
    out: &mut Vec<u8>,
) -> Result<()> {
    let n = seqs.len();
    let positions: Vec<usize> = seqs.iter().map(|q| q.seq_len).collect();
    let stash: Vec<usize> = (0..n).collect();
    let mut conf = Vec::new();
    let drafts = {
        let mut refs: Vec<&mut SequenceState> = seqs.iter_mut().collect();
        model.run_mtp_propose_batched(
            last,
            &positions,
            &stash,
            nd,
            &mut refs,
            0,
            Some(&mut conf),
        )?
    }
    .with_context(|| format!("the model declined a batched propose of {n} sequences"))?;
    ensure!(
        drafts.len() == n && drafts.iter().all(|d| d.len() == nd) && conf.len() == n,
        "the batched propose returned a short result"
    );
    let mut l = vec![0u8; n * model.vocab_size() * 2];
    model.copy_logits_to_host(model.logits_buffer_ptr(), &mut l)?;
    out.extend(l);
    out.extend(drafts.iter().flatten().flat_map(|t| t.to_le_bytes()));
    out.extend(conf.iter().flatten().flat_map(|c| c.to_le_bytes()));
    for seq in seqs.iter_mut() {
        model.trim_proposer_state(seq, 0, 0)?;
    }
    Ok(())
}

/// 2026-09-30: One run over `prompts` at the widths `ks`, `steps` steps; with `drafter`, the
/// batched propose of that many drafts after each step.
fn run_batch(
    model: &dyn Model,
    (prompts, fragment, drafter): (&[Vec<u32>], bool, Option<usize>),
    ks: &[usize],
    steps: usize,
    feed: Feed<'_>,
) -> Result<(Vec<BatchStep>, Run)> {
    let (n, vocab) = (prompts.len(), model.vocab_size());
    let off = offsets(ks);
    let rows = off[n];
    let mut seqs: Vec<SequenceState> = Vec::with_capacity(n);
    let mut spare: Option<SequenceState> = None;
    let result = (|| {
        let mut prefill = Vec::new();
        let mut last = Vec::with_capacity(n);
        for (i, p) in prompts.iter().enumerate() {
            if fragment && i == n / 2 {
                spare = Some(model.alloc_sequence()?);
            }
            seqs.push(model.alloc_sequence()?);
            ensure!(
                seqs[i].slot_idx == slot_of(i, n, fragment),
                "sequence {i} took slot {}; the widths were capped for slot {}",
                seqs[i].slot_idx,
                slot_of(i, n, fragment)
            );
            let l = logits(model, model.prefill(p, &mut seqs[i], 0)?)?;
            last.push(argmax_bf16(&l));
            prefill.extend(l);
        }
        let mut pos = vec![0usize; n];
        let (mut log, mut out) = (Vec::new(), Vec::new());
        let (mut enqueue_ms, mut step_ms) = (Vec::new(), Vec::new());
        for s in 0..steps {
            let tokens: Vec<u32> = match feed {
                Feed::Reference { greedy } => (0..n)
                    .flat_map(|i| {
                        std::iter::once(last[i]).chain(drafts(
                            &greedy[i],
                            pos[i],
                            s + i,
                            ks[i],
                            vocab as u32,
                        ))
                    })
                    .collect(),
                Feed::Replay { steps } => steps[s].tokens.clone(),
            };
            ensure!(
                model.can_batch_verify(ks),
                "the model refuses a batched verify of {ks:?}"
            );
            model.sync_secondary()?;
            let t0 = std::time::Instant::now();
            let outs = {
                let mut refs: Vec<&mut SequenceState> = seqs.iter_mut().collect();
                let opts = VerifyBatchedOpts {
                    write_on_accept: true,
                };
                model.decode_verify_batched(&tokens, ks, &mut refs, 0, opts)?
            };
            let t1 = std::time::Instant::now();
            let mut l = vec![0u8; rows * vocab * 2];
            model.copy_logits_to_host(model.logits_buffer_ptr(), &mut l)?;
            enqueue_ms.push((t1 - t0).as_secs_f64() * 1e3);
            step_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            let na: Vec<usize> = match feed {
                Feed::Replay { steps } => steps[s].accepted.clone(),
                Feed::Reference { .. } => (0..n)
                    .map(|i| accepted(&tokens[off[i]..off[i + 1]], &outs[off[i]..off[i + 1]]))
                    .collect(),
            };
            if drafter.is_some() {
                // 2026-09-30: Sequence i's accepted row, before the propose overwrites it.
                let rows: Vec<usize> = (0..n).map(|i| off[i] + na[i]).collect();
                model.stash_verify_hidden_rows(&rows, 0)?;
            }
            // 2026-09-30: As `step_verify_k4_batched` does: the GDN fold of the whole batch,
            // then each sequence's verdict (`k4_apply_verdict`): rewind the rejected rows and
            // commit rows `0..=na`.
            let slots: Vec<usize> = seqs.iter().map(|q| q.slot_idx).collect();
            let fold_rows: Vec<u32> = na.iter().map(|&a| (a + 1) as u32).collect();
            let k_max = ks.iter().copied().max().unwrap_or(2);
            model.gdn_fold_accepted(&slots, &fold_rows, k_max)?;
            for (i, seq) in seqs.iter_mut().enumerate() {
                let rejected = ks[i] - 1 - na[i];
                seq.seq_len -= rejected;
                let keep = seq.tokens.len() - rejected;
                seq.tokens.truncate(keep);
                model.commit_accepted_prefix(seq, na[i] + 1, ks[i])?;
                last[i] = match feed {
                    Feed::Replay { steps } => steps
                        .get(s + 1)
                        .map_or(outs[off[i] + na[i]], |nx| nx.tokens[off[i]]),
                    Feed::Reference { .. } => outs[off[i] + na[i]],
                };
                pos[i] += na[i] + 1;
            }
            if let Some(nd) = drafter {
                propose_all(model, &mut seqs, &last, nd, &mut l)?;
            }
            log.push(BatchStep {
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
    let mid = if spare.is_some() {
        (n / 2).min(seqs.len())
    } else {
        0
    };
    let (low, high) = seqs.split_at_mut(mid);
    for s in high
        .iter_mut()
        .rev()
        .chain(spare.as_mut())
        .chain(low.iter_mut().rev())
    {
        model.free_sequence(s)?;
    }
    result
}

/// 2026-09-30: One batch shape's results.
#[derive(Debug, Serialize)]
pub(crate) struct ShapeReport {
    pub ks: Vec<usize>,
    /// 2026-09-30: The reference's accept counts, per step, per sequence.
    pub accepted: Vec<Vec<usize>>,
    pub comparisons: Vec<Comparison>,
    pub timings: Vec<Timing>,
    pub detection_control: Comparison,
}

/// 2026-09-30: Run the reference and every forward in `forwards` for every width in `widths`
/// at every shape of [`batch_shapes`].
pub(crate) fn diff_verify_batch(
    model: &dyn Model,
    (widths, ks, fragment, mtp): (&[usize], &[usize], bool, bool),
    num_drafts: usize,
    steps: usize,
    forwards: &[(&'static str, ForwardSelect)],
) -> Result<Vec<ShapeReport>> {
    ensure!(!forwards.is_empty(), "no forward to compare");
    let mut out = Vec::new();
    for &n in widths {
        ensure!(n >= 2, "a batched verify is at least 2 sequences");
        let prompts = super::batch::batch_prompts(n, model.vocab_size());
        model.set_forward(&ForwardSelect::Legacy)?;
        let k_top = ks.iter().copied().max().unwrap_or(2);
        let greedy: Vec<Vec<u32>> = prompts
            .iter()
            .map(|p| continuation(model, p, steps * k_top + 1))
            .collect::<Result<_>>()?;
        let cap = |i: usize| {
            metrale_model_layers::ssm_reserve::verify_slot_drafts(
                slot_of(i, n, fragment),
                num_drafts,
            ) + 1
        };
        let mut shapes: Vec<Vec<usize>> =
            batch_shapes(n, ks).iter().map(|s| capped(s, cap)).collect();
        shapes.dedup();
        for shape in shapes {
            model.set_forward(&ForwardSelect::Legacy)?;
            let rows = (prompts.as_slice(), fragment, mtp.then_some(num_drafts));
            if mtp {
                // 2026-09-30: The first propose of a process differs from every later one
                // (legacy against legacy), so one discarded run goes first.
                run_batch(model, rows, &shape, 1, Feed::Reference { greedy: &greedy })?;
            }
            let reference_feed = Feed::Reference { greedy: &greedy };
            let (log, reference) = run_batch(model, rows, &shape, steps, reference_feed)?;
            let (mut comparisons, mut timings) = (Vec::new(), vec![timing("legacy", &reference)]);
            for (name, sel) in forwards {
                model.set_forward(sel)?;
                let (_, r) = run_batch(model, rows, &shape, steps, Feed::Replay { steps: &log })?;
                let c = compare(name, &reference, &r);
                tracing::info!("circuit diff: verify-batch {shape:?} {name}: {c:?}");
                timings.push(timing(name, &r));
                comparisons.push(c);
            }
            // 2026-09-30: The control runs under the last forward, still selected.
            let mut changed = log.clone();
            let at = steps / 2;
            let row = offsets(&shape)[n / 2] + shape[n / 2] - 1;
            changed[at].tokens[row] = (changed[at].tokens[row] + 1) % model.vocab_size() as u32;
            let (_, r) = run_batch(model, rows, &shape, steps, Feed::Replay { steps: &changed })?;
            out.push(ShapeReport {
                accepted: log.iter().map(|s| s.accepted.clone()).collect(),
                ks: shape,
                comparisons,
                timings,
                detection_control: compare("last forward, one draft changed", &reference, &r),
            });
        }
    }
    model.set_forward(&ForwardSelect::Legacy)?;
    Ok(out)
}

/// 2026-09-30: Why a batched-verify report fails; empty for a pass. A shape whose reference
/// never accepted every draft of some sequence, or never rejected one, did not exercise both
/// commits and fails as thin.
pub(crate) fn verify_batch_failures(reports: &[ShapeReport]) -> Vec<String> {
    let mut out = Vec::new();
    for r in reports {
        for c in &r.comparisons {
            if !c.prefill_equal || c.mismatched_steps > 0 {
                out.push(format!(
                    "verify-batch {:?}, {}: prefill equal {}, {} of {} steps differ (first at {:?})",
                    r.ks, c.variant, c.prefill_equal, c.mismatched_steps, c.steps, c.first_mismatch
                ));
            }
        }
        if r.detection_control.mismatched_steps == 0 {
            out.push(format!(
                "verify-batch {:?}: the detection control saw no difference after changing a draft",
                r.ks
            ));
        }
        let pairs = r.accepted.iter().flat_map(|a| a.iter().zip(&r.ks));
        let full = pairs.clone().filter(|&(&a, &k)| a == k - 1).count();
        let partial = pairs.filter(|&(&a, &k)| a < k - 1).count();
        if full == 0 || partial == 0 {
            out.push(format!(
                "verify-batch {:?}: {full} full and {partial} partial accepts; it needs both",
                r.ks
            ));
        }
    }
    out
}

#[derive(Debug, Serialize)]
struct VerifyBatchReport {
    graphs: &'static str,
    fragment_slots: bool,
    drafter: bool,
    steps: usize,
    variants: Vec<super::Variant>,
    shapes: Vec<ShapeReport>,
    verdict: &'static str,
    reasons: Vec<String>,
}

/// 2026-09-30: The `--verify-batch` diff: every width and shape, then the verdict.
pub(crate) fn verify_batch_report(
    model: &dyn Model,
    shape: (&[usize], &[usize], bool, bool),
    num_drafts: usize,
    steps: usize,
    forwards: &[(&'static str, ForwardSelect)],
    out: &Path,
) -> Result<()> {
    let variants = super::disclosed(model, forwards)?;
    let shapes = diff_verify_batch(model, shape, num_drafts, steps, forwards)?;
    let reasons = verify_batch_failures(&shapes);
    let report = VerifyBatchReport {
        // 2026-09-30: The batched verify's graphs are off when `METRALE_NO_MTP_VERIFY_GRAPHS` is
        // present (model-engine `verify_e2::verify_graphs_enabled`).
        graphs: if std::env::var_os("METRALE_NO_MTP_VERIFY_GRAPHS").is_some() {
            "eager"
        } else {
            "graphed"
        },
        fragment_slots: shape.2,
        drafter: shape.3,
        steps,
        variants,
        shapes,
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

#[cfg(test)]
mod tests {
    use super::*;

    // 2026-09-30: Uniform shapes for every width, then one ragged shape, deepest first.
    #[test]
    fn shapes_cover_every_width_and_one_ragged_mix() {
        assert_eq!(batch_shapes(2, &[4]), vec![vec![4, 4]]);
        assert_eq!(
            batch_shapes(7, &[2, 4, 3]),
            vec![
                vec![4; 7],
                vec![3; 7],
                vec![2; 7],
                vec![4, 4, 4, 3, 3, 2, 2]
            ]
        );
        assert_eq!(
            batch_shapes(2, &[2, 3, 4]).len(),
            3,
            "too few sequences to mix"
        );
    }
}
