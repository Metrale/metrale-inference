// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: `met circuit diff --prefill`: the prefill parity instrument (LIFECYCLE-DESIGN.md
//! sections 7.1 and 15.4). For each prompt length it prefills one sequence single-pass and, with
//! `--prefill-chunk`, in chunks, under legacy (twice: the reference and the repeat that shows the
//! path is repeatable at all) and under every circuit forward, and compares the last-position
//! logits byte for byte. The reference run of each path records its launch trace (kernel, grid,
//! block per op), which is the map the circuit's prefill rules reproduce. A detection control
//! changes one prompt token and must change the logits.
//!
//! Owner: server CLI.
//! Invariants:
//! - The verdict is PASS only when every run of every path matches its reference and the
//!   control differs; the exit status is non-zero otherwise.
//! - Every run prefills a fresh sequence, freed before the next.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use metrale_model_engine::traits::{ForwardSelect, Model};
use serde::Serialize;

/// 2026-10-03: A prompt of exactly `len` tokens, ids from the diff's 64-bit LCG over
/// `[64, vocab - 64)`, seeded by `len` so each length is its own prompt.
pub(crate) fn prompt_of_len(len: usize, vocab: usize) -> Vec<u32> {
    let mut x: u64 = 0x5eed_1003 ^ (len as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let span = (vocab.saturating_sub(128)).max(1) as u64;
    (0..len)
        .map(|_| {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            64 + ((x >> 33) % span) as u32
        })
        .collect()
}

/// 2026-10-03: How a run prefills.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) enum PrefillPath {
    /// 2026-10-03: `Model::prefill`, the whole prompt in one pass.
    Single,
    /// 2026-10-03: `Model::prefill_chunk` over chunks of this many tokens.
    Chunked(usize),
}

/// 2026-10-03: One op of a launch trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct TracedOp {
    pub op: String,
    pub grid: [u32; 3],
    pub block: [u32; 3],
}

/// 2026-10-03: Prefill `prompt` on a fresh sequence along `path`; the last-position logits, and
/// the launch trace when `trace`.
fn prefill_once(
    model: &dyn Model,
    prompt: &[u32],
    path: PrefillPath,
    trace: bool,
) -> Result<(Vec<u8>, Vec<TracedOp>)> {
    let mut seq = model.alloc_sequence()?;
    let result = (|| {
        if trace {
            metrale_telemetry::launch_trace::begin();
        }
        let ptr = match path {
            PrefillPath::Single => model.prefill(prompt, &mut seq, 0)?,
            PrefillPath::Chunked(c) => {
                ensure!(c > 0, "a chunk of 0 tokens");
                let mut ptr = metrale_gpu_runtime::gpu::DevicePtr::NULL;
                let mut off = 0;
                while off < prompt.len() {
                    let len = c.min(prompt.len() - off);
                    let last = off + len == prompt.len();
                    ptr = model.prefill_chunk(prompt, &mut seq, off, len, last, 0)?;
                    off += len;
                }
                ptr
            }
        };
        let ops = if trace {
            metrale_telemetry::launch_trace::end_and_take()
                .iter()
                .map(|e| TracedOp {
                    op: metrale_telemetry::launch_trace::op_name(e),
                    grid: e.grid,
                    block: e.block,
                })
                .collect()
        } else {
            Vec::new()
        };
        Ok((super::logits(model, ptr)?, ops))
    })();
    if trace {
        metrale_telemetry::launch_trace::end_and_take();
    }
    model.free_sequence(&mut seq)?;
    result
}

/// 2026-10-03: One run's comparison with its path's reference.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct PrefillComparison {
    pub variant: String,
    pub equal: bool,
    pub first_diff: Option<usize>,
}

/// 2026-10-03: One prompt length on one path.
#[derive(Debug, Serialize)]
struct PathReport {
    path: PrefillPath,
    comparisons: Vec<PrefillComparison>,
    /// 2026-10-03: The reference run's ops, in issue order.
    trace: Vec<TracedOp>,
}

#[derive(Debug, Serialize)]
struct LenReport {
    tokens: usize,
    paths: Vec<PathReport>,
}

#[derive(Debug, Serialize)]
struct Report {
    lengths: Vec<LenReport>,
    detection_control: PrefillComparison,
    verdict: &'static str,
    reasons: Vec<String>,
}

fn compare(variant: &str, reference: &[u8], run: &[u8]) -> PrefillComparison {
    PrefillComparison {
        variant: variant.to_string(),
        equal: reference == run,
        first_diff: reference
            .iter()
            .zip(run)
            .position(|(a, b)| a != b)
            .or((reference.len() != run.len()).then_some(reference.len().min(run.len()))),
    }
}

/// 2026-10-03: Why a report fails; empty for a pass.
pub(crate) fn prefill_failures(
    per_len: &[(usize, PrefillPath, Vec<PrefillComparison>)],
    control: &PrefillComparison,
) -> Vec<String> {
    let mut out: Vec<String> = per_len
        .iter()
        .flat_map(|(t, path, cs)| {
            cs.iter()
                .filter(|c| !c.equal)
                .map(move |c| format!("{t} tokens, {path:?}, {}: logits differ at byte {:?}", c.variant, c.first_diff))
        })
        .collect();
    if control.equal {
        out.push("the detection control saw no difference after changing a prompt token".into());
    }
    out
}

/// 2026-10-03: The `--prefill` diff.
pub(super) fn prefill_report(
    model: &dyn Model,
    (lens, chunk): (&[usize], Option<usize>),
    forwards: &[(&'static str, ForwardSelect)],
    out: &Path,
) -> Result<()> {
    ensure!(!lens.is_empty() && lens.iter().all(|&t| t > 0), "--prefill takes lengths >= 1");
    let paths: Vec<PrefillPath> = std::iter::once(PrefillPath::Single)
        .chain(chunk.map(PrefillPath::Chunked))
        .collect();
    let mut lengths = Vec::new();
    let mut flat = Vec::new();
    for &t in lens {
        let prompt = prompt_of_len(t, model.vocab_size());
        let mut reports = Vec::new();
        for &path in &paths {
            model.set_forward(&ForwardSelect::Legacy)?;
            let (reference, trace) = prefill_once(model, &prompt, path, true)?;
            let mut cs = Vec::new();
            for (name, sel) in forwards {
                model.set_forward(sel)?;
                let (run, _) = prefill_once(model, &prompt, path, false)?;
                let c = compare(name, &reference, &run);
                tracing::info!("circuit diff --prefill: {t} tokens {path:?} {name}: {c:?}");
                cs.push(c);
            }
            flat.push((t, path, cs.clone()));
            reports.push(PathReport {
                path,
                comparisons: cs,
                trace,
            });
        }
        lengths.push(LenReport {
            tokens: t,
            paths: reports,
        });
    }
    model.set_forward(&ForwardSelect::Legacy)?;
    let t = *lens.iter().max().context("no length")?;
    let prompt = prompt_of_len(t, model.vocab_size());
    let mut changed = prompt.clone();
    let at = t / 2;
    changed[at] = (changed[at] + 1) % model.vocab_size() as u32;
    let (a, _) = prefill_once(model, &prompt, PrefillPath::Single, false)?;
    let (b, _) = prefill_once(model, &changed, PrefillPath::Single, false)?;
    let control = compare("legacy, prompt token changed", &a, &b);
    let reasons = prefill_failures(&flat, &control);
    let report = Report {
        lengths,
        detection_control: control,
        verdict: if reasons.is_empty() { "PASS" } else { "FAIL" },
        reasons: reasons.clone(),
    };
    std::fs::write(out, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("writing {}", out.display()))?;
    if !reasons.is_empty() {
        bail!("circuit diff --prefill FAILED:\n  {}", reasons.join("\n  "));
    }
    Ok(())
}
