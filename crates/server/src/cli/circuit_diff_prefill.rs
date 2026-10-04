// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: `met circuit diff --prefill`: the prefill parity instrument (LIFECYCLE-DESIGN.md
//! sections 7.1 and 15.4). For each prompt length it prefills one sequence single-pass and, with
//! `--prefill-chunk`, in chunks, under legacy (twice: the reference and the repeat that shows the
//! path is repeatable at all) and under every circuit forward, and compares the last-position
//! logits byte for byte. Every run records its launch trace (kernel, grid, block per op); the
//! reference's is reported, the map the circuit's prefill rules reproduce, and a legacy run whose
//! ops differ from the reference's took another path (a prefix-cache restore, a split moved) and
//! fails as such, never as a numeric difference. Each run gets its own session hash, so a
//! snapshot one run saved is not restored by the next. A detection control changes one prompt
//! token and must change the logits. Every run's state is digested after the pass
//! (`ModelCircuit::state_digest`: each recurrent layer's h and conv, each attention layer's K and
//! V blocks); a run whose state differs from the reference's fails on the first differing entry,
//! even when its logits agree.
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

/// 2026-10-03: What one prefill run leaves: the last-position logits, the launch trace and the
/// state digest.
pub(crate) struct PrefillRun {
    pub logits: Vec<u8>,
    pub ops: Vec<TracedOp>,
    pub state: Vec<(String, u64)>,
}

/// 2026-10-03: Prefill `prompt` on a fresh sequence of session `session` along `path`.
pub(super) fn prefill_once(
    model: &dyn Model,
    prompt: &[u32],
    path: PrefillPath,
    session: u64,
) -> Result<PrefillRun> {
    let mut seq = model.alloc_sequence()?;
    // 2026-10-03: Its own prefix-cache namespace (`adapter_id`, one radix root per id) as well as
    // its own session: no run matches another's KV blocks or snapshots, so every run of a strict
    // leg takes the reference's cold path. The cached leg (`cached.rs`) shares a namespace on
    // purpose.
    seq.session_hash = session;
    seq.adapter_id = session;
    let result = (|| {
        metrale_telemetry::launch_trace::begin();
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
        let ops = metrale_telemetry::launch_trace::end_and_take()
            .iter()
            .map(|e| TracedOp {
                op: metrale_telemetry::launch_trace::op_name(e),
                grid: e.grid,
                block: e.block,
            })
            .collect();
        let logits = super::logits(model, ptr)?;
        Ok(PrefillRun {
            logits,
            ops,
            state: model.state_digest(&seq)?,
        })
    })();
    metrale_telemetry::launch_trace::end_and_take();
    model.free_sequence(&mut seq)?;
    result
}

/// 2026-10-03: One run's comparison with its path's reference.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct PrefillComparison {
    pub variant: String,
    pub equal: bool,
    pub first_diff: Option<usize>,
    /// 2026-10-03: For a legacy run, the first op where its trace leaves the reference's (a
    /// different path); `None` when the traces agree or the run is not legacy.
    pub path_diff: Option<usize>,
    /// 2026-10-03: The first state entry that differs from the reference's; `None` when the
    /// states agree.
    pub state_diff: Option<String>,
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
    /// 2026-10-03: The cached-prefix leg (`cached.rs`).
    cached: Vec<cached::CachedReport>,
    detection_control: PrefillComparison,
    verdict: &'static str,
    reasons: Vec<String>,
}

pub(super) fn compare(variant: &str, reference: &[u8], run: &[u8]) -> PrefillComparison {
    PrefillComparison {
        path_diff: None,
        state_diff: None,
        variant: variant.to_string(),
        equal: reference == run,
        first_diff: reference
            .iter()
            .zip(run)
            .position(|(a, b)| a != b)
            .or((reference.len() != run.len()).then_some(reference.len().min(run.len()))),
    }
}

/// 2026-10-03: The first op where `run` leaves `reference`, or where one ends first.
pub(crate) fn first_op_diff(reference: &[TracedOp], run: &[TracedOp]) -> Option<usize> {
    reference
        .iter()
        .zip(run)
        .position(|(a, b)| a != b)
        .or((reference.len() != run.len()).then_some(reference.len().min(run.len())))
}

/// 2026-10-03: The label of the first state entry where `run` differs from `reference` (or a
/// missing entry).
pub(crate) fn first_state_diff(
    reference: &[(String, u64)],
    run: &[(String, u64)],
) -> Option<String> {
    reference
        .iter()
        .zip(run)
        .find(|(a, b)| a != b)
        .map(|(a, _)| a.0.clone())
        .or_else(|| (reference.len() != run.len()).then(|| "entry count".to_string()))
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
                .filter(|c| !c.equal || c.path_diff.is_some() || c.state_diff.is_some())
                .map(move |c| match (c.path_diff, &c.state_diff) {
                    (Some(op), _) => format!(
                        "{t} tokens, {path:?}, {}: took another path (ops differ from op {op})",
                        c.variant
                    ),
                    (None, Some(entry)) if c.equal => format!(
                        "{t} tokens, {path:?}, {}: logits equal, state differs at {entry}",
                        c.variant
                    ),
                    _ => format!(
                        "{t} tokens, {path:?}, {}: logits differ at byte {:?}, state at {:?}",
                        c.variant, c.first_diff, c.state_diff
                    ),
                })
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
    (lens, chunks): (&[usize], &[usize]),
    forwards: &[(&'static str, ForwardSelect)],
    out: &Path,
) -> Result<()> {
    ensure!(
        !lens.is_empty() && lens.iter().all(|&t| t > 0),
        "--prefill takes lengths >= 1"
    );
    let paths: Vec<PrefillPath> = std::iter::once(PrefillPath::Single)
        .chain(chunks.iter().map(|&c| PrefillPath::Chunked(c)))
        .collect();
    let mut lengths = Vec::new();
    let mut flat = Vec::new();
    let mut session = 0x00c1_4c00_0000_0000u64;
    let mut next_session = || {
        session += 1;
        session
    };
    for &t in lens {
        let prompt = prompt_of_len(t, model.vocab_size());
        let mut reports = Vec::new();
        for &path in &paths {
            model.set_forward(&ForwardSelect::Legacy)?;
            let reference = prefill_once(model, &prompt, path, next_session())?;
            let mut cs = Vec::new();
            for (name, sel) in forwards {
                model.set_forward(sel)?;
                let run = prefill_once(model, &prompt, path, next_session())?;
                let mut c = compare(name, &reference.logits, &run.logits);
                c.state_diff = first_state_diff(&reference.state, &run.state);
                if matches!(sel, ForwardSelect::Legacy) {
                    c.path_diff = first_op_diff(&reference.ops, &run.ops);
                }
                tracing::info!("circuit diff --prefill: {t} tokens {path:?} {name}: {c:?}");
                cs.push(c);
            }
            flat.push((t, path, cs.clone()));
            reports.push(PathReport {
                path,
                comparisons: cs,
                trace: reference.ops,
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
    let a = prefill_once(model, &prompt, PrefillPath::Single, next_session())?;
    let b = prefill_once(model, &changed, PrefillPath::Single, next_session())?;
    let mut control = compare("legacy, prompt token changed", &a.logits, &b.logits);
    control.state_diff = first_state_diff(&a.state, &b.state);
    let (cached_reports, cached_flat) =
        cached::cached_report(model, lens, &paths, forwards, &mut next_session)?;
    flat.extend(cached_flat);
    let reasons = prefill_failures(&flat, &control);
    let report = Report {
        cached: cached_reports,
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

#[path = "circuit_diff_prefill_cached.rs"]
mod cached;
