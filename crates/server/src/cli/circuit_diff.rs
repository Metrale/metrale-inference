// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `met circuit diff`: build the model as `met serve` would, then decode the same
//! prompts under the legacy forward and the circuit forward and compare every step's logits
//! byte for byte.
//!
//! Each prompt runs, in order: legacy (greedy, which fixes the token stream), legacy again (the
//! control that the run is repeatable at all), the circuit with reference rules only, the circuit
//! with every rule, and, on the first prompt, the full circuit with one token of the stream
//! replaced halfway (the control that a difference is seen). Every run after the first is fed the
//! first run's tokens, so all runs decode the same inputs.
//!
//! Owner: server CLI.
//! Invariants:
//! - The verdict is PASS only when both controls behave and every circuit run matches legacy
//!   at every step; the exit status is non-zero otherwise.
//! - Nothing here decides a kernel: the forwards are the model's own (`ModelCircuit`).

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use metrale_model_engine::traits::{ForwardSelect, Model};
use metrale_model_layers::circuit_exec::{Fusions, sources};
use serde::Serialize;

use super::CircuitDiffArgs;
use crate::main_modules::serve_load::engine::load_engine;
use crate::main_modules::serve_phases::circuit_target;

/// 2026-09-28: One decode run: the prefill's logits, then each step's.
struct Run {
    prefill: Vec<u8>,
    steps: Vec<Vec<u8>>,
    tokens: Vec<u32>,
    /// 2026-09-28: Per step, the wall time of `decode` (the host enqueue: decode does not
    /// synchronize) and of `decode` plus the logits copy (which does), in ms.
    enqueue_ms: Vec<f64>,
    step_ms: Vec<f64>,
}

/// 2026-09-28: How a run picks each step's input token.
enum Feed<'a> {
    /// 2026-09-28: The argmax of the previous logits.
    Greedy,
    /// 2026-09-28: These tokens, `steps + 1` of them.
    Forced(&'a [u32]),
}

/// 2026-09-28: The index of the largest BF16 value; the lowest index wins a tie, NaN never
/// wins.
pub(crate) fn argmax_bf16(bytes: &[u8]) -> u32 {
    let mut best = (0u32, f32::NEG_INFINITY);
    for (i, c) in bytes.chunks_exact(2).enumerate() {
        let v = f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16);
        if v > best.1 {
            best = (i as u32, v);
        }
    }
    best.0
}

/// 2026-09-28: The prompts: token ids from a 64-bit LCG over `[64, vocab - 64)`, prompt `i`
/// `23 + 41 * i` tokens long.
pub(crate) fn prompts(n: usize, vocab: usize) -> Vec<Vec<u32>> {
    let mut x: u64 = 0x5eed_0928;
    let span = (vocab.saturating_sub(128)).max(1) as u64;
    (0..n)
        .map(|i| {
            (0..23 + 41 * i)
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

fn logits(model: &dyn Model, ptr: metrale_gpu_runtime::gpu::DevicePtr) -> Result<Vec<u8>> {
    let width = if model.logits_ptr_is_fp32(ptr) { 4 } else { 2 };
    ensure!(width == 2, "the diff compares BF16 logits");
    let mut out = vec![0u8; model.vocab_size() * width];
    model.copy_logits_to_host(ptr, &mut out)?;
    Ok(out)
}

fn run(model: &dyn Model, prompt: &[u32], steps: usize, feed: Feed<'_>) -> Result<Run> {
    let mut seq = model.alloc_sequence()?;
    let result = (|| {
        let prefill = logits(model, model.prefill(prompt, &mut seq, 0)?)?;
        let mut tokens = vec![match feed {
            Feed::Greedy => argmax_bf16(&prefill),
            Feed::Forced(t) => t[0],
        }];
        let mut out = Vec::with_capacity(steps);
        let (mut enqueue_ms, mut step_ms) = (Vec::new(), Vec::new());
        for step in 0..steps {
            let t0 = std::time::Instant::now();
            let ptr = model.decode(tokens[step], &mut seq, 0)?;
            let t1 = std::time::Instant::now();
            let l = logits(model, ptr)?;
            enqueue_ms.push((t1 - t0).as_secs_f64() * 1e3);
            step_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            tokens.push(match feed {
                Feed::Greedy => argmax_bf16(&l),
                Feed::Forced(t) => t[step + 1],
            });
            out.push(l);
        }
        Ok(Run {
            prefill,
            steps: out,
            tokens,
            enqueue_ms,
            step_ms,
        })
    })();
    model.free_sequence(&mut seq)?;
    result
}

/// 2026-09-28: How one run compares with the reference run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct Comparison {
    pub variant: String,
    pub prefill_equal: bool,
    pub steps: usize,
    pub mismatched_steps: usize,
    pub first_mismatch: Option<usize>,
    pub max_mismatched_bytes: usize,
    /// 2026-09-29: The first differing byte of the prefill bytes, which locates the part that
    /// differs where they concatenate several (the verify diff's bootstrap).
    pub prefill_first_diff: Option<usize>,
}

/// 2026-09-28: Compare `run` with `reference`, step by step.
fn compare(variant: &str, reference: &Run, run: &Run) -> Comparison {
    let diff = |a: &[u8], b: &[u8]| {
        a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
    };
    let per_step: Vec<usize> = reference
        .steps
        .iter()
        .zip(&run.steps)
        .map(|(a, b)| diff(a, b))
        .collect();
    Comparison {
        variant: variant.to_string(),
        prefill_equal: reference.prefill == run.prefill,
        steps: per_step.len(),
        mismatched_steps: per_step.iter().filter(|&&d| d > 0).count(),
        first_mismatch: per_step.iter().position(|&d| d > 0),
        max_mismatched_bytes: per_step.iter().copied().max().unwrap_or(0),
        prefill_first_diff: reference
            .prefill
            .iter()
            .zip(&run.prefill)
            .position(|(a, b)| a != b),
    }
}

/// 2026-09-28: The forward a variant ran, as the model disclosed it.
#[derive(Debug, Clone, Serialize)]
struct Variant {
    name: &'static str,
    plan_digest: Option<String>,
    launches_per_step: Option<usize>,
}

/// 2026-09-28: A run's per-step times. The first step is excluded: under graphs it captures.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct Timing {
    pub variant: String,
    pub enqueue_ms_median: f64,
    pub step_ms_median: f64,
}

/// 2026-09-28: The median of `xs[1..]`.
pub(crate) fn median_after_first(xs: &[f64]) -> f64 {
    let mut v: Vec<f64> = xs.iter().skip(1).copied().collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn timing(variant: &str, r: &Run) -> Timing {
    Timing {
        variant: variant.to_string(),
        enqueue_ms_median: median_after_first(&r.enqueue_ms),
        step_ms_median: median_after_first(&r.step_ms),
    }
}

#[derive(Debug, Serialize)]
struct PromptReport {
    tokens: usize,
    comparisons: Vec<Comparison>,
    timings: Vec<Timing>,
}

#[derive(Debug, Serialize)]
struct Report {
    graphs: &'static str,
    steps: usize,
    variants: Vec<Variant>,
    prompts: Vec<PromptReport>,
    detection_control: Comparison,
    verdict: &'static str,
    reasons: Vec<String>,
}

/// 2026-09-28: Why a report fails; empty for a pass.
pub(crate) fn failures(prompts: &[Vec<Comparison>], control: &Comparison) -> Vec<String> {
    let mut out = Vec::new();
    for (p, cs) in prompts.iter().enumerate() {
        for c in cs {
            if !c.prefill_equal || c.mismatched_steps > 0 {
                out.push(format!(
                    "prompt {p}, {}: prefill equal {}, {} of {} steps differ (first at {:?})",
                    c.variant, c.prefill_equal, c.mismatched_steps, c.steps, c.first_mismatch
                ));
            }
        }
    }
    if control.mismatched_steps == 0 {
        out.push("the detection control saw no difference after changing a token".to_string());
    }
    out
}

/// 2026-09-28: Run `met circuit diff`.
pub(crate) fn run_diff(args: CircuitDiffArgs) -> Result<()> {
    ensure!(
        args.steps >= 2 && args.prompts >= 1,
        "need at least 2 steps and 1 prompt"
    );
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .try_init();
    let checkpoint = args
        .serve
        .model
        .clone()
        .context("the diff needs the checkpoint id as the model argument")?;
    // 2026-09-28: As `met serve` does before its load: validate the flags, then publish the
    // kernel-path cells (`--weight-quantization` among them) the build reads.
    if let Err(msg) = crate::cli::validate_serve_args(&args.serve) {
        bail!("{msg}");
    }
    crate::main_modules::serve_flags::publish_kernel_flags(&args.serve);
    let Some(engine) = load_engine(args.serve)? else {
        bail!("this rank is an expert-parallel worker; run the diff on the head");
    };
    let model = engine.model.as_ref();
    model.bind_gpu_to_thread()?;
    let instance = sources::instance_for(&checkpoint, &circuit_target(&engine.ptx_set.target)?)?;
    let modules = metrale_model_layers::circuit_exec::TargetModules(engine.ptx_set.modules.clone());
    let circuit = |fusions| ForwardSelect::Circuit {
        instance: Box::new(instance.clone()),
        fusions,
        modules: modules.clone(),
        config_json: engine.config_json.clone(),
    };
    let forwards = [
        ("legacy-repeat", ForwardSelect::Legacy),
        ("circuit-reference", circuit(Fusions::ReferenceOnly)),
        ("circuit", circuit(Fusions::All)),
    ];
    ensure!(
        args.batch.is_empty() || args.verify.is_empty(),
        "--batch and --verify are separate diffs"
    );
    if !args.batch.is_empty() {
        return batch_report(
            model,
            (&args.batch, args.fragment_slots),
            args.steps,
            &forwards,
            &args.out,
        );
    }
    if !args.verify.is_empty() {
        let prompt = &prompts(1, model.vocab_size())[0];
        return verify_report(
            model,
            prompt,
            (&args.verify, args.mtp),
            args.steps,
            &forwards,
            &args.out,
        );
    }
    let mut variants = Vec::new();
    let mut per_prompt = Vec::new();
    let mut reports = Vec::new();
    let mut control = None;
    for (p, prompt) in prompts(args.prompts, model.vocab_size()).iter().enumerate() {
        model.set_forward(&ForwardSelect::Legacy)?;
        let reference = run(model, prompt, args.steps, Feed::Greedy)?;
        let mut cs = Vec::new();
        let mut ts = vec![timing("legacy", &reference)];
        for (name, sel) in &forwards {
            model.set_forward(sel)?;
            if p == 0 {
                let d = model.forward_disclosure();
                variants.push(Variant {
                    name,
                    plan_digest: d.plan_digest,
                    launches_per_step: d.launches_per_step,
                });
            }
            let r = run(model, prompt, args.steps, Feed::Forced(&reference.tokens))?;
            let c = compare(name, &reference, &r);
            ts.push(timing(name, &r));
            tracing::info!("circuit diff: prompt {p} {name}: {c:?}");
            cs.push(c);
        }
        if p == 0 {
            let mut changed = reference.tokens.clone();
            let at = args.steps / 2;
            changed[at] = (changed[at] + 1) % model.vocab_size() as u32;
            let r = run(model, prompt, args.steps, Feed::Forced(&changed))?;
            control = Some(compare("circuit, token changed at step", &reference, &r));
        }
        reports.push(PromptReport {
            tokens: prompt.len(),
            comparisons: cs.clone(),
            timings: ts,
        });
        per_prompt.push(cs);
    }
    model.set_forward(&ForwardSelect::Legacy)?;
    let control = control.context("no prompt ran")?;
    let reasons = failures(&per_prompt, &control);
    let report = Report {
        graphs: if std::env::var("METRALE_DEBUG_NO_GRAPH").as_deref() == Ok("1") {
            "eager"
        } else {
            "graphed"
        },
        steps: args.steps,
        variants,
        prompts: reports,
        detection_control: control,
        verdict: if reasons.is_empty() { "PASS" } else { "FAIL" },
        reasons: reasons.clone(),
    };
    write_report(&args.out, &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !reasons.is_empty() {
        bail!("circuit diff FAILED:\n  {}", reasons.join("\n  "));
    }
    Ok(())
}

#[derive(Debug, Serialize)]
struct BatchReport {
    graphs: &'static str,
    fragment_slots: bool,
    steps: usize,
    variants: Vec<Variant>,
    widths: Vec<batch::WidthReport>,
    verdict: &'static str,
    reasons: Vec<String>,
}

/// 2026-09-28: The `--batch` diff: every width, then the verdict.
fn batch_report(
    model: &dyn Model,
    (widths, fragment): (&[usize], bool),
    steps: usize,
    forwards: &[(&'static str, ForwardSelect)],
    out: &Path,
) -> Result<()> {
    let variants = disclosed(model, forwards)?;
    let widths = batch::diff_widths(model, (widths, fragment), steps, forwards)?;
    let reasons = batch::batch_failures(&widths);
    let report = BatchReport {
        graphs: if std::env::var("METRALE_NO_DECODE_GRAPHS_MULTISEQ").as_deref() == Ok("1") {
            "eager"
        } else {
            "graphed"
        },
        fragment_slots: fragment,
        steps,
        variants,
        widths,
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

#[derive(Debug, Serialize)]
struct VerifyDiffReport {
    graphs: &'static str,
    mtp: bool,
    steps: usize,
    variants: Vec<Variant>,
    verify: Vec<verify::VerifyReport>,
    verdict: &'static str,
    reasons: Vec<String>,
}

/// 2026-09-29: The `--verify` diff: every `K`, then the verdict.
fn verify_report(
    model: &dyn Model,
    prompt: &[u32],
    (ks, mtp): (&[usize], bool),
    steps: usize,
    forwards: &[(&'static str, ForwardSelect)],
    out: &Path,
) -> Result<()> {
    let variants = disclosed(model, forwards)?;
    let verify = verify::diff_verify(model, prompt, ks, steps, mtp, forwards)?;
    let reasons = verify::verify_failures(&verify);
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

/// 2026-09-29: Each forward's disclosure, selecting each in turn.
fn disclosed(
    model: &dyn Model,
    forwards: &[(&'static str, ForwardSelect)],
) -> Result<Vec<Variant>> {
    let mut variants = Vec::new();
    for (name, sel) in forwards {
        model.set_forward(sel)?;
        let d = model.forward_disclosure();
        variants.push(Variant {
            name,
            plan_digest: d.plan_digest,
            launches_per_step: d.launches_per_step,
        });
    }
    Ok(variants)
}

fn write_report(path: &Path, report: &Report) -> Result<()> {
    std::fs::write(path, serde_json::to_vec_pretty(report)?)
        .with_context(|| format!("writing {}", path.display()))
}

#[path = "circuit_diff_batch.rs"]
mod batch;

#[path = "circuit_diff_verify.rs"]
mod verify;

#[cfg(test)]
#[path = "circuit_diff_tests.rs"]
mod circuit_diff_tests;
