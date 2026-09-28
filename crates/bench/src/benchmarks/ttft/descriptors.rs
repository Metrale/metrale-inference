// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The descriptors and metadata of the TTFT gates: warm (cached
//! prefix) and cold (uncached prefill), over synthetic prompts of several
//! lengths or (2026-09-27) the committed 32k-token prompt, one high-ISL pair
//! per subject. Each compares a run with a baseline the gate stores for the
//! same host and model.
//!
//! Owner: bench, ttft.
//! Invariants: none beyond the types.

use super::{Mode, TtftGate};
use crate::benchmark::{BenchmarkDescriptor, ModelExpectation};
use crate::hardware::Sensitivity;
use crate::metadata::PluginMetadata;

const WARM_SUMMARY: &str = "Cached-prefix TTFT vs the stored same-box baseline";
const COLD_SUMMARY: &str = "Uncached prefill TTFT vs the stored same-box baseline";
pub const WARM_METADATA: PluginMetadata = PluginMetadata::metrale(WARM_SUMMARY);
pub const COLD_METADATA: PluginMetadata = PluginMetadata::metrale(COLD_SUMMARY);

pub const WARM_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "ttft-warm-gate",
    name: "Warm TTFT Regression Gate",
    summary: WARM_SUMMARY,
    detail: "Measures time-to-first-token on the WARM path: each sample repeats a bit-identical \
             prompt so the prefix cache hits. Gates at median ≤3% and p90 ≤5% against a baseline \
             recorded on this box — the guard that catches an optimization silently falling back \
             to a slow path while the correctness gates stay green.",
    duration_hint: "~3–6 min",
    expected_secs: 160,
    updated: "2026-07-31",
    needs_confirmation: false,
    // 2026-09-26: The gate compares with a baseline it stores itself, keyed by
    // model and checked for the same host, so it applies to any checkpoint.
    intended_for: None,
    threshold_params: &[],
    // 2026-09-26: Speed: the verdict compares TTFT latencies, which a thermal
    // throttle during the run moves.
    sensitivity: Sensitivity::Speed,
    ctor: || Box::new(TtftGate::new(Mode::Warm)),
};

pub const COLD_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "ttft-cold-gate",
    name: "Cold TTFT Regression Gate",
    summary: COLD_SUMMARY,
    detail: "Measures time-to-first-token with the prefix cache guaranteed to MISS: every sample \
             carries a unique prefix_tag, so each request pays a full prefill. This is the prefill path \
             on its own, with the cache's contribution removed — the warm gate cannot see a \
             prefill regression that caching is hiding.",
    duration_hint: "~3–6 min",
    expected_secs: 120,
    updated: "2026-07-31",
    needs_confirmation: false,
    // 2026-09-26: The gate compares with a baseline it stores itself, keyed by
    // model and checked for the same host, so it applies to any checkpoint.
    intended_for: None,
    threshold_params: &[],
    // 2026-09-26: Speed, for the same reason as the warm gate.
    sensitivity: Sensitivity::Speed,
    ctor: || Box::new(TtftGate::new(Mode::Cold)),
};

const HIGH_ISL_COLD_SUMMARY: &str = "Uncached 32k-token prefill TTFT on the dense flagship";
const HIGH_ISL_WARM_SUMMARY: &str = "Cached 32k-token prefix TTFT on the dense flagship";
const HIGH_ISL_COLD_MOE_SUMMARY: &str = "Uncached 32k-token prefill TTFT on the 35B MoE";
const HIGH_ISL_WARM_MOE_SUMMARY: &str = "Cached 32k-token prefix TTFT on the 35B MoE";
const HIGH_ISL_COLD_METADATA: PluginMetadata = PluginMetadata::metrale(HIGH_ISL_COLD_SUMMARY);
const HIGH_ISL_WARM_METADATA: PluginMetadata = PluginMetadata::metrale(HIGH_ISL_WARM_SUMMARY);
const HIGH_ISL_COLD_MOE_METADATA: PluginMetadata =
    PluginMetadata::metrale(HIGH_ISL_COLD_MOE_SUMMARY);
const HIGH_ISL_WARM_MOE_METADATA: PluginMetadata =
    PluginMetadata::metrale(HIGH_ISL_WARM_MOE_SUMMARY);

/// 2026-09-27: The high-ISL gates' checkpoint families: one default subject per
/// gate id, as with `concurrency-sweep` and `concurrency-sweep-moe`.
const DENSE: ModelExpectation = ModelExpectation {
    families: &["qwen3.8-27b"],
    note: "The high-ISL pair without the -moe suffix is defined on the dense Qwen3.8-27B \
           flagship; the -moe pair measures the Qwen3.6-35B-A3B MoE.",
};
const MOE: ModelExpectation = ModelExpectation {
    families: &["qwen3.6-35b-a3b"],
    note: "The -moe high-ISL pair is defined on the Qwen3.6-35B-A3B MoE; the pair without \
           the suffix measures the dense Qwen3.8-27B flagship.",
};

/// 2026-09-27: `expected_secs` below are one-shot runs (owner, 2026-09-27) at
/// the 2026-09-27 measurements: one 32k-token prefill per cold run (MoE 14 s,
/// dense 45 s), plus a priming prefill before the warm one.
pub const HIGH_ISL_COLD_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "high-isl-ttft-cold",
    name: "High-ISL Cold TTFT Gate",
    summary: HIGH_ISL_COLD_SUMMARY,
    detail: "Time-to-first-token on a committed 32k-token prompt (Moby-Dick, see \
             ttft/prompts/NOTICE.md), one-shot: an unmeasured short warm-up request, then \
             one request with a unique tag at the start, so it pays the whole prefill. The \
             server's usage.prompt_tokens is checked: a missing count or one below \
             min_prompt_tokens makes the run invalid. Compared with a same-box baseline like \
             ttft-cold-gate; the BENCH.toml ceiling is vLLM's TTFT on the same prompt and box.",
    duration_hint: "~1–2 min",
    expected_secs: 70,
    updated: "2026-09-27",
    needs_confirmation: false,
    intended_for: Some(DENSE),
    threshold_params: &[],
    // 2026-09-27: Speed, as for the synthetic TTFT gates.
    sensitivity: Sensitivity::Speed,
    ctor: || {
        Box::new(TtftGate::high_isl(
            Mode::Cold,
            &HIGH_ISL_COLD_DESCRIPTOR,
            &HIGH_ISL_COLD_METADATA,
        ))
    },
};

pub const HIGH_ISL_WARM_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "high-isl-ttft-warm",
    name: "High-ISL Warm TTFT Gate",
    summary: HIGH_ISL_WARM_SUMMARY,
    detail: "Time-to-first-token on the committed 32k-token prompt with one fixed tag, \
             one-shot: an unmeasured short warm-up request, then the prompt is primed and \
             re-sent byte for byte, so the prefix cache holds the whole prompt. The server's \
             usage.prompt_tokens is checked, as in high-isl-ttft-cold.",
    duration_hint: "~1–2 min",
    expected_secs: 80,
    updated: "2026-09-27",
    needs_confirmation: false,
    intended_for: Some(DENSE),
    threshold_params: &[],
    sensitivity: Sensitivity::Speed,
    ctor: || {
        Box::new(TtftGate::high_isl(
            Mode::Warm,
            &HIGH_ISL_WARM_DESCRIPTOR,
            &HIGH_ISL_WARM_METADATA,
        ))
    },
};

pub const HIGH_ISL_COLD_MOE_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "high-isl-ttft-cold-moe",
    name: "High-ISL Cold TTFT Gate (MoE)",
    summary: HIGH_ISL_COLD_MOE_SUMMARY,
    detail: "high-isl-ttft-cold on the Qwen3.6-35B-A3B MoE: the same prompt and the same \
             checks, with its own id so its baseline and bounds are read against its own \
             history only.",
    duration_hint: "~1 min",
    expected_secs: 30,
    updated: "2026-09-27",
    needs_confirmation: false,
    intended_for: Some(MOE),
    threshold_params: &[],
    sensitivity: Sensitivity::Speed,
    ctor: || {
        Box::new(TtftGate::high_isl(
            Mode::Cold,
            &HIGH_ISL_COLD_MOE_DESCRIPTOR,
            &HIGH_ISL_COLD_MOE_METADATA,
        ))
    },
};

pub const HIGH_ISL_WARM_MOE_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "high-isl-ttft-warm-moe",
    name: "High-ISL Warm TTFT Gate (MoE)",
    summary: HIGH_ISL_WARM_MOE_SUMMARY,
    detail: "high-isl-ttft-warm on the Qwen3.6-35B-A3B MoE: the same prompt and the same \
             checks, with its own id so its baseline and bounds are read against its own \
             history only.",
    duration_hint: "~1 min",
    expected_secs: 30,
    updated: "2026-09-27",
    needs_confirmation: false,
    intended_for: Some(MOE),
    threshold_params: &[],
    sensitivity: Sensitivity::Speed,
    ctor: || {
        Box::new(TtftGate::high_isl(
            Mode::Warm,
            &HIGH_ISL_WARM_MOE_DESCRIPTOR,
            &HIGH_ISL_WARM_MOE_METADATA,
        ))
    },
};
