// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The BFCL `mini` draw ([`DrawSpec::mini`](super::draw::DrawSpec::mini), 192
//! samples) and the per-model gate ids that run it as a model's cheap accuracy check:
//! `bfcl-subset-mini` itself is a measurement draw, and `bfcl-subset-mini-nemotron-nano` /
//! `-nemotron-super` are promotion candidates, each with its own default subject in its model's
//! BENCH.toml (one required subject per gate id per box class).
//!
//! Owner: bench, BFCL benchmark.
//! Invariants: none beyond the types.

use super::{Bfcl, Variant};
use crate::benchmark::{BenchmarkDescriptor, ModelExpectation};
use crate::hardware::Sensitivity;
use crate::metadata::PluginMetadata;

const MINI_SUMMARY: &str = "The mini n=192 draw (golden's mix at a fifth), AST-scored";

const MINI_DETAIL: &str = "Berkeley Function Calling Leaderboard v4, single-turn, on the mini \
     draw: the golden draw's three categories at 12.5/2/2 with a floor of 5, which is 192 \
     samples. A model's cheap accuracy check, run whole (not sharded). Its scores are NOT \
     comparable to the golden or echolp draws; each gated model carries its own floors. The \
     record's dataset_fingerprint names the draw: the SHA-256 of the ordered sample ids, N and \
     the category percentages.";

pub const MINI_METADATA: PluginMetadata = PluginMetadata::metrale(MINI_SUMMARY);

const GATE_THRESHOLD_PARAMS: &[(&str, &str)] = &[
    ("min_overall", "overall_accuracy"),
    ("min_normalized", "normalized_single_turn_score"),
];

pub const SUBSET_MINI_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "bfcl-subset-mini",
    name: "BFCL (subset, mini draw)",
    summary: MINI_SUMMARY,
    detail: MINI_DETAIL,
    duration_hint: "~25 min on a 30B-A3B",
    expected_secs: 1500,
    updated: "2026-10-03",
    needs_confirmation: false,
    intended_for: None,
    threshold_params: GATE_THRESHOLD_PARAMS,
    sensitivity: Sensitivity::Correctness,
    ctor: || Box::new(Bfcl::new(Variant::SubsetMini)),
};

/// 2026-10-03: The mini draw on NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4, a separate gate id so the
/// model has its own required subject (`gate::check::record_is_required_subject`).
pub const MINI_NEMOTRON_NANO_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "bfcl-subset-mini-nemotron-nano",
    name: "BFCL (mini draw, Nemotron-3-Nano)",
    summary: MINI_SUMMARY,
    detail: MINI_DETAIL,
    duration_hint: "~25 min",
    expected_secs: 1500,
    updated: "2026-10-03",
    needs_confirmation: false,
    intended_for: Some(ModelExpectation {
        families: &["nemotron-3-nano-30b-a3b"],
        note: "The Nemotron-3-Nano accuracy check; its floors are cut from its own runs.",
    }),
    threshold_params: GATE_THRESHOLD_PARAMS,
    sensitivity: Sensitivity::Correctness,
    ctor: || Box::new(Bfcl::new(Variant::SubsetMini)),
};

/// 2026-10-03: The same on NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4.
pub const MINI_NEMOTRON_SUPER_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "bfcl-subset-mini-nemotron-super",
    name: "BFCL (mini draw, Nemotron-3-Super)",
    summary: MINI_SUMMARY,
    detail: MINI_DETAIL,
    duration_hint: "~45 min",
    expected_secs: 2700,
    updated: "2026-10-03",
    needs_confirmation: false,
    intended_for: Some(ModelExpectation {
        families: &["nemotron-super-120b-a12b"],
        note: "The Nemotron-3-Super accuracy check; its floors are cut from its own runs.",
    }),
    threshold_params: GATE_THRESHOLD_PARAMS,
    sensitivity: Sensitivity::Correctness,
    ctor: || Box::new(Bfcl::new(Variant::SubsetMini)),
};
