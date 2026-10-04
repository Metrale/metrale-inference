// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Per-model gate ids for the KAT equality check on the Nemotron-3 models: the same
//! driver (`driver::KatEquality`), each id with its own default subject in its model's
//! BENCH.toml, because a gate id has one required subject per box class
//! (`gate::check::record_is_required_subject`). Promotion candidates until their BENCH entries
//! are measured.
//!
//! Owner: bench, kat_equality.
//! Invariants: none beyond the types.

use crate::benchmark::{BenchmarkDescriptor, ModelExpectation};
use crate::hardware::Sensitivity;

use super::driver::{DESCRIPTOR, KatEquality};

const DETAIL: &str = "The KAT equality check (see `kat-equality-gate`) on one Nemotron-3 \
     checkpoint, at the sample cap its BENCH.toml entry pins: one BFCL draw in two request \
     orders at temperature 0, every reply byte-identical across the orders, served hermetic.";

pub const NEMOTRON_NANO_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "kat-equality-gate-nemotron-nano",
    name: "KAT Equality Gate (Nemotron-3-Nano)",
    summary: DESCRIPTOR.summary,
    detail: DETAIL,
    duration_hint: "~20 min at sample_cap 64",
    expected_secs: 1200,
    updated: "2026-10-03",
    needs_confirmation: false,
    intended_for: Some(ModelExpectation {
        families: &["nemotron-3-nano-30b-a3b"],
        note: "The Nemotron-3-Nano order-independence check.",
    }),
    threshold_params: &[],
    sensitivity: Sensitivity::Correctness,
    ctor: || Box::new(KatEquality::default()),
};

pub const NEMOTRON_SUPER_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "kat-equality-gate-nemotron-super",
    name: "KAT Equality Gate (Nemotron-3-Super)",
    summary: DESCRIPTOR.summary,
    detail: DETAIL,
    duration_hint: "~35 min at sample_cap 64",
    expected_secs: 2100,
    updated: "2026-10-03",
    needs_confirmation: false,
    intended_for: Some(ModelExpectation {
        families: &["nemotron-super-120b-a12b"],
        note: "The Nemotron-3-Super order-independence check.",
    }),
    threshold_params: &[],
    sensitivity: Sensitivity::Correctness,
    ctor: || Box::new(KatEquality::default()),
};
