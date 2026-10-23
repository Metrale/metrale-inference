// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The registered benchmarks that are not required PR gates, each with its reason,
//! split verbatim out of `coverage.rs` (the file-size cap); `coverage` re-exports it, so
//! `gate::coverage::NOT_REQUIRED` is the one path every caller names.
//!
//! Owner: bench (gate).
//! Invariants: every entry names a registered benchmark and a non-empty reason
//! (`every_excusal_names_a_real_benchmark_and_a_reason`).

/// 2026-09-26: Registered benchmarks that are not required gates, each with the reason
/// (`every_excusal_names_a_real_benchmark_and_a_reason`).
pub const NOT_REQUIRED: [(&str, &str); 14] = [
    (
        "quick-speed-bench",
        "a single-user speed probe with no thresholds and no baseline — a MEASUREMENT tool, \
         deliberately never a gate: the required gates already cost hours per PR, and \
         its warm-path numbers (primed prefix cache + SSM snapshot) are not regression evidence",
    ),
    (
        "bfcl-full",
        "the unsampled ~3600-sample draw; the two subset gates cover the same code at a \
         fraction of the GPU time, and a full run would dominate every PR",
    ),
    (
        "serve-matrix",
        "a multi-checkpoint survey used for release notes; it measures breadth, not regression",
    ),
    (
        "cross-contamination",
        "not required YET: a promotion candidate (see PROMOTION_CANDIDATES) run on release cuts \
         and recorded as debt until it has proven itself; a fresh gate that fails on day one \
         would train people to override it",
    ),
    (
        "scheduler-equivalence",
        "not required YET: a promotion candidate (see PROMOTION_CANDIDATES). It certifies the \
         asynchronous device router against the synchronous one, and `--scheduler-config \
         async` is not the default; it becomes required when that changes, on a green run \
         on every served model",
    ),
    (
        "kat-equality-gate-nemotron-nano",
        "not required YET: a promotion candidate (see PROMOTION_CANDIDATES), the Nemotron-3-Nano \
         correctness suite. Its BENCH.toml entry is unmeasured; it becomes required once three \
         runs set its bounds (measure, then declare)",
    ),
    (
        "kat-equality-gate-nemotron-super",
        "not required YET: a promotion candidate (see PROMOTION_CANDIDATES), the Nemotron-3-Super \
         correctness suite. Its BENCH.toml entry is unmeasured; it becomes required once three \
         runs set its bounds (measure, then declare)",
    ),
    (
        "bfcl-subset-mini-nemotron-nano",
        "not required YET: a promotion candidate (see PROMOTION_CANDIDATES), the Nemotron-3-Nano \
         accuracy check on the mini draw. Its floors are cut from three measured runs first",
    ),
    (
        "bfcl-subset-mini-nemotron-super",
        "not required YET: a promotion candidate (see PROMOTION_CANDIDATES), the Nemotron-3-Super \
         accuracy check on the mini draw. Its floors are cut from three measured runs first",
    ),
    (
        "bfcl-subset-mini",
        "the 192-sample mini draw itself, a measurement draw with no baseline: models are gated \
         on it through their own gate ids (bfcl-subset-mini-<model>), each with its own floors",
    ),
    (
        "default-tier-boot",
        "not required YET: a promotion candidate (see PROMOTION_CANDIDATES). It lands \
         informational, with no bounds, so its first records measure the default tier's memory \
         plan; it becomes required once BENCH.toml declares bounds from them",
    ),
    (
        "spec-cost",
        "a calibration MEASUREMENT for the speculative cost model, with no baseline and no \
         bounds: it records the per-step verify and draft cost at each batch width so the \
         cost table can be fitted, and a calibration has nothing to regress against",
    ),
    (
        "kernel-energy",
        "a per-kernel energy MEASUREMENT (J/launch, W) with no serve, no baseline and no \
         bounds: it writes docs/kernel-perf/measurements.toml evidence rows, and a measurement \
         row has nothing to regress against",
    ),
    (
        "mlperf-agentic-subset",
        "not RUNNABLE yet: the official MLPerf Agentic Inference dataset is unpublished \
         upstream (mlcommons/endpoints@7935df4: \"MLCommons storage (link TBD)\"), so the leg \
         cannot be run, scored, or timed, and it refuses proxy datasets on purpose. Not a \
         promotion candidate either — a candidate accrues debt rows, and debt nobody can \
         discharge is worse than no row. Promote only after the dataset ships and a \
         calibration run sizes a <2 h draw",
    ),
];
