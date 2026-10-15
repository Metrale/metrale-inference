// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The `agentic-webserver` benchmark's descriptor and metadata.
//!
//! Owner: bench, agentic.
//! Invariants: none beyond the types.

use super::AgenticWebserver;
use crate::benchmark::BenchmarkDescriptor;
use crate::hardware::Sensitivity;
use crate::metadata::PluginMetadata;

const SUMMARY: &str = "N agentic runs: build a working Axum server, then verify it";
pub const METADATA: PluginMetadata = PluginMetadata::metrale(SUMMARY);

/// 2026-09-26: Under `--pull-request-gate` each model's serve recipe and metric
/// bounds come from its `BENCH.toml` entry for this gate; for the 35B MoE that
/// is recipe `qwen3.6/qwen3.6-35b-a3b-fp8-bf16head`.
pub const DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "agentic-webserver",
    name: "Agentic Webserver Test",
    summary: SUMMARY,
    detail: "Runs the flagship agentic task N times: the model writes a Rust Axum ping/pong \
             server, tests it, runs it and tears it down, using bash/write_file/read_file tools \
             in a fresh sandbox. Each run is scored on OUTCOME (the scorer builds it and gets a \
             'pong') and on PROCESS (did the agent do all six things the prompt asked?), plus \
             wall time. RUNS MODEL-AUTHORED SHELL inside the sandbox directory.",
    duration_hint: "~5 min per iteration",
    expected_secs: 600,
    updated: "2026-08-14",
    needs_confirmation: true,
    // 2026-09-26: The qwen3.6-35b-a3b entry is the default variant. The
    // qwen3.6-27b entry is `unmeasured` and has no metric bounds; the
    // qwen3.8-27b entry is `measured` and has its own.
    intended_for: Some(crate::benchmark::ModelExpectation {
        families: &["qwen3.6-35b-a3b", "qwen3.6-27b", "qwen3.8-27b"],
        note: "This benchmark is defined on the 35B MoE flagship (Qwen3.6-35B-A3B, FP8 or \
               NVFP4 — the required Gate A subject) and on both dense 27B variants. \
               Qwen3.6-27B is registered but UNMEASURED: its BENCH.toml entry has no \
               thresholds, so a run there baselines, it does not gate. Qwen3.8-27B is \
               MEASURED and gates against its own thresholds. Each variant carries its own \
               thresholds and serve recipe; any other checkpoint would produce numbers that \
               compare to nothing.",
    }),
    // 2026-09-26: Unless given with `--param`, `wall_budget_s` and
    // `s_per_turn_budget` take the selected variant's `sum_wall_s` and
    // `s_per_turn` bounds, when that entry has them.
    threshold_params: &[
        ("wall_budget_s", "sum_wall_s"),
        ("s_per_turn_budget", "s_per_turn"),
    ],
    // 2026-09-26: `webserver_ok` and `followed_directions` are correctness
    // counts, but `wall_budget_s` and `s_per_turn_budget` bound speed, so the
    // benchmark is classed as Speed.
    sensitivity: Sensitivity::Speed,
    ctor: || Box::new(AgenticWebserver::new(&DESCRIPTOR)),
};

/// 2026-10-15: `agentic-webserver` on GLM-5.3-Flash under its own id, so one gate id keeps one
/// default subject per hardware and GLM's bounds are read against its own records only. The
/// driver, prompt, scoring and threshold couplings are the flagship's.
pub const GLM_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "agentic-webserver-glm",
    name: "Agentic Webserver Test (GLM)",
    summary: SUMMARY,
    detail: "agentic-webserver on GLM-5.3-Flash served at 512k context: the same task, sandbox, \
             scoring and wall-time bounds, recorded under its own id. RUNS MODEL-AUTHORED SHELL \
             inside the sandbox directory.",
    duration_hint: "~1 min per iteration",
    expected_secs: 480,
    updated: "2026-10-15",
    needs_confirmation: true,
    intended_for: Some(crate::benchmark::ModelExpectation {
        families: &["glm-5.3-flash"],
        note: "agentic-webserver-glm is defined on GLM-5.3-Flash; agentic-webserver measures \
               the Qwen subjects.",
    }),
    threshold_params: DESCRIPTOR.threshold_params,
    sensitivity: Sensitivity::Speed,
    ctor: || Box::new(AgenticWebserver::new(&GLM_DESCRIPTOR)),
};
