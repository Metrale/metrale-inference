// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: `kernel-energy`: J per launch and W of one kernel at one plan shape, by the
//! steady-state method of DESIGN-kernel-energy.md §3.1. A measurement tool, not a gate
//! (`gate::coverage::NOT_REQUIRED`), and it needs no serve (OWAY): the endpoint the executor is
//! given is not contacted.
//!
//! `mode = measure` runs the `kernel_energy_loop` harness (`crates/gpu-examples`, path in
//! `harness`), which samples every available power meter while the kernel loops between
//! interleaved idle baselines and writes the raw trace to `trace`; this benchmark then derives the
//! row from that trace alone. `mode = recompute` derives the row from an existing `trace` (the
//! HPAR check: the same trace gives the same row bytes). Writing the row into
//! `docs/kernel-perf/measurements.toml` is the explicit, separate `kernel_energy write` step.
//!
//! Owner: bench, kernel-energy.
//! Invariants: the verdict is never a pass: info when a row was derived; every unstated
//! parameter (`unstated`) is refused (PCND).

use std::collections::BTreeMap;
use std::future::Future;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use metrale_power::derive::derive;
use metrale_power::trace::{PowerTrace, sha256_hex};

use crate::benchmark::{Benchmark, BenchmarkDescriptor};
use crate::hardware::Sensitivity;
use crate::metadata::PluginMetadata;
use crate::params::{ParamKind, ParamSpec, ParamValue, ParamValues};
use crate::plugin::{Plugin, PluginHandle};
use crate::result::{BenchmarkResult, LogLine, RunStatus, Verdict};

pub mod writer;

const SUMMARY: &str = "Per-kernel energy: J per launch and W of one kernel at one plan shape \
                       (steady state between interleaved idle baselines; every power meter)";
pub const METADATA: PluginMetadata = PluginMetadata::metrale(SUMMARY);

pub const DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "kernel-energy",
    name: "Kernel energy (J/launch, W)",
    summary: SUMMARY,
    detail: "Runs the kernel_energy_loop harness: 2 s warm-up, then idle, load, idle, … (every \
             load flanked by idle baselines A and B), the kernel looped as a CUDA graph of 16 \
             launches over weights rotated across 128 MB, while every power meter that opens \
             is sampled concurrently (NVML energy counter, NVML power sensor, nvidia-smi, \
             hwmon; one that cannot open is recorded with its reason). The trace is written \
             beside the row; J/launch total and dynamic, W and static W come with the envelope \
             of a variance-components interval and a hierarchical BCa bootstrap, every \
             source's reading, the signed delta, Deming fit and Bland-Altman limits of each \
             source pair, ESS, outliers, seed and order. Needs no serve; the endpoint is not \
             contacted. Informational: no bounds.",
    duration_hint: "~2 min per row",
    expected_secs: 120,
    updated: "2026-10-22",
    intended_for: None,
    threshold_params: &[],
    needs_confirmation: false,
    // 2026-10-22: Joules per launch are rates of a clocked device: a throttled box reports energy
    // the kernel did not cause.
    sensitivity: Sensitivity::Speed,
    ctor: || Box::new(KernelEnergy::default()),
};

/// 2026-10-22: The value of a text parameter that means "not stated" and is refused.
const UNSTATED: &str = "unstated";

#[derive(Default)]
pub struct KernelEnergy {
    handle: Option<PluginHandle>,
    started: Option<Instant>,
    args: Vec<(String, String)>,
    harness: String,
    trace: String,
    recompute: bool,
}

fn text(key: &'static str, label: &'static str, help: &'static str) -> ParamSpec {
    ParamSpec::new(
        key,
        label,
        help,
        ParamKind::Text,
        ParamValue::Text(UNSTATED.to_string()),
    )
}

fn int(
    key: &'static str,
    label: &'static str,
    help: &'static str,
    min: i64,
    max: i64,
    v: i64,
) -> ParamSpec {
    ParamSpec::new(
        key,
        label,
        help,
        ParamKind::Int { min, max },
        ParamValue::Int(v),
    )
}

/// 2026-10-22: Derive the row of the trace at `path`; returns (sha256, TOML lines).
pub fn recompute(path: &str) -> Result<(String, Vec<String>)> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading trace {path}"))?;
    let sha = sha256_hex(text.as_bytes());
    let t = PowerTrace::decode(&text).map_err(|e| anyhow::anyhow!("{path}: {e}"))?;
    let d = derive(&t, &sha).map_err(|e| anyhow::anyhow!("{path}: {e}"))?;
    Ok((sha, d.row.toml_lines()))
}

impl KernelEnergy {
    fn elapsed(&self) -> std::time::Duration {
        self.started.map(|s| s.elapsed()).unwrap_or_default()
    }
}

impl Plugin for KernelEnergy {
    fn metadata(&self) -> &'static PluginMetadata {
        &METADATA
    }

    fn load(&mut self, handle: PluginHandle) -> impl Future<Output = Result<()>> + Send {
        self.handle = Some(handle);
        self.started = Some(Instant::now());
        async { Ok(()) }
    }
}

impl Benchmark for KernelEnergy {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        &DESCRIPTOR
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        vec![
            ParamSpec::new(
                "mode",
                "Mode",
                "measure: run the harness; recompute: derive the row from an existing trace.",
                ParamKind::Choice(&["measure", "recompute"]),
                ParamValue::Text("measure".to_string()),
            ),
            text(
                "harness",
                "Harness",
                "Path to the kernel_energy_loop binary (measure).",
            ),
            text(
                "trace",
                "Trace",
                "Where the raw trace is written (measure) or read (recompute).",
            ),
            text("kernel", "Kernel", "Harness arm, e.g. w4a16_gemv_pipe."),
            text("shape", "Shape", "Plan shape, e.g. `kda q|k|v|f_a|b|g_a`."),
            text(
                "platform",
                "Platform",
                "Declared platform (gb10, h100, gfx1151, apple, ...).",
            ),
            text(
                "fingerprint",
                "Fingerprint",
                "GPU, driver, VBIOS, kernel, clock64 MHz, box, binary.",
            ),
            text(
                "energy_source",
                "Provenance",
                "One line naming the run (commit, box, date).",
            ),
            int(
                "rows",
                "Rows",
                "Rows per launch (the plan's row count).",
                1,
                128,
                1,
            ),
            int(
                "repeats",
                "Repeats",
                "Load windows; >= 3 for a measured row.",
                1,
                20,
                3,
            ),
            int(
                "idle_s",
                "Idle s",
                "Each idle baseline, s (>= 10 of record; >= 3 floor).",
                3,
                120,
                10,
            ),
            int(
                "load_s",
                "Load s",
                "Each load window, s (>= 10 of record; >= 3 floor).",
                3,
                120,
                10,
            ),
            int(
                "sample_ms",
                "Sample ms",
                "Sampler tick, ms (40 Hz = 25).",
                5,
                1000,
                25,
            ),
            int(
                "seed",
                "Seed",
                "Bootstrap seed, stored with the row.",
                0,
                i64::MAX,
                0,
            ),
            int(
                "bootstrap_reps",
                "Bootstrap reps",
                "BCa replicates.",
                200,
                100_000,
                2000,
            ),
            ParamSpec::new(
                "ci_level",
                "CI level",
                "Two-sided confidence level.",
                ParamKind::Float {
                    min: 0.5,
                    max: 0.999,
                },
                ParamValue::Float(0.95),
            ),
        ]
    }

    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        values.validate_against(&self.parameters())?;
        self.recompute = values.text("mode")? == "recompute";
        self.trace = values.text("trace")?.to_string();
        if self.trace == UNSTATED {
            bail!("trace: state the trace path with --param trace=...; it has no default");
        }
        self.args.clear();
        if self.recompute {
            return Ok(());
        }
        self.harness = values.text("harness")?.to_string();
        for k in [
            "harness",
            "kernel",
            "shape",
            "platform",
            "fingerprint",
            "energy_source",
        ] {
            if values.text(k)? == UNSTATED {
                bail!("{k}: state it with --param {k}=...; it has no default");
            }
        }
        for k in [
            "kernel",
            "shape",
            "platform",
            "fingerprint",
            "energy_source",
        ] {
            self.args
                .push((k.replace('_', "-"), values.text(k)?.to_string()));
        }
        for k in [
            "rows",
            "repeats",
            "idle_s",
            "load_s",
            "sample_ms",
            "seed",
            "bootstrap_reps",
        ] {
            self.args
                .push((k.replace('_', "-"), values.usize(k)?.to_string()));
        }
        self.args.push((
            "ci-level".to_string(),
            values.float("ci_level")?.to_string(),
        ));
        self.args.push(("out".to_string(), self.trace.clone()));
        Ok(())
    }

    async fn next(&mut self) -> Result<BenchmarkResult> {
        let handle = self
            .handle
            .as_ref()
            .context("benchmark was not loaded")?
            .clone();
        handle.check_cancelled()?;
        let mut log = Vec::new();
        if !self.recompute {
            let mut cmd = tokio::process::Command::new(&self.harness);
            for (k, v) in &self.args {
                cmd.arg(format!("--{k}")).arg(v);
            }
            let out = cmd
                .output()
                .await
                .with_context(|| format!("running {}", self.harness))?;
            if !out.status.success() {
                bail!(
                    "the harness failed: {}",
                    String::from_utf8_lossy(&out.stderr)
                );
            }
            log.push(LogLine::info(
                String::from_utf8_lossy(&out.stdout).trim().to_string(),
            ));
        }
        let (sha, lines) = recompute(&self.trace)?;
        let mut metrics = BTreeMap::new();
        for l in &lines {
            let Some((k, v)) = l.split_once(" = ") else {
                continue;
            };
            if let Ok(x) = v.parse::<f64>() {
                metrics.insert(k.to_string(), x);
            }
        }
        let mut r = BenchmarkResult {
            status: RunStatus::Completed,
            ..BenchmarkResult::running("done", self.elapsed())
        }
        .with_metrics(metrics)
        .with_verdict(Verdict::info(format!(
            "row derived from trace {} ({sha})",
            self.trace
        )));
        for l in log.into_iter().chain(lines.into_iter().map(LogLine::info)) {
            r = r.log_line(l);
        }
        Ok(r)
    }
}

#[cfg(test)]
#[path = "kernel_energy_tests.rs"]
mod tests;
