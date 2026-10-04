// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Speculative step cost calibration: on the serve it is pointed at, the wall time
//! and GPU-rail energy of ONE speculative scheduler step at each batch width `n`, for the
//! draft depth `k` the serve runs, split into its draft (propose) and verify shares. A
//! measurement tool for the speculative cost model, not a gate: `gate::coverage::NOT_REQUIRED`
//! lists it.
//!
//! Per width, ascending (`wave.rs`): `n` concurrent streaming essay requests, each with its
//! own prompt identity; once every stream has delivered a token and `settle_s` has passed,
//! two `/metrics` snapshots `window_s` apart and the rail energy between them; then the
//! streams are closed and the serve drained to idle. The arithmetic, and its proportional
//! energy split, are in `cell.rs`.
//!
//! The pins (the benchmark's definition, not parameters):
//! - Prompt: `wave::PROMPT_TOKENS` of `stats::make_prompt` filler, then the concurrency
//!   sweep's essay ask (`concurrency::ESSAY_TASK`), which keeps a model writing.
//! - Request: temperature 0, seed 0, presence and frequency penalty 0, and
//!   `reasoning_effort: "none"` (thinking off), as in `decode_floor`: speculative dispatch is
//!   off inside `<think>` unless `METRALE_DFLASH_SPEC_THINK=1`.
//! - A speculative serve (`k` > 0) must run with `--telemetry basic` or `kernel`, which
//!   exports the scheduler phase histogram; without it the run fails.
//!
//! Owner: bench, spec-cost.
//! Invariants: the run verdict is never a pass: info when at least one width was measured,
//! INCONCLUSIVE otherwise.

use crate::hardware::Sensitivity;
use std::collections::BTreeMap;
use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

use crate::benchmark::{Benchmark, BenchmarkDescriptor};
use crate::hardware::energy_sampler::EnergyMeter;
use crate::http;
use crate::metadata::PluginMetadata;
use crate::params::{ParamKind, ParamSpec, ParamValue, ParamValues};
use crate::plugin::{Plugin, PluginHandle};
use crate::result::{
    BenchmarkResult, Cell, CellStyle, Column, LogLine, ResultTable, RunStatus, Verdict,
};

pub mod acceptance;
mod cell;
mod prom;
pub mod table_input;
mod wave;

use cell::CellVerdict;

const SUMMARY: &str = "Speculative step cost calibration: wall time and GPU-rail energy of one \
                       speculative step per batch width, split into draft and verify";
pub const METADATA: PluginMetadata = PluginMetadata::metrale(SUMMARY);

pub const DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "spec-cost",
    name: "Speculative step cost calibration",
    summary: SUMMARY,
    detail: "For each batch width n (param widths, ascending), sends n concurrent streaming \
             essay requests (distinct prompts, temperature 0, seed 0, thinking off, max_tokens \
             = osl). Once every stream has a token and settle_s has passed, it reads /metrics \
             twice, window_s apart, and integrates the GPU rail over the window. Per width it \
             records n{n}_steps, n{n}_verify_ms, n{n}_verify_j, n{n}_draft_ms, n{n}_draft_j, \
             n{n}_tok_per_step, n{n}_wall_ms, n{n}_nvml_j and n{n}_vacuous, plus k. Steps and \
             their time come from the scheduler phase histogram (phase step_mtp, its propose \
             share from phase propose), so a speculative serve needs --telemetry basic; at \
             k = 0 a step is one token per sequence. Joules are the bench's GPU-rail integral \
             over the window; n{n}_nvml_j is the change in the serve's NVML energy counter per \
             step, a cross-check that repeats less well. Energy is split between draft and \
             verify in proportion to time (an assumption, not a measurement). A width is \
             vacuous when a stream ends \
             inside the window, fewer than 20 steps were seen, or the serve did not time the \
             propose phase. Informational: no bounds.",
    duration_hint: "~2 min",
    expected_secs: 120,
    updated: "2026-10-04",
    intended_for: None,
    threshold_params: &[],
    needs_confirmation: false,
    // 2026-10-04: Step time and joules per step are rates: a throttled box reports a cost the
    // code did not cause.
    sensitivity: Sensitivity::Speed,
    ctor: || Box::new(SpecCost::default()),
};

/// 2026-10-04: The `k` value that means "not stated"; `configure` refuses it, so the caller
/// must state the depth the serve runs.
const UNSTATED_K: i64 = -1;
/// 2026-10-04: The deepest draft the serve's verify arms take (K4 = three drafts).
const MAX_K: i64 = 3;

#[derive(Default)]
pub struct SpecCost {
    handle: Option<PluginHandle>,
    started: Option<Instant>,
    timeout: Duration,
    k: u8,
    widths: Vec<usize>,
    osl: usize,
    window: Duration,
    settle: Duration,
    probed: bool,
    /// 2026-10-04: The probe's `/metrics` page, the start of the acceptance counts.
    first_scrape: Option<prom::Scrape>,
    rows: Vec<(usize, CellVerdict)>,
    /// 2026-10-04: GPU-rail sampling, started after the probe (idle baseline).
    energy: EnergyMeter,
}

impl SpecCost {
    fn handle(&self) -> Result<&PluginHandle> {
        self.handle.as_ref().context("benchmark was not loaded")
    }

    fn elapsed(&self) -> Duration {
        self.started.map(|s| s.elapsed()).unwrap_or_default()
    }

    fn table(&self) -> ResultTable {
        let mut t = ResultTable::new(
            "SPECULATIVE STEP COST",
            vec![
                Column::right("n", 4),
                Column::right("Steps", 7),
                Column::right("Verify ms", 10),
                Column::right("Verify J", 9),
                Column::right("Draft ms", 9),
                Column::right("Draft J", 8),
                Column::right("Tok/step", 9),
            ],
        );
        let joules = |v: Option<f64>| v.map(|v| format!("{v:.3}")).unwrap_or_else(|| "—".into());
        for (n, verdict) in &self.rows {
            let row = match verdict {
                CellVerdict::Vacuous(_) => {
                    let mut row = vec![
                        Cell::new(n.to_string()),
                        Cell::styled("vacuous", CellStyle::Warn),
                    ];
                    row.extend((0..5).map(|_| Cell::new("—")));
                    row
                }
                CellVerdict::Measured(c) => vec![
                    Cell::new(n.to_string()),
                    Cell::new(format!("{:.0}", c.steps)),
                    Cell::styled(format!("{:.2}", c.verify_ms), CellStyle::Accent),
                    Cell::new(joules(c.verify_j)),
                    Cell::new(format!("{:.2}", c.draft_ms)),
                    Cell::new(joules(c.draft_j)),
                    Cell::new(format!("{:.2}", c.tok_per_step)),
                ],
            };
            t.push(row);
        }
        t
    }

    fn finish(&self, mut metrics: BTreeMap<String, f64>) -> BenchmarkResult {
        metrics.insert(cell::KEY_K.to_string(), f64::from(self.k));
        for (n, verdict) in &self.rows {
            cell::record(*n, verdict, &mut metrics);
        }
        self.energy.metrics(&mut metrics);
        let measured = self
            .rows
            .iter()
            .filter(|(_, v)| matches!(v, CellVerdict::Measured(_)))
            .count();
        let verdict = if measured == 0 {
            Verdict::fail("INCONCLUSIVE: every width was vacuous")
        } else {
            Verdict::info(format!(
                "{measured} of {} widths measured at k = {} — a calibration table, no bounds",
                self.rows.len(),
                self.k
            ))
        };
        let total = self.widths.len() as u64;
        BenchmarkResult {
            status: RunStatus::Completed,
            ..BenchmarkResult::running("done", self.elapsed())
        }
        .with_progress(total, total)
        .with_table(self.table())
        .with_metrics(metrics)
        .with_verdict(verdict)
    }
}

impl Plugin for SpecCost {
    fn metadata(&self) -> &'static PluginMetadata {
        &METADATA
    }

    fn load(&mut self, handle: PluginHandle) -> impl Future<Output = Result<()>> + Send {
        self.handle = Some(handle);
        self.started = Some(Instant::now());
        async { Ok(()) }
    }
}

impl Benchmark for SpecCost {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        &DESCRIPTOR
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        vec![
            ParamSpec::new(
                "k",
                "Draft depth",
                "Drafts per speculative step the serve runs (0 = a serve without \
                 --speculative). Required: the default -1 means not stated and is refused.",
                ParamKind::Int {
                    min: UNSTATED_K,
                    max: MAX_K,
                },
                ParamValue::Int(UNSTATED_K),
            ),
            ParamSpec::new(
                "widths",
                "Batch widths",
                "Concurrent streams per measured window, strictly ascending.",
                ParamKind::IntList { min: 1, max: 256 },
                ParamValue::IntList(vec![1, 2, 4, 8, 16]),
            ),
            ParamSpec::new(
                "osl",
                "Output budget",
                "max_tokens of every stream. Must outlast settle + window at the serve's \
                 per-stream rate, or the width is vacuous.",
                ParamKind::Int {
                    min: 64,
                    max: 16384,
                },
                ParamValue::Int(1024),
            ),
            ParamSpec::new(
                "window_s",
                "Window",
                "Seconds between the two /metrics snapshots of a width.",
                ParamKind::Int { min: 1, max: 120 },
                ParamValue::Int(8),
            ),
            ParamSpec::new(
                "settle_s",
                "Settle",
                "Seconds to wait after every stream has its first token before the window \
                 opens.",
                ParamKind::Int { min: 0, max: 60 },
                ParamValue::Int(3),
            ),
            ParamSpec::new(
                "request_timeout_s",
                "Request timeout",
                "Seconds before a single stream, or a /metrics read, is abandoned. \
                 Transport-side only.",
                ParamKind::Int { min: 30, max: 3600 },
                ParamValue::Int(300),
            ),
        ]
    }

    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        let specs = self.parameters();
        values.validate_against(&specs)?;
        let k = values.int("k")?;
        if k == UNSTATED_K {
            bail!(
                "k: state the draft depth the serve runs with --param k=N (0..={MAX_K}; 0 for a \
                 serve without --speculative); it has no default"
            );
        }
        self.k = u8::try_from(k).context("k is out of range")?;
        let widths = values
            .int_list("widths")?
            .iter()
            .map(|&w| usize::try_from(w).context("widths: a width is negative"))
            .collect::<Result<Vec<_>>>()?;
        if widths.windows(2).any(|p| p[0] >= p[1]) {
            bail!("widths: {widths:?} is not strictly ascending");
        }
        self.widths = widths;
        self.osl = values.usize("osl")?;
        self.window = Duration::from_secs(values.usize("window_s")? as u64);
        self.settle = Duration::from_secs(values.usize("settle_s")? as u64);
        self.timeout = Duration::from_secs(values.usize("request_timeout_s")? as u64);
        self.probed = false;
        self.first_scrape = None;
        self.rows.clear();
        self.energy = EnergyMeter::default();
        Ok(())
    }

    async fn next(&mut self) -> Result<BenchmarkResult> {
        let handle = self.handle()?.clone();
        handle.check_cancelled()?;
        let total = self.widths.len() as u64;

        if !self.probed {
            self.probed = true;
            http::probe(handle.target(), Duration::from_secs(10))
                .await
                .context("endpoint probe failed — check the target URL and port")?;
            self.first_scrape = Some(wave::scrape_metrics(handle.target(), self.timeout).await?);
            // 2026-10-04: Nothing is in flight yet, so the idle baseline is taken now.
            for line in self.energy.start(handle.target()).await {
                handle.log(line.level, line.text);
            }
            return Ok(BenchmarkResult::running("probe", self.elapsed())
                .with_progress(0, total)
                .log_line(LogLine::info(format!(
                    "{} · k {} · widths {:?} · ~{} prompt tokens · osl {} · settle {}s · \
                     window {}s",
                    handle.target().base_url,
                    self.k,
                    self.widths,
                    wave::PROMPT_TOKENS,
                    self.osl,
                    self.settle.as_secs(),
                    self.window.as_secs(),
                ))));
        }

        let Some(&n) = self.widths.get(self.rows.len()) else {
            if let Some(line) = self.energy.stop().await {
                handle.log(line.level, line.text);
            }
            let last = wave::scrape_metrics(handle.target(), self.timeout).await?;
            let first = self
                .first_scrape
                .take()
                .context("no scrape before the first width")?;
            let mut acceptance = BTreeMap::new();
            acceptance::record(&first, &last, &mut acceptance)?;
            return Ok(self.finish(acceptance));
        };
        handle.status(format!("width {n}: {n} concurrent streams"));
        let outcome = self.measure_width(n).await?;
        let mut line = match &outcome.verdict {
            CellVerdict::Vacuous(why) => LogLine::warn(format!("n {n}: vacuous — {why}")),
            CellVerdict::Measured(c) => LogLine::info(format!(
                "n {n}: {:.0} steps · step {:.2} ms (verify {:.2}, draft {:.2}) · wall {:.2} ms \
                 · {:.2} tok/step",
                c.steps, c.step_ms, c.verify_ms, c.draft_ms, c.wall_ms, c.tok_per_step
            )),
        };
        for failure in &outcome.failures {
            line.text.push_str(&format!(" · stream failed: {failure}"));
        }
        self.rows.push((n, outcome.verdict));
        let done = self.rows.len() as u64;
        handle.progress(done, total);
        Ok(BenchmarkResult::running("timed", self.elapsed())
            .with_progress(done, total)
            .with_table(self.table())
            .log_line(line))
    }
}

#[cfg(test)]
#[path = "spec_cost_tests.rs"]
mod tests;
