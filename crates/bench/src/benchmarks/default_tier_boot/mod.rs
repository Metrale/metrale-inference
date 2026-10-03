// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Default-Tier Boot: a boot-only check of the serve's memory plan. Once the
//! serve is up it sends one short warmup wave, then reads `GET /memory` and reports the KV
//! pool's block count and whether the serve's device footprint stays inside its
//! `--gpu-memory-utilization` budget.
//!
//! Every other gate pins a reduced-precision serve; this one exists for the serve's default
//! precision tier, whose KV pool can shrink without any of them noticing. It lands
//! informational: the metrics are recorded and the verdict is `info` until bounds are declared
//! for it in BENCH.toml.
//!
//! The pins (the benchmark's definition, not parameters):
//! - Warmup: `WARMUP_REQUESTS` concurrent chat requests, each a distinct prompt of about
//!   `WARMUP_PROMPT_TOKENS` tokens with `max_tokens` = `WARMUP_OUTPUT_TOKENS`, temperature 0,
//!   seed 0. It brings the serve to the state a first real wave leaves it in.
//! - Read: one `GET /memory` after the wave has finished.
//!
//! Owner: bench, default-tier-boot.
//! Invariants: the run verdict is never a pass (`memory::verdict_for`).

use crate::hardware::Sensitivity;
use std::collections::BTreeMap;
use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde_json::json;

use crate::benchmark::{Benchmark, BenchmarkDescriptor, ModelExpectation};
use crate::benchmarks::stats::{self, PromptMode};
use crate::http;
use crate::metadata::PluginMetadata;
use crate::params::{ParamKind, ParamSpec, ParamValue, ParamValues};
use crate::plugin::{Plugin, PluginHandle};
use crate::result::{BenchmarkResult, CellStyle, LogLine, RunStatus, Stat};

pub mod memory;
pub use memory::{Footprint, MemoryReport, footprint, verdict_for};

/// 2026-10-01: Concurrent warmup requests.
const WARMUP_REQUESTS: usize = 8;
/// 2026-10-01: Approximate prompt length of each warmup request (`stats::make_prompt`).
const WARMUP_PROMPT_TOKENS: usize = 128;
/// 2026-10-01: `max_tokens` of each warmup request.
const WARMUP_OUTPUT_TOKENS: usize = 128;
/// 2026-10-01: The server's memory endpoint.
pub const MEMORY_PATH: &str = "/memory";

const SUMMARY: &str = "Boot check of the default precision tier: KV blocks and device \
                       footprint against the memory budget after one warmup wave";
pub const METADATA: PluginMetadata = PluginMetadata::metrale(SUMMARY);

pub const DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "default-tier-boot",
    name: "Default-Tier Boot Check",
    summary: SUMMARY,
    detail: "Boots the serve, sends one warmup wave of 8 concurrent chat requests (~128 prompt \
             tokens, 128 output tokens each, temperature 0, seed 0), then reads GET /memory. \
             Reports kv_blocks (the main KV pool), max_batch_size (the slots the model was \
             built with), budget_mib (total device memory x --gpu-memory-utilization), \
             device_footprint_mib (host MemAvailable at process start minus now, minus the \
             server's RssAnon: on unified memory, what the serve holds on the device) and \
             footprint_over_budget_mib (negative = within budget). Informational: no bounds \
             are declared yet, so the verdict is info. Needs a server that answers /memory \
             with host readings (Linux); a null reading makes the run INCONCLUSIVE.",
    duration_hint: "~2 min",
    expected_secs: 120,
    updated: "2026-10-01",
    intended_for: Some(ModelExpectation {
        families: &["qwen3.8-27b"],
        note: "The check is declared for unsloth/Qwen3.8-27B-NVFP4 served at its default \
               precision tier. Other checkpoints report their own memory plan, with nothing \
               to compare it against.",
    }),
    threshold_params: &[],
    needs_confirmation: false,
    // 2026-10-01: Block counts and byte figures; a throttled box reports the same ones.
    sensitivity: Sensitivity::Correctness,
    ctor: || Box::new(DefaultTierBoot::default()),
};

#[derive(Default)]
pub struct DefaultTierBoot {
    handle: Option<PluginHandle>,
    timeout: Duration,
    started: Option<Instant>,
    probed: bool,
    warmed: bool,
}

impl DefaultTierBoot {
    fn handle(&self) -> Result<&PluginHandle> {
        self.handle.as_ref().context("benchmark was not loaded")
    }

    fn elapsed(&self) -> Duration {
        self.started.map(|s| s.elapsed()).unwrap_or_default()
    }

    fn request_body(model: &str, prompt: String) -> serde_json::Value {
        json!({
            "model": model,
            "stream": true,
            "temperature": 0.0,
            "seed": 0,
            "max_tokens": WARMUP_OUTPUT_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        })
    }

    /// 2026-10-01: The warmup wave, all requests in flight at once. Each prompt carries its own
    /// tag, so no two share a cached prefix. Any failed request fails the run: a serve that
    /// cannot answer a small wave has no memory plan worth reporting.
    async fn warmup(&self) -> Result<usize> {
        let handle = self.handle()?;
        let target = handle.target();
        let bodies: Vec<serde_json::Value> = (0..WARMUP_REQUESTS)
            .map(|i| {
                let tag = super::unique_prefix_tag(&format!("boot-{i}"), handle.run_id());
                let prompt = stats::make_prompt(WARMUP_PROMPT_TOKENS, PromptMode::Count, &tag);
                Self::request_body(&target.model, prompt)
            })
            .collect();
        let outcomes = futures::future::join_all(
            bodies
                .iter()
                .map(|body| http::chat_stream(target, body, self.timeout)),
        )
        .await;
        let mut tokens = 0;
        for (i, outcome) in outcomes.into_iter().enumerate() {
            let outcome = outcome.with_context(|| format!("warmup request {}", i + 1))?;
            tokens += outcome.completion_tokens;
        }
        Ok(tokens)
    }
}

impl Plugin for DefaultTierBoot {
    fn metadata(&self) -> &'static PluginMetadata {
        &METADATA
    }

    fn load(&mut self, handle: PluginHandle) -> impl Future<Output = Result<()>> + Send {
        self.handle = Some(handle);
        self.started = Some(Instant::now());
        async { Ok(()) }
    }
}

impl Benchmark for DefaultTierBoot {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        &DESCRIPTOR
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        vec![ParamSpec::new(
            "request_timeout_s",
            "Request timeout",
            "Seconds before a single warmup request, or the /memory read, is abandoned. \
             Transport-side only.",
            ParamKind::Int { min: 30, max: 3600 },
            ParamValue::Int(300),
        )]
    }

    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        let specs = self.parameters();
        values.validate_against(&specs)?;
        self.timeout = Duration::from_secs(values.usize("request_timeout_s")? as u64);
        self.probed = false;
        self.warmed = false;
        Ok(())
    }

    async fn next(&mut self) -> Result<BenchmarkResult> {
        let handle = self.handle()?.clone();
        handle.check_cancelled()?;

        if !self.probed {
            self.probed = true;
            http::probe(handle.target(), Duration::from_secs(10))
                .await
                .context("endpoint probe failed — check the target URL and port")?;
            return Ok(BenchmarkResult::running("probe", self.elapsed())
                .with_progress(0, 2)
                .log_line(LogLine::info(format!(
                    "{} · warmup {WARMUP_REQUESTS} x (~{WARMUP_PROMPT_TOKENS} prompt, \
                     {WARMUP_OUTPUT_TOKENS} output) tokens · then GET {MEMORY_PATH}",
                    handle.target().base_url
                ))));
        }

        if !self.warmed {
            self.warmed = true;
            handle.status(format!("warmup: {WARMUP_REQUESTS} concurrent requests"));
            let tokens = self.warmup().await?;
            if tokens == 0 {
                bail!("the warmup wave produced no output token — the serve is not decoding");
            }
            handle.progress(1, 2);
            return Ok(BenchmarkResult::running("warmup", self.elapsed())
                .with_progress(1, 2)
                .log_line(LogLine::info(format!(
                    "warmup: {WARMUP_REQUESTS} requests, {tokens} output tokens"
                ))));
        }

        let doc = http::get_json(handle.target(), MEMORY_PATH, self.timeout)
            .await
            .with_context(|| format!("reading {MEMORY_PATH}"))?;
        let report: MemoryReport = serde_json::from_value(doc)
            .with_context(|| format!("parsing {MEMORY_PATH}: not a MemoryReport"))?;
        let derived = footprint(&report);
        let verdict = verdict_for(&derived);
        let mut metrics = BTreeMap::new();
        let summary = match &derived {
            Err(_) => Vec::new(),
            Ok(f) => {
                f.metrics(&mut metrics);
                vec![
                    Stat::new("KV blocks", f.kv_blocks.to_string(), "")
                        .with_style(CellStyle::Accent),
                    Stat::new("Slots", f.max_batch_size.to_string(), ""),
                    Stat::new(
                        "Device footprint",
                        format!("{:.0}", f.device_footprint_mib),
                        "MiB",
                    ),
                    Stat::new("Budget", format!("{:.0}", f.budget_mib), "MiB"),
                    Stat::new(
                        "Over budget",
                        format!("{:+.0}", f.footprint_over_budget_mib),
                        "MiB",
                    ),
                ]
            }
        };
        handle.progress(2, 2);
        Ok(BenchmarkResult {
            status: RunStatus::Completed,
            ..BenchmarkResult::running("done", self.elapsed())
        }
        .with_progress(2, 2)
        .with_summary(summary)
        .with_metrics(metrics)
        .with_verdict(verdict)
        .log_line(LogLine::info(format!(
            "{MEMORY_PATH}: {} blocks x {} bytes, ledger {}",
            report.kv_blocks,
            report.kv_block_bytes,
            report
                .ledger_live_bytes
                .map(|b| format!("{b} bytes"))
                .unwrap_or_else(|| "null".into()),
        ))))
    }
}

#[cfg(test)]
#[path = "default_tier_boot_tests.rs"]
mod tests;
