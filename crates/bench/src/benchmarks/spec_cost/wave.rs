// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: One width's measurement on the live serve: `n` concurrent streams, the
//! window between two `/metrics` snapshots once every stream is decoding, then abort and
//! drain until the serve is idle.
//!
//! Owner: bench, spec-cost.
//! Invariants:
//! - The window opens only after every stream has delivered a token, plus `settle`.
//! - No width starts before the previous width's streams are gone and the serve's decoded
//!   token counter has stopped moving.

use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde_json::json;
use tokio::task::JoinHandle;

use super::SpecCost;
use super::cell::{self, CellVerdict, Counters, Window};
use super::prom::Scrape;
use crate::benchmarks::concurrency::ESSAY_TASK;
use crate::benchmarks::stats::{self, PromptMode};
use crate::http::{self, ChatOutcome};
use crate::plugin::TargetEndpoint;

/// 2026-10-04: The serve's Prometheus endpoint.
pub(super) const METRICS_PATH: &str = "/metrics";
/// 2026-10-04: Approximate prompt length of every stream (`stats::make_prompt` filler before
/// the essay ask).
pub(super) const PROMPT_TOKENS: usize = 512;
/// 2026-10-04: Interval between the drain's counter reads; two equal reads in a row mean idle.
const DRAIN_POLL: Duration = Duration::from_millis(500);
/// 2026-10-04: The longest the drain waits for the serve to stop decoding.
const DRAIN_LIMIT: Duration = Duration::from_secs(120);

/// 2026-10-04: One width's outcome: the cell, plus the errors of streams that failed.
pub(super) struct WidthOutcome {
    pub(super) verdict: CellVerdict,
    pub(super) failures: Vec<String>,
}

impl SpecCost {
    /// 2026-10-04: The essay fixture of the concurrency sweep, with presence and frequency
    /// penalties pinned to 0 as that fixture pins them, and thinking off.
    pub(super) fn request_body(&self, model: &str, prefix_tag: &str) -> serde_json::Value {
        let mut prompt = stats::make_prompt(PROMPT_TOKENS, PromptMode::Natural, prefix_tag);
        prompt.push_str(ESSAY_TASK);
        json!({
            "model": model,
            "stream": true,
            "max_tokens": self.osl,
            "temperature": 0.0,
            "seed": 0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "reasoning_effort": "none",
            "messages": [{"role": "user", "content": prompt}],
        })
    }

    async fn snapshot(&self, target: &TargetEndpoint) -> Result<Counters> {
        let text = http::get_text(target, METRICS_PATH, self.timeout).await?;
        Counters::read(&Scrape::parse(&text)?)
    }

    pub(super) async fn measure_width(&self, n: usize) -> Result<WidthOutcome> {
        let handle = self.handle()?;
        let target = handle.target().clone();
        let mut tasks: Vec<JoinHandle<Result<ChatOutcome>>> = Vec::with_capacity(n);
        let mut firsts = Vec::with_capacity(n);
        for i in 0..n {
            // 2026-10-04: A tag unique to this run, width and stream: no prefix-cache hits.
            let tag = crate::benchmarks::unique_prefix_tag(
                &format!("spec-cost-n{n}-{i}"),
                handle.run_id(),
            );
            let body = self.request_body(&target.model, &tag);
            let (signal, first) = tokio::sync::oneshot::channel();
            let (target, timeout) = (target.clone(), self.timeout);
            tasks.push(tokio::spawn(async move {
                http::chat_stream_signalled(&target, &body, timeout, Some(signal)).await
            }));
            firsts.push(first);
        }
        let measured = self.window(&target, n, &tasks, firsts).await;
        let failures = stop_streams(tasks).await;
        self.drain(&target).await?;
        Ok(WidthOutcome {
            verdict: measured?,
            failures,
        })
    }

    /// 2026-10-04: Wait for every first token, settle, then take the two snapshots.
    async fn window(
        &self,
        target: &TargetEndpoint,
        n: usize,
        tasks: &[JoinHandle<Result<ChatOutcome>>],
        firsts: Vec<tokio::sync::oneshot::Receiver<Instant>>,
    ) -> Result<CellVerdict> {
        for first in firsts {
            // 2026-10-04: A dropped sender means the stream ended before its first token;
            // every stream carries its own request timeout, so this wait is bounded.
            if first.await.is_err() {
                return Ok(CellVerdict::Vacuous(
                    "a stream ended before its first token".to_string(),
                ));
            }
        }
        tokio::time::sleep(self.settle).await;
        let t0 = Instant::now();
        let before = self.snapshot(target).await?;
        cell::require_step_series(self.k, &before)?;
        tokio::time::sleep(self.window.saturating_sub(t0.elapsed())).await;
        let t1 = Instant::now();
        let after = self.snapshot(target).await?;
        let ended_early = tasks.iter().filter(|t| t.is_finished()).count();
        Ok(cell::evaluate(&Window {
            n,
            k: self.k,
            before,
            after,
            window_s: t1.duration_since(t0).as_secs_f64(),
            smi_energy_j: self.energy.window(t0, t1).map(|w| w.energy_j),
            ended_early,
        }))
    }

    /// 2026-10-04: Poll the decoded-token counter until two reads `DRAIN_POLL` apart agree.
    async fn drain(&self, target: &TargetEndpoint) -> Result<()> {
        let started = Instant::now();
        let mut last = self.snapshot(target).await?.tokens;
        loop {
            tokio::time::sleep(DRAIN_POLL).await;
            let now = self.snapshot(target).await?.tokens;
            if now == last {
                return Ok(());
            }
            if started.elapsed() > DRAIN_LIMIT {
                bail!(
                    "the serve was still decoding {}s after this width's streams were closed",
                    DRAIN_LIMIT.as_secs()
                );
            }
            last = now;
        }
    }
}

/// 2026-10-04: Abort every stream still running (closing its socket, which ends the request
/// on the serve) and return the errors of the streams that failed on their own.
async fn stop_streams(tasks: Vec<JoinHandle<Result<ChatOutcome>>>) -> Vec<String> {
    for task in &tasks {
        task.abort();
    }
    let mut failures = Vec::new();
    for task in tasks {
        match task.await {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => failures.push(format!("{e:#}")),
            Err(e) if e.is_cancelled() => {}
            Err(e) => failures.push(format!("stream task: {e}")),
        }
    }
    failures
}

/// 2026-10-04: One `/metrics` page whose counters read, so the serve is a usable target. The
/// probe takes one before the first width and the run another after the last, for the
/// whole-run acceptance counts (`acceptance.rs`).
pub(super) async fn scrape_metrics(target: &TargetEndpoint, timeout: Duration) -> Result<Scrape> {
    let text = http::get_text(target, METRICS_PATH, timeout)
        .await
        .with_context(|| format!("the serve must answer GET {METRICS_PATH}"))?;
    let scrape = Scrape::parse(&text)?;
    Counters::read(&scrape)?;
    Ok(scrape)
}
