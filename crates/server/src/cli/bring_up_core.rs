// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The pure side of `met ml-utils model-bring-up-bench`: the run plan, model
//! resolution from `/v1/models`, engine identity, prefix-cache counters read from `/metrics`
//! text, the per-bench row and its cache verdict, and the record format. Rendering lives in
//! `bring_up_render`; every network and file access lives in `bring_up`.
//!
//! Owner: server CLI (`met ml-utils`).
//! Invariants:
//! - A bench's statistics are the TTFT gate's own metrics (`median_ms`, `p90_ms`, `min_ms`,
//!   `samples`, `prompt_tokens`); nothing here recomputes them.
//! - A cache verdict is `NotExposed` whenever the engine publishes no prefix-cache counter;
//!   it never reads as verified.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 2026-10-10: The record format; `--compare` refuses any other.
pub(crate) const SCHEMA: &str = "metrale-bring-up-bench/1";

/// 2026-10-10: The four single-stream TTFT benches, in the order they run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum Bench {
    Cold,
    Warm,
    HighIslCold,
    HighIslWarm,
}

impl Bench {
    /// 2026-10-10: The TTFT gate that measures it (`metrale_bench::benchmarks::ttft`).
    pub(crate) fn gate(self) -> &'static str {
        match self {
            Bench::Cold => "ttft-cold-gate",
            Bench::Warm => "ttft-warm-gate",
            Bench::HighIslCold => "high-isl-ttft-cold",
            Bench::HighIslWarm => "high-isl-ttft-warm",
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Bench::Cold => "cold",
            Bench::Warm => "warm",
            Bench::HighIslCold => "high-isl cold",
            Bench::HighIslWarm => "high-isl warm",
        }
    }

    pub(crate) fn is_warm(self) -> bool {
        matches!(self, Bench::Warm | Bench::HighIslWarm)
    }

    /// 2026-10-10: The gate parameter that carries the prompt size.
    pub(crate) fn size_param(self) -> &'static str {
        match self {
            Bench::Cold | Bench::Warm => "prompt_lengths",
            Bench::HighIslCold | Bench::HighIslWarm => "min_prompt_tokens",
        }
    }
}

/// 2026-10-10: One gate run: a bench at one exact prompt size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Step {
    pub bench: Bench,
    pub tokens: usize,
}

/// 2026-10-10: Cold then warm at each synthetic size, then high-ISL cold and warm, all
/// sequential at one stream.
pub(crate) fn plan(isl: &[usize], high_isl: usize) -> Vec<Step> {
    let synthetic = [Bench::Cold, Bench::Warm]
        .into_iter()
        .flat_map(|bench| isl.iter().map(move |&tokens| Step { bench, tokens }));
    synthetic
        .chain([Bench::HighIslCold, Bench::HighIslWarm].map(|bench| Step {
            bench,
            tokens: high_isl,
        }))
        .collect()
}

/// 2026-10-10: The served model ids `/v1/models` lists.
pub(crate) fn served_models(models: &Value) -> Vec<String> {
    models["data"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|m| m["id"].as_str().map(str::to_string))
        .collect()
}

/// 2026-10-10: The model to request: `explicit` when the endpoint serves it, else the one
/// model the endpoint serves. Several, or none, is an error naming what was listed.
pub(crate) fn resolve_model(models: &Value, explicit: Option<&str>) -> Result<String> {
    let ids = served_models(models);
    match (explicit, ids.as_slice()) {
        (Some(m), _) if ids.iter().any(|id| id == m) => Ok(m.to_string()),
        (Some(m), _) => bail!("--model {m}: the endpoint serves {ids:?}, not {m}"),
        (None, [one]) => Ok(one.clone()),
        (None, []) => bail!("the endpoint's /v1/models lists no model"),
        (None, many) => bail!("the endpoint serves {many:?}; pass --model to pick one"),
    }
}

/// 2026-10-10: Which engine answered, from its own reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum EngineKind {
    Metrale,
    Vllm,
    Unknown,
}

/// 2026-10-10: `owned_by` of the measured model decides; a Metrale serve also answers
/// `/serve-config` with its identity, which overrides an ambiguous `owned_by`.
pub(crate) fn engine_kind(owned_by: Option<&str>, serve_identity: bool) -> EngineKind {
    match owned_by {
        _ if serve_identity => EngineKind::Metrale,
        Some("metrale") => EngineKind::Metrale,
        Some("vllm") => EngineKind::Vllm,
        _ => EngineKind::Unknown,
    }
}

/// 2026-10-10: What the record says about the engine measured.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Engine {
    pub kind: EngineKind,
    /// 2026-10-10: `--label`, or the kind's name.
    pub label: String,
    pub owned_by: Option<String>,
    /// 2026-10-10: `GET /version` (vLLM).
    pub version: Option<String>,
    /// 2026-10-10: `max_model_len` of the measured model in `/v1/models`.
    pub max_model_len: Option<u64>,
    /// 2026-10-10: `GET /serve-config` (Metrale): binary, argv and lever digests and the pid.
    pub serve_identity: Option<Value>,
    /// 2026-10-10: The labels of vLLM's `vllm:cache_config_info` gauge (prefix caching,
    /// block size, cache dtype, ...): the resolved cache config the engine reports.
    pub cache_config: BTreeMap<String, String>,
}

/// 2026-10-10: The prefix-cache hit-token counters, in tokens, per engine.
const HIT_TOKEN_COUNTERS: &[&str] = &[
    "metrale_prefix_cache_hit_tokens_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_hits",
];

/// 2026-10-10: Every sample of a Prometheus metric family, summed over its label sets
/// (a multi-engine vLLM labels one series per engine).
fn sum_family(text: &str, name: &str) -> Option<f64> {
    let mut total = None;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let (series, value) = match line.rsplit_once(' ') {
            Some(p) => p,
            None => continue,
        };
        let family = series.split('{').next().unwrap_or(series).trim();
        if family == name
            && let Ok(v) = value.trim().parse::<f64>()
        {
            total = Some(total.unwrap_or(0.0) + v);
        }
    }
    total
}

/// 2026-10-10: The first prefix-cache hit-token counter `/metrics` exposes, and its value.
pub(crate) fn hit_tokens(metrics: &str) -> Option<(&'static str, f64)> {
    HIT_TOKEN_COUNTERS
        .iter()
        .find_map(|name| sum_family(metrics, name).map(|v| (*name, v)))
}

/// 2026-10-10: The labels of `vllm:cache_config_info` (empty when absent).
pub(crate) fn cache_config(metrics: &str) -> BTreeMap<String, String> {
    let Some(line) = metrics
        .lines()
        .find(|l| l.starts_with("vllm:cache_config_info{"))
    else {
        return BTreeMap::new();
    };
    let inner = line
        .split_once('{')
        .and_then(|(_, r)| r.rsplit_once('}'))
        .map_or("", |(l, _)| l);
    let mut out = BTreeMap::new();
    let mut rest = inner;
    while let Some((key, after)) = rest.split_once("=\"") {
        let Some((value, tail)) = after.split_once('"') else {
            break;
        };
        out.insert(
            key.trim_start_matches(',').trim().to_string(),
            value.to_string(),
        );
        rest = tail;
    }
    out
}

/// 2026-10-10: Whether the engine's own counters show the cache behaving as the bench needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum CacheVerdict {
    /// 2026-10-10: Warm: the hits cover at least `WARM_COVER` of every measured prompt.
    /// Cold: at most `COLD_ALLOWANCE` tokens per request hit (the chat-template header).
    Verified,
    Violated,
    /// 2026-10-10: The engine publishes no prefix-cache counter.
    NotExposed,
}

/// 2026-10-10: The share of each measured warm prompt the hits must cover. A cache reuses
/// whole blocks, so the partial last block of a prompt is recomputed.
pub(crate) const WARM_COVER: f64 = 0.75;
/// 2026-10-10: Hit tokens per cold request still counted as cold: one 64-token block, the
/// largest of the engines' block sizes here, which the chat template's shared header can fill.
pub(crate) const COLD_ALLOWANCE: f64 = 64.0;

/// 2026-10-10: Judge the hit-token delta over one gate run. `requests` counts every request
/// the run sent (warm-up, primes and samples).
pub(crate) fn cache_verdict(
    bench: Bench,
    delta: Option<f64>,
    samples: usize,
    requests: usize,
    tokens: usize,
) -> CacheVerdict {
    let Some(delta) = delta else {
        return CacheVerdict::NotExposed;
    };
    let ok = if bench.is_warm() {
        delta >= samples as f64 * tokens as f64 * WARM_COVER
    } else {
        delta <= requests as f64 * COLD_ALLOWANCE
    };
    if ok {
        CacheVerdict::Verified
    } else {
        CacheVerdict::Violated
    }
}

/// 2026-10-10: The requests one gate run sends: the warm-up, then one per sample, two
/// (prime and measure) in a warm bench.
pub(crate) fn requests_of(bench: Bench, samples: usize) -> usize {
    1 + samples * if bench.is_warm() { 2 } else { 1 }
}

/// 2026-10-10: The prefix-cache evidence of one row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct CacheCheck {
    pub counter: Option<String>,
    pub hit_tokens: Option<f64>,
    pub verdict: CacheVerdict,
}

/// 2026-10-10: One bench at one prompt size.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Row {
    pub bench: Bench,
    pub gate: String,
    /// 2026-10-10: The exact user-message size, in tokens of the target tokenizer.
    pub tokens: usize,
    /// 2026-10-10: The smallest `usage.prompt_tokens` the server reported (template included).
    pub prompt_tokens: Option<f64>,
    pub reps: usize,
    pub samples: usize,
    pub p50_ms: Option<f64>,
    pub p90_ms: Option<f64>,
    pub min_ms: Option<f64>,
    /// 2026-10-10: The most `usage.prompt_tokens_details.cached_tokens` any sample reported.
    pub cached_prompt_tokens: Option<f64>,
    pub cache: CacheCheck,
    /// 2026-10-10: `completed`, or why the run did not complete.
    pub status: String,
    /// 2026-10-10: The gate's full run record (parameters, log, hardware state).
    pub run: Value,
}

/// 2026-10-10: A row from the gate's terminal metrics and the counter readings around it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn row(
    step: &Step,
    reps: usize,
    metrics: &BTreeMap<String, f64>,
    status: String,
    before: Option<(&'static str, f64)>,
    after: Option<(&'static str, f64)>,
    run: Value,
) -> Row {
    let samples = metrics.get("samples").map_or(0, |v| *v as usize);
    let delta = match (before, after) {
        (Some((a, x)), Some((b, y))) if a == b => Some(y - x),
        _ => None,
    };
    let requests = requests_of(step.bench, samples);
    Row {
        bench: step.bench,
        gate: step.bench.gate().to_string(),
        tokens: step.tokens,
        prompt_tokens: metrics.get("prompt_tokens").copied(),
        reps,
        samples,
        p50_ms: metrics.get("median_ms").copied(),
        p90_ms: metrics.get("p90_ms").copied(),
        min_ms: metrics.get("min_ms").copied(),
        cached_prompt_tokens: metrics.get("cached_prompt_tokens").copied(),
        cache: CacheCheck {
            counter: after.map(|(n, _)| n.to_string()),
            hit_tokens: delta,
            verdict: cache_verdict(step.bench, delta, samples, requests, step.tokens),
        },
        status,
        run,
    }
}

/// 2026-10-10: One `model-bring-up-bench` run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Record {
    pub schema: String,
    pub url: String,
    pub model: String,
    /// 2026-10-10: The tokenizer.json the prompts were cut with, and its SHA-256 (`None` with
    /// `--skip-ttft`, which cuts no prompt).
    pub tokenizer: Option<String>,
    pub tokenizer_sha256: Option<String>,
    pub engine: Engine,
    /// 2026-10-10: Unix seconds when the first bench started.
    pub started_at: u64,
    pub metrale_version: String,
    /// 2026-10-10: The TTFT benches, in run order (empty with `--skip-ttft`).
    pub ttft: Vec<Row>,
    /// 2026-10-10: The concurrency ladder (`None` with `--skip-concurrency`).
    pub concurrency: Option<super::bring_up_conc::Section>,
}

impl Record {
    /// 2026-10-10: Parse a record, refusing another schema.
    pub(crate) fn parse(text: &str) -> Result<Self> {
        let r: Record = serde_json::from_str(text)?;
        if r.schema != SCHEMA {
            bail!("schema {:?}, expected {SCHEMA:?}", r.schema);
        }
        Ok(r)
    }
}

#[cfg(test)]
#[path = "bring_up_core_tests.rs"]
mod tests;
