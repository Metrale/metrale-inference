// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The embedder port and the pure half of embedding (owner decision 6: OpenRouter,
//! model and dimension from config): the declared mode, the request and response codec of the
//! OpenAI-shaped `/embeddings` API, the batch plan over texts the cache lacks, and the retry loop
//! with exponential backoff on the Clock and Timer ports (rate-limited free tier).
//!
//! Owner: metrale-kg-derive.
//! Invariants:
//! - Two modes only, declared and reported: [`Mode::Available`] or [`Mode::Unavailable`] with its
//!   reason. A failing endpoint does not switch modes silently; the caller reports `graph_only`
//!   with the failure as the reason for that call.
//! - The key never passes through this crate: the host holds it.
//! - A text whose sha the cache holds is never sent again.

use crate::hash::Sha256;
use crate::prelude::*;
use crate::secrets::secret_pattern;
use metrale_abstract_io::{Clock, MonoNs, Timer};
use metrale_error::{ErrorCode, Result, error};
use serde_json::Value;

/// 2026-10-23: The embedder's declared mode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mode {
    Available { model: String, dim: u32 },
    Unavailable { reason: String },
}

/// 2026-10-23: One endpoint answer.
#[derive(Clone, Debug, PartialEq)]
pub enum Reply {
    /// 2026-10-23: One vector per input, in input order.
    Vectors(Vec<Vec<f32>>),
    /// 2026-10-23: HTTP 429 or a 5xx (the free tier answers 504 after a 60 s upstream timeout,
    /// probed 2026-10-23): wait and retry.
    Retryable {
        status: u16,
        retry_after_ms: Option<u64>,
    },
}

/// 2026-10-23: What an input is to an asymmetric embedder: a search query or a stored document.
/// The provider embeds the two differently (probed 2026-10-23: the same text as `query` and as
/// `passage` has cosine 0.66, and an untyped request is embedded as a query), so every request
/// names its role and the host maps it to the configured `input_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Query,
    Document,
}

/// 2026-10-23: The embedder port; hosts implement it (metrale-io-host-kg's OpenRouter host).
#[expect(
    async_fn_in_trait,
    reason = "2026-10-23: as metrale-abstract-io's ports: the host's executor drives the future"
)]
pub trait Embedder {
    /// 2026-10-23: The declared mode.
    fn mode(&self) -> Mode;
    /// 2026-10-23: Embed `inputs`, all of one `role` (one request).
    async fn embed(&self, role: Role, inputs: &[String]) -> Result<Reply>;
}

/// 2026-10-23: The OpenAI-shaped request body, with the asymmetric model's `input_type`.
pub fn request_body(model: &str, input_type: &str, inputs: &[String]) -> String {
    let mut o = serde_json::Map::new();
    o.insert("model".into(), Value::String(model.to_string()));
    o.insert("input_type".into(), Value::String(input_type.to_string()));
    o.insert("input".into(), Value::from(inputs.to_vec()));
    let mut s = String::new();
    crate::jsonl::canon(&Value::Object(o), &mut s);
    s
}

/// 2026-10-23: Parse a response body: `data[].embedding` reordered by `data[].index`, checked
/// against the request count and the declared dimension. Components narrow to `f32`, the
/// precision the provider computes in (the JSON carries f32 values printed as decimals).
#[allow(clippy::cast_possible_truncation)]
pub fn parse_response(body: &str, count: usize, dim: u32) -> Result<Vec<Vec<f32>>> {
    let bad = |why: &str| error!(ErrorCode::KgEmbedProtocol, why.to_string());
    let v: Value = serde_json::from_str(body).map_err(|e| bad(&e.to_string()))?;
    let data = v.get("data").and_then(Value::as_array).ok_or_else(|| {
        let msg = v
            .get("error")
            .map_or_else(|| "no `data` array".to_string(), Value::to_string);
        bad(&secret_pattern(&msg).map_or(msg.clone(), |p| format!("[dropped: {p}]")))
    })?;
    let mut out: Vec<Option<Vec<f32>>> = vec![None; count];
    for d in data {
        let i = d
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|i| usize::try_from(i).ok())
            .ok_or_else(|| bad("an item has no index"))?;
        let e = d
            .get("embedding")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("an item has no embedding"))?;
        let vec: Vec<f32> = e
            .iter()
            .filter_map(Value::as_f64)
            .map(|x| x as f32)
            .collect();
        if vec.len() != e.len() {
            return Err(bad("an embedding holds a non-number"));
        }
        if u32::try_from(vec.len()).ok() != Some(dim) {
            return Err(error!(ErrorCode::KgEmbedDim, vec.len(), dim));
        }
        let slot = out
            .get_mut(i)
            .ok_or_else(|| bad("an index is out of range"))?;
        *slot = Some(vec);
    }
    out.into_iter()
        .map(|v| v.ok_or_else(|| bad("an input has no embedding")))
        .collect()
}

/// 2026-10-23: How the pipeline batches and retries (all from config).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub batch: usize,
    pub max_attempts: u32,
    pub backoff_ms: u64,
    /// 2026-10-23: The longest input sent (chars); a longer one is refused unsent.
    pub max_input_chars: usize,
}

/// 2026-10-23: The wait before retry `attempt` (1-based): the server's `retry_after` when it sent
/// one, else `backoff_ms * 2^(attempt-1)`, capped at 64x.
pub fn backoff(plan: &Plan, attempt: u32, retry_after_ms: Option<u64>) -> u64 {
    retry_after_ms.unwrap_or_else(|| {
        plan.backoff_ms
            .saturating_mul(1u64 << attempt.saturating_sub(1).min(6))
    })
}

/// 2026-10-23: The texts `(sha, text)` whose sha `cached` lacks, deduplicated by sha, in sha order.
pub fn missing<'a>(
    texts: impl Iterator<Item = (Sha256, &'a str)>,
    cached: &BTreeSet<Sha256>,
) -> Vec<(Sha256, String)> {
    let mut m: BTreeMap<Sha256, String> = BTreeMap::new();
    for (sha, t) in texts {
        if !cached.contains(&sha) {
            m.entry(sha).or_insert_with(|| t.to_string());
        }
    }
    m.into_iter().collect()
}

/// 2026-10-23: What [`embed_all`] got: the vectors, the inputs the provider refused one by one
/// (a batch it refuses as a request is split until the refusal names single inputs; they are
/// skipped, never guessed at), and the error that stopped the run, if any.
#[derive(Debug, Default)]
pub struct Embedded {
    pub done: Vec<(Sha256, Vec<f32>)>,
    pub refused: Vec<(Sha256, String)>,
    pub stopped: Option<metrale_error::Error>,
}

/// 2026-10-23: Embed `todo` in batches, retrying rate-limited batches with backoff. A batch the
/// provider refuses (`KgEmbedProtocol`, e.g. HTTP 400) is halved and retried, so one bad input
/// costs O(log batch) requests and no other input; the cache keeps all progress.
pub async fn embed_all<E: Embedder, C: Clock, T: Timer>(
    e: &E,
    clock: &C,
    timer: &T,
    plan: &Plan,
    role: Role,
    todo: &[(Sha256, String)],
) -> Embedded {
    let mut out = Embedded::default();
    let mut fit: Vec<(Sha256, String)> = Vec::with_capacity(todo.len());
    for (sha, t) in todo {
        let n = t.chars().count();
        if n > plan.max_input_chars {
            out.refused.push((
                *sha,
                format!(
                    "{n} chars exceeds max_input_chars {} (not sent)",
                    plan.max_input_chars
                ),
            ));
        } else {
            fit.push((*sha, t.clone()));
        }
    }
    let mut work: Vec<&[(Sha256, String)]> = fit.chunks(plan.batch.max(1)).rev().collect();
    while let Some(chunk) = work.pop() {
        let inputs: Vec<String> = chunk.iter().map(|(_, t)| t.clone()).collect();
        let mut attempt = 0;
        loop {
            attempt += 1;
            match e.embed(role, &inputs).await {
                Ok(Reply::Vectors(v)) if v.len() == chunk.len() => {
                    out.done.extend(chunk.iter().map(|(s, _)| *s).zip(v));
                    break;
                }
                Ok(Reply::Vectors(_)) => {
                    out.stopped = Some(error!(
                        ErrorCode::KgEmbedProtocol,
                        "the reply count differs from the batch"
                    ));
                    return out;
                }
                Ok(Reply::Retryable { retry_after_ms, .. }) if attempt < plan.max_attempts => {
                    let wait = backoff(plan, attempt, retry_after_ms).saturating_mul(1_000_000);
                    timer
                        .sleep_until(MonoNs(clock.monotonic_ns().0.saturating_add(wait)))
                        .await;
                }
                Ok(Reply::Retryable { .. }) => {
                    out.stopped = Some(error!(ErrorCode::KgEmbedRateLimited, attempt));
                    return out;
                }
                Err(err) if err.code() == ErrorCode::KgEmbedProtocol && chunk.len() > 1 => {
                    let (a, b) = chunk.split_at(chunk.len() / 2);
                    work.push(b);
                    work.push(a);
                    break;
                }
                Err(err) if err.code() == ErrorCode::KgEmbedProtocol => {
                    out.refused.push((chunk[0].0, err.to_string()));
                    break;
                }
                Err(err) => {
                    out.stopped = Some(err);
                    return out;
                }
            }
        }
    }
    out
}

/// 2026-10-23: Cosine similarity (0 for a zero vector).
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        d / (libm::sqrt(na) * libm::sqrt(nb))
    }
}
