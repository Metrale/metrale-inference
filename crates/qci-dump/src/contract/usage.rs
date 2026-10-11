// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: `InferenceUsage` v1, the runtime's own record of one engine request
//! (investor-mvp #28, #37, #49). It joins the agent's usage record by `request_id` and the
//! request's trace by `trace_id`. It carries counts, timings and ids, never prompt or
//! response text, and no tenant or project (the runtime never learns them).
//!
//! Timings are anchored at the moment the HTTP layer received the request:
//! - `queue_ms`: received → the scheduler's first prefill of the request (measured).
//! - `ttft_ms`: received → first output token: `queue + scheduler ttft` (derived).
//! - `total_ms`: received → the scheduler's finish frame: `queue + ttft + decode` (derived;
//!   response serialization after the finish frame is not in it).
//! - `tpot_us`: `decode / (completion − 1)`; absent below two output tokens (derived).
//!
//! Owner: qci-dump (contract binding).
//! Invariants: a record never holds request or response text.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::PLATFORM_VERSION;
use super::field::{AbsentReason, Field};
use super::ids::{GitCommit, RequestId, Sha256Digest, TraceId};
use super::rules::{self, Violation, require};

/// 2026-10-11: Engine failure classes. Closed; never an engine's own string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineErrorKind {
    BadRequest,
    ContextTooLong,
    ModelNotLoaded,
    Overloaded,
    Unavailable,
    Internal,
}

impl EngineErrorKind {
    /// 2026-10-11: The class of an HTTP error status the runtime answered with after the
    /// scheduler accepted the request.
    pub fn of_status(status: u16) -> Self {
        match status {
            429 => Self::Overloaded,
            503 => Self::Unavailable,
            400..=499 => Self::BadRequest,
            _ => Self::Internal,
        }
    }
}

/// 2026-10-11: How a request the scheduler accepted ended, as the runtime sees it.
/// `channel_lost` and `unknown` belong to the agent and are not runtime outcomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Outcome {
    Completed,
    Cancelled,
    EngineError { error: EngineErrorKind },
    Timeout,
}

impl Outcome {
    /// 2026-10-11: The outcome of a finish frame: the server-side deadline reason
    /// (`timeout`) is `Timeout`; every other finish reason completed the request.
    pub fn of_finish_reason(reason: &str, timeout_reason: &str) -> Self {
        if reason == timeout_reason {
            Self::Timeout
        } else {
            Self::Completed
        }
    }

    /// 2026-10-11: A bounded label for counters: one of four words.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
            Self::EngineError { .. } => "engine_error",
            Self::Timeout => "timeout",
        }
    }
}

/// 2026-10-11: Token counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tokens {
    pub prompt: Field<u32>,
    pub completion: Field<u32>,
    /// 2026-10-11: Prompt tokens served from the prefix cache.
    pub cached_prompt: Field<u32>,
    /// 2026-10-11: Output tokens inside the reasoning block; `unsupported` for a model
    /// with no reasoning channel.
    pub reasoning: Field<u32>,
}

impl Tokens {
    /// 2026-10-11: Counts of a request the scheduler finished. `reasoning` is `None` when
    /// the served model has no reasoning channel.
    pub fn finished(
        prompt: usize,
        completion: usize,
        cached_prompt: usize,
        reasoning: Option<usize>,
    ) -> Self {
        Self {
            prompt: Field::count(prompt),
            completion: Field::count(completion),
            cached_prompt: Field::count(cached_prompt),
            reasoning: reasoning.map_or(Field::Absent(AbsentReason::Unsupported), Field::count),
        }
    }

    /// 2026-10-11: Counts of a request that ended without a finish frame: only the prompt
    /// length, known at dispatch, survives.
    pub fn unfinished(prompt: usize) -> Self {
        let lost = Field::Absent(AbsentReason::Unavailable);
        Self {
            prompt: Field::count(prompt),
            completion: lost,
            cached_prompt: lost,
            reasoning: lost,
        }
    }
}

/// 2026-10-11: Which rail an energy figure covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnergyRail {
    GpuRail,
    AcceleratorRail,
    Board,
}

/// 2026-10-11: Energy attributed to one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Energy {
    pub millijoules: Field<u64>,
    pub rail: EnergyRail,
    pub window_mono_ns: Field<(u64, u64)>,
}

impl Energy {
    /// 2026-10-11: The runtime does not attribute energy to one request on its response
    /// path. The schema requires a rail even when the figure is absent; `gpu_rail` names
    /// the rail the device layer would read.
    pub fn not_attributed() -> Self {
        Self {
            millijoules: Field::Absent(AbsentReason::Unsupported),
            rail: EnergyRail::GpuRail,
            window_mono_ns: Field::Absent(AbsentReason::Unsupported),
        }
    }
}

/// 2026-10-11: Timings on the runtime's clock, anchored at receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeTimings {
    pub received_at_unix_ms: i64,
    pub queue_ms: Field<u32>,
    pub ttft_ms: Field<u32>,
    pub total_ms: Field<u32>,
    pub tpot_us: Field<u32>,
}

/// 2026-10-11: What the scheduler's finish frame says about one request, plus the
/// receipt-to-first-prefill wait measured against the HTTP receive instant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchedulerSpan {
    /// 2026-10-11: Received → first scheduled prefill. `None` when the prefill instant
    /// precedes the receive instant (two clocks disagreeing), which is not a measurement.
    pub queue: Option<Duration>,
    /// 2026-10-11: First prefill → first output token, scheduler clock, milliseconds.
    pub ttft_ms: f64,
    /// 2026-10-11: First output token → finish frame, scheduler clock, milliseconds.
    pub decode_ms: f64,
}

fn whole(ms: f64) -> Option<u32> {
    (ms.is_finite() && ms >= 0.0 && ms <= f64::from(u32::MAX)).then(|| ms.round() as u32)
}

impl RuntimeTimings {
    /// 2026-10-11: Timings of a request with a finish frame (`span`) that produced
    /// `completion` output tokens, or of one without (`span` = `None`).
    pub fn derive(
        received_at_unix_ms: i64,
        span: Option<SchedulerSpan>,
        completion: usize,
    ) -> Self {
        let unavailable = Field::Absent(AbsentReason::Unavailable);
        let Some(span) = span else {
            return Self {
                received_at_unix_ms,
                queue_ms: unavailable,
                ttft_ms: unavailable,
                total_ms: unavailable,
                tpot_us: unavailable,
            };
        };
        let queue_ms = span.queue.map(|q| q.as_secs_f64() * 1000.0);
        let field = |ms: Option<f64>, measured: bool| match ms.and_then(whole) {
            Some(v) if measured => Field::Measured(v),
            Some(v) => Field::Derived(v),
            None => unavailable,
        };
        let tpot_us = if completion >= 2 {
            field(
                Some(span.decode_ms * 1000.0 / (completion - 1) as f64),
                false,
            )
        } else {
            Field::Absent(AbsentReason::NotApplicable)
        };
        Self {
            received_at_unix_ms,
            queue_ms: field(queue_ms, true),
            ttft_ms: field(queue_ms.map(|q| q + span.ttft_ms), false),
            total_ms: field(queue_ms.map(|q| q + span.ttft_ms + span.decode_ms), false),
            tpot_us,
        }
    }
}

/// 2026-10-11: One engine request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceUsage {
    pub version: u32,
    pub request_id: RequestId,
    pub trace_id: TraceId,
    pub served_model: String,
    /// 2026-10-11: The recipe digest the runtime was launched from; `null` when none was
    /// given. Always written, and a reader refuses it missing.
    #[serde(deserialize_with = "rules::present")]
    pub recipe: Option<Sha256Digest>,
    pub runtime_commit: GitCommit,
    pub outcome: Outcome,
    pub tokens: Tokens,
    pub timings: RuntimeTimings,
    pub energy: Energy,
    pub finished_at_unix_ms: i64,
}

/// 2026-10-11: The process-wide part of every record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeIdentity {
    pub runtime_commit: GitCommit,
    pub recipe: Option<Sha256Digest>,
}

impl InferenceUsage {
    /// 2026-10-11: The schema rules serde cannot express.
    pub fn validate(&self) -> Result<(), Violation> {
        rules::version(self.version)?;
        require(!self.trace_id.is_zero(), "/trace_id")?;
        require(rules::text(&self.served_model), "/served_model")?;
        require(
            rules::timestamp(self.timings.received_at_unix_ms),
            "/timings/received_at_unix_ms",
        )?;
        require(
            rules::timestamp(self.finished_at_unix_ms),
            "/finished_at_unix_ms",
        )
    }

    /// 2026-10-11: Assemble one record. Pure: every clock reading is an argument.
    pub fn new(
        runtime: RuntimeIdentity,
        ids: (RequestId, TraceId),
        served_model: &str,
        outcome: Outcome,
        tokens: Tokens,
        timings: RuntimeTimings,
        finished_at_unix_ms: i64,
    ) -> Self {
        Self {
            version: PLATFORM_VERSION,
            request_id: ids.0,
            trace_id: ids.1,
            served_model: served_model.to_string(),
            recipe: runtime.recipe,
            runtime_commit: runtime.runtime_commit,
            outcome,
            tokens,
            timings,
            energy: Energy::not_attributed(),
            finished_at_unix_ms,
        }
    }
}

#[cfg(test)]
#[path = "usage_tests.rs"]
mod tests;
