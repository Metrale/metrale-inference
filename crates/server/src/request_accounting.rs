// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The one place an inference request the scheduler accepted is accounted
//! (investor-mvp #28, #37, #49): the token and latency counters of `/metrics`, the outcome
//! counter, and one `InferenceUsage` record per request, logged under
//! [`USAGE_LOG_TARGET`] and returned for the response's `usage`.
//!
//! A [`Pending`] is made before dispatch and marked [`Pending::dispatched`] once the
//! scheduler has the request. It is settled exactly once: [`Pending::complete`] on a finish
//! frame, [`Pending::fail`] on an error after dispatch, or `cancelled` when it is dropped
//! unsettled (the client went away). A request refused before dispatch gets no record:
//! its error body and headers carry the ids, and the agent records the refusal.
//!
//! Owner: server (accounting).
//! Invariants:
//! - At most one record and one outcome count per request.
//! - Records carry counts, timings and ids only; never prompt or output text.
//! - Every record written passes `InferenceUsage::validate` (the vendored contract rules).
//! - A binary without a valid `METRALE_BUILD_COMMIT` writes no record (the contract keys
//!   every record by runtime commit); it says so once in the log.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use metrale_qci_dump::contract::field::{AbsentReason, Field};
use metrale_qci_dump::contract::ids::GitCommit;
use metrale_qci_dump::contract::usage::{
    EnergyRail, EngineErrorKind, InferenceUsage, Outcome, RequestEnergy, RuntimeIdentity,
    RuntimeTimings, SchedulerSpan, Tokens,
};

use crate::request_stamp::RequestStamp;

/// 2026-10-11: The log target of the per-request usage records (one JSON object a line).
pub(crate) const USAGE_LOG_TARGET: &str = "metrale::inference_usage";

/// 2026-10-11: What the scheduler's finish frame(s) report for one request. For a
/// blocking request with several choices: total completion tokens, the first choice's
/// prefill instant and TTFT, the last choice's decode window.
pub(crate) struct Finish {
    /// 2026-10-11: `Outcome::of_finish_reason` of the finish frame; `Timeout` when any
    /// choice hit the server-side deadline.
    pub(crate) outcome: Outcome,
    pub(crate) completion: usize,
    pub(crate) cached_prompt: usize,
    pub(crate) reasoning: usize,
    pub(crate) prefill_started_at: Instant,
    pub(crate) ttft_ms: f64,
    pub(crate) decode_ms: f64,
    /// 2026-10-11: The device layer's energy for the request (every choice merged).
    pub(crate) energy: metrale_telemetry::energy::RequestEnergy,
}

pub(crate) struct Pending {
    stamp: RequestStamp,
    served_model: String,
    /// 2026-10-11: The launch recipe's digest (`model_identity`), `None` without one.
    recipe: Option<metrale_qci_dump::contract::ids::Sha256Digest>,
    prompt_len: usize,
    /// 2026-10-11: The served model has a reasoning channel, so a reasoning count of 0 is
    /// a measurement rather than an unsupported field.
    reasoning_channel: bool,
    dispatched: AtomicBool,
    settled: AtomicBool,
}

impl Pending {
    /// 2026-10-11: A pending record for a request to `state`'s model with a prompt of
    /// `prompt_len` tokens. The model has a reasoning channel when it has a think-end token
    /// or speaks Harmony.
    pub(crate) fn new(stamp: RequestStamp, state: &crate::AppState, prompt_len: usize) -> Self {
        Self {
            stamp,
            served_model: state.model_name.clone(),
            recipe: state.model_identity.recipe,
            prompt_len,
            reasoning_channel: state.think_end_token_id.is_some()
                || state.tokenizer.harmony().is_some(),
            dispatched: AtomicBool::new(false),
            settled: AtomicBool::new(false),
        }
    }

    /// 2026-10-11: The request's identity, for error bodies written after dispatch.
    pub(crate) fn trace(&self) -> &metrale_qci_dump::trace::RequestTrace {
        &self.stamp.trace
    }

    /// 2026-10-11: The scheduler accepted the request; from here an unsettled drop is a
    /// cancellation.
    pub(crate) fn dispatched(&self) {
        self.dispatched.store(true, Ordering::Release);
    }

    fn settle(&self) -> bool {
        self.dispatched.load(Ordering::Acquire) && !self.settled.swap(true, Ordering::AcqRel)
    }

    /// 2026-10-11: The request finished. Counts it and returns its record (`None` when the
    /// build is unstamped or the request was already settled).
    pub(crate) fn complete(&self, f: Finish) -> Option<InferenceUsage> {
        if !self.settle() {
            return None;
        }
        let queue = f
            .prefill_started_at
            .checked_duration_since(self.stamp.received_at);
        count_finished(&self.served_model, self.prompt_len, &f, queue);
        let span = SchedulerSpan {
            queue,
            ttft_ms: f.ttft_ms,
            decode_ms: f.decode_ms,
        };
        let tokens = Tokens::finished(
            self.prompt_len,
            f.completion,
            f.cached_prompt,
            self.reasoning_channel.then_some(f.reasoning),
        );
        let timings = RuntimeTimings::derive(self.stamp.received_unix_ms, Some(span), f.completion);
        self.emit(f.outcome, tokens, timings, energy_field(f.energy))
    }

    /// 2026-10-11: The request failed after dispatch.
    pub(crate) fn fail(&self, error: EngineErrorKind) {
        if self.settle() {
            self.emit_unfinished(Outcome::EngineError { error });
        }
    }

    /// 2026-10-11: [`Self::fail`] for the HTTP status a handler answered with; a status
    /// below 400 settles nothing.
    pub(crate) fn fail_status(&self, status: axum::http::StatusCode) {
        if status.is_client_error() || status.is_server_error() {
            self.fail(EngineErrorKind::of_status(status.as_u16()));
        }
    }

    fn emit_unfinished(&self, outcome: Outcome) {
        let timings = RuntimeTimings::derive(self.stamp.received_unix_ms, None, 0);
        // 2026-10-11: No finish frame, so the scheduler attributed nothing.
        let off = metrale_telemetry::global().level() == metrale_telemetry::Level::Off;
        let energy = Field::Absent(if off {
            AbsentReason::Unsupported
        } else {
            AbsentReason::Unavailable
        });
        self.emit(
            outcome,
            Tokens::unfinished(self.prompt_len),
            timings,
            energy,
        );
    }

    fn emit(
        &self,
        outcome: Outcome,
        tokens: Tokens,
        timings: RuntimeTimings,
        energy: Field<RequestEnergy>,
    ) -> Option<InferenceUsage> {
        crate::metrics::REQUEST_OUTCOMES
            .with_label_values(&[outcome.label()])
            .inc();
        let runtime = RuntimeIdentity {
            runtime_commit: runtime_commit()?,
            recipe: self.recipe,
        };
        let record = InferenceUsage::new(
            runtime,
            self.stamp.ids(),
            &self.served_model,
            outcome,
            tokens,
            timings,
            energy,
            crate::ids::unix_millis(),
        );
        // 2026-10-11: A record that breaks the contract (a served-model name over 256
        // characters) is withheld rather than written.
        if let Err(v) = record.validate() {
            tracing::error!("inference usage record withheld: {v}");
            return None;
        }
        match serde_json::to_string(&record) {
            Ok(line) => tracing::info!(target: USAGE_LOG_TARGET, "{line}"),
            Err(e) => tracing::error!("inference usage record did not serialize: {e}"),
        }
        Some(record)
    }
}

impl Drop for Pending {
    fn drop(&mut self) {
        if self.settle() {
            self.emit_unfinished(Outcome::Cancelled);
        }
    }
}

/// 2026-10-11: The outcome of one finish frame's `finish_reason`.
pub(crate) fn outcome_of(finish_reason: &str) -> Outcome {
    Outcome::of_finish_reason(finish_reason, crate::ir::FINISH_REASON_TIMEOUT)
}

/// 2026-10-11: The runtime commit this binary was stamped with at build time
/// (`crate::build_stamp`). `None`, logged once, when the stamp is missing or invalid.
pub(crate) fn runtime_commit() -> Option<GitCommit> {
    static WARNED: OnceLock<()> = OnceLock::new();
    let commit = crate::build_stamp::build_commit();
    if commit.is_none() {
        WARNED.get_or_init(|| {
            tracing::warn!(
                "inference usage records are withheld: this binary was built without a \
                 40-hex METRALE_BUILD_COMMIT (got {:?}), and every record is keyed by the \
                 runtime commit",
                crate::build_stamp::RAW
            );
        });
    }
    commit
}

/// 2026-10-11: A request's energy as its record carries it: the device layer's token share,
/// derived, read on the GPU rail between two measured counter samples; absent with the
/// reason the device layer gave.
pub(crate) fn energy_field(e: metrale_telemetry::energy::RequestEnergy) -> Field<RequestEnergy> {
    use metrale_telemetry::energy::RequestEnergy as Device;
    match e {
        Device::Attributed(s) => Field::Derived(RequestEnergy {
            millijoules: s.millijoules,
            rail: EnergyRail::GpuRail,
            window_mono_ns: Field::Measured(s.window_ns),
        }),
        Device::Off => Field::Absent(AbsentReason::Unsupported),
        Device::Unattributed => Field::Absent(AbsentReason::Unavailable),
    }
}

/// 2026-10-11: The `/metrics` counters of a finished request.
fn count_finished(model: &str, prompt_len: usize, f: &Finish, queue: Option<Duration>) {
    use crate::metrics::*;
    PROMPT_TOKENS_TOTAL.inc_by(prompt_len as u64);
    GENERATION_TOKENS_TOTAL.inc_by(f.completion as u64);
    PREFILL_TOKENS_TOTAL.inc_by(prompt_len.saturating_sub(f.cached_prompt) as u64);
    TTFT_SECONDS
        .with_label_values(&[model])
        .observe(f.ttft_ms / 1000.0);
    if let Some(q) = queue {
        QUEUE_SECONDS
            .with_label_values(&[model])
            .observe(q.as_secs_f64());
    }
    for (counter, ms) in [
        (&*PREFILL_SECONDS_TOTAL, f.ttft_ms),
        (&*DECODE_SECONDS_TOTAL, f.decode_ms),
    ] {
        if ms.is_finite() && ms > 0.0 {
            counter.inc_by(ms / 1000.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use metrale_telemetry::energy::{EnergyShare, RequestEnergy as Device};

    use super::*;

    #[test]
    fn device_energy_becomes_a_derived_figure_or_an_absent_reason_never_zero() {
        let share = Device::Attributed(EnergyShare {
            millijoules: 1_900,
            window_ns: (100, 3_100),
        });
        assert_eq!(
            serde_json::to_value(energy_field(share)).unwrap(),
            serde_json::json!({"derived": {"millijoules": 1900, "rail": "gpu_rail",
                                           "window_mono_ns": {"measured": [100, 3100]}}})
        );
        assert_eq!(
            energy_field(Device::Off),
            Field::Absent(AbsentReason::Unsupported)
        );
        assert_eq!(
            energy_field(Device::Unattributed),
            Field::Absent(AbsentReason::Unavailable)
        );
    }
}
