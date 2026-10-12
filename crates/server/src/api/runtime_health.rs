// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: `GET /health/runtime`: this process's `RuntimeHealth` v1 document for the node
//! agent (investor-mvp #41, #28). Outside `/v1`, like `/health` and `/metrics`, so it answers
//! without a key, under a GPU fault and while no model is loaded.
//!
//! State: a latched GPU fault is `degraded` / `accelerator_error`; no model while a swap
//! runs is `degraded` / `model_loading`; no model otherwise is `starting`; a loaded model is
//! `ready`. KV exhaustion, overload and draining are not detected here and never reported.
//!
//! Queue depth and KV occupancy come from the scheduler's telemetry tick and the accelerator
//! sample from the NVML device layer, so both exist only at `--telemetry basic|kernel`; below
//! that they are `absent: unsupported` and `accelerators` is empty.
//!
//! An unstamped build cannot write the document (`runtime.git_commit` is required), so it
//! answers 503 `unstamped_build`.
//!
//! Owner: server (HTTP layer).
//! Invariants: [`assemble`] does no I/O; every document served passes
//! `RuntimeHealth::validate`.

use std::sync::{Arc, OnceLock};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use metrale_qci_dump::contract::PLATFORM_VERSION;
use metrale_qci_dump::contract::field::{AbsentReason, Field};
use metrale_qci_dump::contract::health::{
    DegradedReason, GpuSample, HealthState, RuntimeBuild, RuntimeHealth, ServedModel,
};
use metrale_qci_dump::contract::ids::{BootId, GitCommit, Sha256Digest};
use metrale_telemetry::device::DeviceReading;
use metrale_telemetry::sched::SchedShape;

use crate::main_modules::model_host::ModelHost;

/// 2026-10-11: The `runtime.name` this engine reports.
pub(crate) const RUNTIME_NAME: &str = "metrale-inference";

/// 2026-10-11: The device layer's latest reading of the sampled GPU.
pub(crate) struct DeviceFacts {
    pub(crate) ordinal: usize,
    pub(crate) reading: DeviceReading,
    pub(crate) mono_ns: u64,
}

/// 2026-10-11: Everything the document is built from, read by the handler.
pub(crate) struct HealthFacts<'a> {
    pub(crate) commit: GitCommit,
    pub(crate) version: &'a str,
    pub(crate) boot_id: BootId,
    pub(crate) fault: bool,
    pub(crate) swapping: bool,
    /// 2026-10-11: The served model and its recipe digest, when one is loaded.
    pub(crate) model: Option<(&'a str, Option<Sha256Digest>)>,
    pub(crate) now_unix_ms: i64,
    pub(crate) start_unix_ms: i64,
    pub(crate) active_requests: i64,
    /// 2026-10-11: `false` at `--telemetry off`.
    pub(crate) telemetry_on: bool,
    /// 2026-10-11: The scheduler's last tick, when it has published one.
    pub(crate) sched: Option<SchedShape>,
    pub(crate) device: Option<DeviceFacts>,
}

fn state_of(f: &HealthFacts) -> (HealthState, Option<DegradedReason>) {
    match (f.fault, f.model.is_some(), f.swapping) {
        (true, _, _) => (
            HealthState::Degraded,
            Some(DegradedReason::AcceleratorError),
        ),
        (false, false, true) => (HealthState::Degraded, Some(DegradedReason::ModelLoading)),
        (false, false, false) => (HealthState::Starting, None),
        (false, true, _) => (HealthState::Ready, None),
    }
}

fn measured<T, U: TryFrom<T>>(v: Option<T>) -> Field<U> {
    match v {
        Some(v) => U::try_from(v).map_or(Field::Absent(AbsentReason::Lost), Field::Measured),
        None => Field::Absent(AbsentReason::Unavailable),
    }
}

fn sample(d: &DeviceFacts) -> GpuSample {
    let r = &d.reading;
    GpuSample {
        device: format!("nvml:{}", d.ordinal),
        // 2026-10-11: The device layer does not read the product name.
        product: Field::Absent(AbsentReason::Unsupported),
        energy_cumulative_mj: measured(r.energy_counter_mj),
        power_mw: measured(r.power_mw),
        temperature_mc: measured(r.temperature_c.and_then(|c| c.checked_mul(1000))),
        // 2026-10-11: A percent past 100 is not a reading; it is left out, not clamped.
        gpu_util_permille: measured(r.gpu_util_pct.filter(|p| *p <= 100).map(|p| p * 10)),
        memory_used_bytes: measured(r.mem_used_bytes),
        memory_total_bytes: measured(r.mem_total_bytes),
        mono_ns: d.mono_ns,
    }
}

/// 2026-10-11: The document `f` describes. Pure.
pub(crate) fn assemble(f: &HealthFacts) -> RuntimeHealth {
    let (state, reason) = state_of(f);
    let off = Field::Absent(AbsentReason::Unsupported);
    let (queue_depth, kv_occupancy_permille) = match (f.telemetry_on, f.sched) {
        (false, _) => (off, off),
        (true, None) => (
            Field::Absent(AbsentReason::NotYet),
            Field::Absent(AbsentReason::NotYet),
        ),
        (true, Some(s)) => (
            measured(Some(s.pending)),
            match s.kv_blocks_total {
                0 => Field::Absent(AbsentReason::Unavailable),
                total => Field::Derived(
                    (total.saturating_sub(s.kv_blocks_free).min(total) * 1000 / total) as u32,
                ),
            },
        ),
    };
    RuntimeHealth {
        version: PLATFORM_VERSION,
        runtime: RuntimeBuild {
            name: RUNTIME_NAME.to_string(),
            version: f.version.to_string(),
            git_commit: f.commit,
        },
        boot_id: f.boot_id,
        state,
        reason,
        models: f
            .model
            .map(|(name, recipe)| ServedModel {
                served_model: name.to_string(),
                recipe,
            })
            .into_iter()
            .collect(),
        observed_at_unix_ms: f.now_unix_ms,
        uptime_ms: u64::try_from(f.now_unix_ms.saturating_sub(f.start_unix_ms)).unwrap_or(0),
        queue_depth,
        active_requests: measured(Some(f.active_requests)),
        kv_occupancy_permille,
        accelerators: f.device.iter().map(sample).collect(),
    }
}

/// 2026-10-11: Random per process; minted on first use.
fn boot_id() -> Option<BootId> {
    static BOOT: OnceLock<Option<BootId>> = OnceLock::new();
    *BOOT.get_or_init(|| {
        let mut bytes = [0u8; 16];
        crate::ids::getrandom(&mut bytes).ok()?;
        Some(BootId(bytes))
    })
}

fn unavailable(code: &str, message: &str) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(serde_json::json!({
            "error": {"message": message, "type": "server_error", "code": code}
        })),
    )
        .into_response()
}

/// 2026-10-11: `GET /health/runtime`.
pub async fn runtime_health(State(host): State<Arc<ModelHost>>) -> Response {
    let Some(commit) = crate::request_accounting::runtime_commit() else {
        return unavailable(
            "unstamped_build",
            "this binary was built without a 40-hex METRALE_BUILD_COMMIT; RuntimeHealth \
             requires runtime.git_commit",
        );
    };
    let Some(boot_id) = boot_id() else {
        return unavailable("entropy_unavailable", "no entropy for the boot id");
    };
    let Some(start_s) = crate::metrics::PROCESS_START_UNIX_S.get().copied() else {
        return unavailable("starting", "the process start time is not recorded yet");
    };
    let live = host.current();
    let tel = metrale_telemetry::global();
    let telemetry_on = tel.level() != metrale_telemetry::Level::Off;
    let snap = tel.snapshot();
    let ordinal = host.args().map(|a| a.gpu_ordinal);
    let device = snap
        .device
        .as_ref()
        .zip(ordinal)
        .map(|(d, ordinal)| DeviceFacts {
            ordinal,
            reading: d.reading,
            mono_ns: snap.at_ns.saturating_sub(d.age_ns),
        });
    let facts = HealthFacts {
        commit,
        version: crate::cli::METRALE_VERSION,
        boot_id,
        fault: metrale_core::fault::global().fault().is_some(),
        swapping: host.is_swapping(),
        model: live
            .as_ref()
            .map(|s| (s.model_name.as_str(), s.model_identity.recipe)),
        now_unix_ms: crate::ids::unix_millis(),
        start_unix_ms: (start_s * 1000.0) as i64,
        active_requests: crate::metrics::REQUESTS_ACTIVE.get(),
        telemetry_on,
        sched: (telemetry_on && snap.sched.shape.ticks > 0).then_some(snap.sched.shape),
        device,
    };
    let doc = assemble(&facts);
    if let Err(v) = doc.validate() {
        tracing::error!("runtime health withheld: {v}");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": {"message": format!("runtime health breaks its contract: {v}"),
                          "type": "server_error", "code": "contract_violation"}
            })),
        )
            .into_response();
    }
    Json(doc).into_response()
}

#[cfg(test)]
#[path = "runtime_health_tests.rs"]
mod tests;
