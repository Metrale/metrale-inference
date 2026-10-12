// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: `RuntimeHealth` v1, what the runtime process reports about itself to the node
//! agent (investor-mvp #41, #28): build, boot, state, served models with their recipe,
//! queue and KV pressure, and one sample per accelerator it reads.
//!
//! Owner: qci-dump (contract binding).
//! Invariants: `reason` is set exactly when `state` is `degraded`; a quantity the runtime
//! cannot read is absent, never zero.

use serde::{Deserialize, Serialize};

use super::field::Field;
use super::ids::{BootId, GitCommit, Sha256Digest};
use super::rules::{self, Violation, require};

/// 2026-10-11: The build answering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeBuild {
    pub name: String,
    pub version: String,
    pub git_commit: GitCommit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthState {
    Starting,
    Ready,
    Degraded,
    Draining,
    Stopped,
}

/// 2026-10-11: Why a runtime is `degraded`. Closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DegradedReason {
    KvCacheExhausted,
    AcceleratorError,
    Overloaded,
    ModelLoading,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServedModel {
    pub served_model: String,
    /// 2026-10-11: The launch recipe's digest; `null` when the process has none.
    #[serde(deserialize_with = "rules::present")]
    pub recipe: Option<Sha256Digest>,
}

/// 2026-10-11: One accelerator reading. `mono_ns` is the reading's time on the writer's
/// monotonic clock.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuSample {
    pub device: String,
    pub product: Field<String>,
    pub energy_cumulative_mj: Field<u64>,
    pub power_mw: Field<u64>,
    /// 2026-10-11: Milli-degrees Celsius.
    pub temperature_mc: Field<i32>,
    pub gpu_util_permille: Field<u32>,
    pub memory_used_bytes: Field<u64>,
    pub memory_total_bytes: Field<u64>,
    pub mono_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeHealth {
    pub version: u32,
    pub runtime: RuntimeBuild,
    pub boot_id: BootId,
    pub state: HealthState,
    #[serde(deserialize_with = "rules::present")]
    pub reason: Option<DegradedReason>,
    pub models: Vec<ServedModel>,
    pub observed_at_unix_ms: i64,
    pub uptime_ms: u64,
    pub queue_depth: Field<u32>,
    pub active_requests: Field<u32>,
    pub kv_occupancy_permille: Field<u32>,
    pub accelerators: Vec<GpuSample>,
}

fn permille(f: &Field<u32>) -> bool {
    f.value().is_none_or(|v| *v <= 1000)
}

impl RuntimeHealth {
    /// 2026-10-11: The schema rules serde cannot express.
    pub fn validate(&self) -> Result<(), Violation> {
        rules::version(self.version)?;
        require(rules::text(&self.runtime.name), "/runtime/name")?;
        require(rules::text(&self.runtime.version), "/runtime/version")?;
        require(
            (self.state == HealthState::Degraded) == self.reason.is_some(),
            "/reason",
        )?;
        for (i, m) in self.models.iter().enumerate() {
            require(
                rules::model_name(&m.served_model),
                &format!("/models/{i}/served_model"),
            )?;
        }
        require(
            rules::timestamp(self.observed_at_unix_ms),
            "/observed_at_unix_ms",
        )?;
        require(
            permille(&self.kv_occupancy_permille),
            "/kv_occupancy_permille",
        )?;
        for (i, a) in self.accelerators.iter().enumerate() {
            require(rules::text(&a.device), &format!("/accelerators/{i}/device"))?;
            require(
                a.product.value().is_none_or(|p| rules::text(p)),
                &format!("/accelerators/{i}/product"),
            )?;
            require(
                permille(&a.gpu_util_permille),
                &format!("/accelerators/{i}/gpu_util_permille"),
            )?;
        }
        Ok(())
    }
}
