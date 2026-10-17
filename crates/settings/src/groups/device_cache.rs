// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `CacheTuning`, the levers the prefix cache (radix tree, SSM snapshot index, KV
//! block trace) reads at run time, projected once from the resolved [`InferenceServerSettings`]
//! and passed to the cache when it is built (ROTI).
//!
//! Owner: settings.
//! Invariants: reads only `InferenceServerSettings`; every Text lever keeps the parse of the
//! reader it replaced.

use crate::resolve::{InferenceServerSettings, SettingsError};

/// 2026-10-17: `METRALE_SSM_TAIL_LEASE_TTL` when unset or not a `u32`.
pub const TAIL_LEASE_TTL_BUILT_IN: u32 = 64;
/// 2026-10-17: The range `METRALE_SNAP_EVICT_ALPHA` is clamped to.
pub const SNAP_EVICT_ALPHA_MAX: f64 = 8.0;

/// 2026-10-17: The levers the prefix cache reads at run time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CacheTuning {
    /// 2026-10-17: `METRALE_PREFIX_SUBBLOCK`: sub-block tail matching (unsound; off by default).
    pub prefix_subblock: bool,
    /// 2026-10-17: `METRALE_KV_TRACE`.
    pub kv_trace: bool,
    /// 2026-10-17: The SSM tail lease: on unless `METRALE_DISABLE_SSM_TAIL_PROTECT` is `1`, `on`
    /// or `true`.
    pub tail_lease: bool,
    /// 2026-10-17: `METRALE_SSM_TAIL_LEASE_TTL`, in evictions.
    pub tail_lease_ttl: u32,
    /// 2026-10-17: `METRALE_SNAP_EVICT_ALPHA`, clamped to `0..=8`; 0 when unset.
    pub snap_evict_alpha: f64,
    /// 2026-10-17: `METRALE_SNAP_LOOKUP_DBG` (any value).
    pub snap_lookup_dbg: bool,
    /// 2026-10-17: `METRALE_SSM_SNAP_STATS` (any value).
    pub ssm_snap_stats: bool,
    /// 2026-10-17: `METRALE_SNAP_EVICT_LEGACY` (any value).
    pub snap_evict_legacy: bool,
    /// 2026-10-17: Hermetic mode (`--hermetic`, carried in the CLI layer as `METRALE_HERMETIC=1`,
    /// or `METRALE_HERMETIC=1`): snapshot lookups miss.
    pub hermetic: bool,
}

impl CacheTuning {
    /// 2026-10-17: The tuning of an empty environment, no CLI and no recipe: the declared defaults.
    ///
    /// # Errors
    /// The resolver's error, should a declared default fail to parse.
    pub fn declared() -> Result<Self, SettingsError> {
        Ok(Self::from(&super::device::declared()?))
    }
}

impl From<&InferenceServerSettings> for CacheTuning {
    fn from(s: &InferenceServerSettings) -> Self {
        let sched = &s.scheduler;
        Self {
            prefix_subblock: s.cache.prefix_subblock,
            kv_trace: s.telemetry.kv_trace,
            tail_lease: !matches!(
                sched.disable_ssm_tail_protect.as_deref(),
                Some("1" | "on" | "true")
            ),
            tail_lease_ttl: sched
                .ssm_tail_lease_ttl
                .as_deref()
                .and_then(|v| v.parse().ok())
                .unwrap_or(TAIL_LEASE_TTL_BUILT_IN),
            snap_evict_alpha: s
                .runtime
                .snap_evict_alpha
                .map_or(0.0, |a| a.clamp(0.0, SNAP_EVICT_ALPHA_MAX)),
            snap_lookup_dbg: s.kernels.snap_lookup_dbg,
            ssm_snap_stats: s.telemetry.ssm_snap_stats,
            snap_evict_legacy: s.runtime.snap_evict_legacy,
            hermetic: s.server.hermetic.as_deref() == Some("1"),
        }
    }
}

#[cfg(test)]
#[path = "device_cache_tests.rs"]
mod tests;
