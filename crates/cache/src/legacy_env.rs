// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: LEGACY. Every process-environment read left in metrale-cache, kept so that the
//! constructors the server and model crates still call (`RadixTree::new`, `PagedKvCache::new`)
//! keep their current meaning until those callers pass a resolved [`CacheTuning`]. Each
//! `legacy_env_*_from` reproduces the deleted reader's parse over an injected lookup, and
//! `legacy_env_tests.rs` proves it equals the `CacheTuning` projection except for the listed
//! deviations. The 2c-device-wire commit deletes this module.
//!
//! Owner: cache.
//! Invariants: the only module in metrale-cache that reads the process environment.

use std::env::VarError;

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::legacy_env::legacy_env_hermetic_from;
use metrale_settings::groups::device_cache::{
    CacheTuning, SNAP_EVICT_ALPHA_MAX, TAIL_LEASE_TTL_BUILT_IN,
};

use crate::kv_cache::{KvCacheConfig, PagedKvCache};
use crate::radix_tree::RadixTree;

/// 2026-10-17: The process-environment lookup the `legacy_env_*` readers are given in production.
fn process_var(name: &str) -> Result<String, VarError> {
    std::env::var(name)
}

/// 2026-10-17: The deleted `std::env::var_os(name).is_some()` reads: set to any value, `0` and
/// the empty string included.
fn present(var: &impl Fn(&str) -> Result<String, VarError>, name: &str) -> bool {
    !matches!(var(name), Err(VarError::NotPresent))
}

/// 2026-10-17: The deleted `partial_tail::enabled_from`: only `1` is on. Unset and `0` are off;
/// any other value is logged as an error and is off, so a typo cannot turn the unsound path on.
fn prefix_subblock_from(raw: Option<&str>) -> bool {
    match raw {
        None => false,
        Some("1") => true,
        Some("0") => false,
        Some(other) => {
            tracing::error!(
                "METRALE_PREFIX_SUBBLOCK={other:?} is not a recognised value (expected \"0\" or \
                 \"1\") — refusing it and leaving sub-block tail matching OFF (see issue #1193)"
            );
            false
        }
    }
}

/// 2026-10-17: `CacheTuning` as the deleted metrale-cache readers parsed it, with `hermetic` as
/// the deleted gpu-runtime env parse: the `== "1"` switches, the presence reads, the lease TTL and
/// α defaulting on a bad value. Infallible, as they were.
pub fn legacy_env_cache_tuning_from(var: impl Fn(&str) -> Result<String, VarError>) -> CacheTuning {
    CacheTuning {
        prefix_subblock: prefix_subblock_from(var("METRALE_PREFIX_SUBBLOCK").ok().as_deref()),
        kv_trace: var("METRALE_KV_TRACE").as_deref() == Ok("1"),
        tail_lease: !matches!(
            var("METRALE_DISABLE_SSM_TAIL_PROTECT").as_deref(),
            Ok("1" | "on" | "true")
        ),
        tail_lease_ttl: var("METRALE_SSM_TAIL_LEASE_TTL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(TAIL_LEASE_TTL_BUILT_IN),
        snap_evict_alpha: var("METRALE_SNAP_EVICT_ALPHA")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .map_or(0.0, |a| a.clamp(0.0, SNAP_EVICT_ALPHA_MAX)),
        snap_lookup_dbg: present(&var, "METRALE_SNAP_LOOKUP_DBG"),
        ssm_snap_stats: present(&var, "METRALE_SSM_SNAP_STATS"),
        snap_evict_legacy: present(&var, "METRALE_SNAP_EVICT_LEGACY"),
        hermetic: legacy_env_hermetic_from(&var),
    }
}

/// 2026-10-17: [`legacy_env_cache_tuning_from`] on the process environment, with `hermetic` from
/// `metrale_gpu_runtime::hermetic_enabled`, which the deleted lookups read: it also answers to
/// `met serve --hermetic`, which the server publishes before it builds a cache.
pub fn legacy_env_cache_tuning() -> CacheTuning {
    CacheTuning {
        hermetic: metrale_gpu_runtime::hermetic_enabled(),
        ..legacy_env_cache_tuning_from(process_var)
    }
}

impl RadixTree {
    /// 2026-10-17: LEGACY. [`RadixTree::with_tuning`] at [`legacy_env_cache_tuning`].
    pub fn new() -> Self {
        Self::with_tuning(&legacy_env_cache_tuning())
    }
}

impl PagedKvCache {
    /// 2026-10-17: LEGACY. [`PagedKvCache::with_tuning`] at [`legacy_env_cache_tuning_from`] on the
    /// process environment. The pool reads only `kv_trace`, so `hermetic` is left at its env parse
    /// rather than resolving the gpu-runtime cell from a constructor.
    ///
    /// # Errors
    /// As [`PagedKvCache::with_tuning`].
    pub fn new(config: KvCacheConfig, num_blocks: usize, gpu: &dyn GpuBackend) -> Result<Self> {
        Self::with_tuning(
            config,
            num_blocks,
            gpu,
            &legacy_env_cache_tuning_from(process_var),
        )
    }
}

#[cfg(test)]
#[path = "legacy_env_tests.rs"]
mod tests;
