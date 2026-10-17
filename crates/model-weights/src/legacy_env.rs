// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: LEGACY. Every process-environment read of a `WeightsSettings` lever left in
//! metrale-model-weights, kept so that the constructors the server, model engine and model
//! crates still call (`GgufLoader::new`/`with_ep`/`default`, `ShardFiles::open`/`open_dir`,
//! `ExpertLru::new`, `ExpertArena::alloc`, `RdmaWeightLoader::new`/`with_ep`,
//! `RdmaLoraLoader::new`) keep their current meaning until those callers pass a resolved
//! [`WeightsSettings`]. [`legacy_env_weights_settings_from`] reproduces the deleted readers'
//! parses (the `gpu_sys::env` RDMA fallback chains included) over an injected lookup, and
//! `legacy_env_tests.rs` proves it equals the `WeightsSettings` projection except for the listed
//! deviations. The 2c-device-wire commit deletes this module.
//!
//! Owner: model-weights.
//! Invariants: the only module in metrale-model-weights that reads a `WeightsSettings` lever
//! from the process environment.

use std::env::VarError;
use std::path::Path;

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_settings::groups::device_weights::{
    EVICT_RANDOM_PCT_BUILT_IN, RAIL2_DEV_BUILT_IN, RDMA_DEV_BUILT_IN, RDMA_GID_BUILT_IN, Rail,
    WeightsSettings,
};

use crate::weight_lora_rdma::RdmaLoraLoader;
use crate::weights::GgufLoader;
use crate::weights::expert_stream::{ExpertArena, ExpertLru, ShardFiles, SlotLayout};

/// 2026-10-17: The process-environment lookup the `legacy_env_*` readers are given in production.
fn process_var(name: &str) -> Result<String, VarError> {
    std::env::var(name)
}

fn eq_one(var: &impl Fn(&str) -> Result<String, VarError>, name: &str) -> bool {
    var(name).as_deref() == Ok("1")
}

fn not_zero(var: &impl Fn(&str) -> Result<String, VarError>, name: &str) -> bool {
    var(name).as_deref() != Ok("0")
}

/// 2026-10-17: The deleted `gpu_sys::env::first_set` use: the first set value, empty included.
fn first_set(var: &impl Fn(&str) -> Result<String, VarError>, keys: &[&str], d: &str) -> String {
    keys.iter()
        .find_map(|k| var(k).ok())
        .unwrap_or_else(|| d.into())
}

/// 2026-10-17: The deleted `gpu_sys::env::first_nonempty` use: an empty value is skipped.
fn first_nonempty(
    var: &impl Fn(&str) -> Result<String, VarError>,
    keys: &[&str],
    d: &str,
) -> String {
    keys.iter()
        .find_map(|k| var(k).ok().filter(|v| !v.is_empty()))
        .unwrap_or_else(|| d.into())
}

/// 2026-10-17: The deleted `gpu_sys::env::first_set_u32` use: a value that does not parse is
/// skipped.
fn first_set_u32(var: &impl Fn(&str) -> Result<String, VarError>, keys: &[&str], d: u32) -> u32 {
    keys.iter()
        .find_map(|k| var(k).ok().and_then(|v| v.parse().ok()))
        .unwrap_or(d)
}

/// 2026-10-17: `WeightsSettings` as the deleted model-weights readers parsed them: the `== "1"`
/// switches, the `!= "0"` texts, the lenient group and eviction parses, and the three RDMA rails
/// as `weight_tier_rdma` and `weight_lora_rdma` chained them. Infallible, as they were.
pub fn legacy_env_weights_settings_from(
    var: impl Fn(&str) -> Result<String, VarError>,
) -> WeightsSettings {
    const W_DEV: &str = "METRALE_WEIGHT_RDMA_DEV";
    const E_DEV: &str = "METRALE_EXPERT_RDMA_DEV";
    WeightsSettings {
        gguf_native_q2: eq_one(&var, "METRALE_GGUF_NATIVE_Q2"),
        q2_group: if var("METRALE_GGUF_Q2_GROUP").as_deref() == Ok("64") {
            64
        } else {
            128
        },
        gguf_force_cpu: eq_one(&var, "METRALE_GGUF_FORCE_CPU"),
        keep_page_cache: eq_one(&var, "METRALE_DS41_KEEP_PAGE_CACHE"),
        direct_reads: not_zero(&var, "METRALE_DS41_DIRECT_READS"),
        kquant_resident: not_zero(&var, "METRALE_DS41_KQUANT_RESIDENT"),
        q6k_head: not_zero(&var, "METRALE_DS41_Q6K_HEAD"),
        evict_random_pct: var("METRALE_DS41_EVICT_RANDOM_PCT")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|p| *p <= 100)
            .unwrap_or(EVICT_RANDOM_PCT_BUILT_IN),
        weight_rail0: Rail {
            dev: first_nonempty(&var, &[W_DEV, E_DEV], RDMA_DEV_BUILT_IN),
            gid: first_set_u32(
                &var,
                &["METRALE_WEIGHT_RDMA_GID", "METRALE_EXPERT_RDMA_GID"],
                RDMA_GID_BUILT_IN,
            ),
        },
        weight_rail1: eq_one(&var, "METRALE_WEIGHT_DUAL_RAIL").then(|| Rail {
            dev: first_nonempty(
                &var,
                &["METRALE_WEIGHT_RAIL2_DEV", "METRALE_EXPERT_RAIL2_DEV"],
                RAIL2_DEV_BUILT_IN,
            ),
            gid: first_set_u32(
                &var,
                &["METRALE_WEIGHT_RAIL2_GID", "METRALE_EXPERT_RAIL2_GID"],
                RDMA_GID_BUILT_IN,
            ),
        }),
        lora_rail: Rail {
            dev: first_set(
                &var,
                &["METRALE_LORA_RDMA_DEV", W_DEV, E_DEV],
                RDMA_DEV_BUILT_IN,
            ),
            gid: first_set_u32(&var, &["METRALE_LORA_RDMA_GID"], RDMA_GID_BUILT_IN),
        },
    }
}

/// 2026-10-17: [`legacy_env_weights_settings_from`] on the process environment.
pub fn legacy_env_weights_settings() -> WeightsSettings {
    legacy_env_weights_settings_from(process_var)
}

impl GgufLoader {
    /// 2026-10-17: LEGACY. [`GgufLoader::with_settings`] with no expert parallelism, at
    /// [`legacy_env_weights_settings`].
    pub fn new() -> Self {
        Self::with_settings(0, 1, 0, legacy_env_weights_settings())
    }

    /// 2026-10-17: LEGACY. [`GgufLoader::with_settings`] at [`legacy_env_weights_settings`].
    pub fn with_ep(ep_rank: usize, ep_world_size: usize, num_experts: usize) -> Self {
        Self::with_settings(
            ep_rank,
            ep_world_size,
            num_experts,
            legacy_env_weights_settings(),
        )
    }
}

impl Default for GgufLoader {
    /// 2026-10-17: LEGACY. [`GgufLoader::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl ShardFiles {
    /// 2026-10-17: LEGACY. [`ShardFiles::open_with`] at [`legacy_env_weights_settings`].
    ///
    /// # Errors
    /// As [`ShardFiles::open_with`].
    pub fn open(first: &Path) -> Result<Self> {
        Self::open_with(first, &legacy_env_weights_settings())
    }

    /// 2026-10-17: LEGACY. [`ShardFiles::open_dir_with`] at [`legacy_env_weights_settings`].
    ///
    /// # Errors
    /// As [`ShardFiles::open_dir_with`].
    pub fn open_dir(dir: &Path) -> Result<Self> {
        Self::open_dir_with(dir, &legacy_env_weights_settings())
    }
}

impl ExpertLru {
    /// 2026-10-17: LEGACY. [`ExpertLru::with_settings`] at [`legacy_env_weights_settings`].
    ///
    /// # Errors
    /// As [`ExpertLru::with_settings`].
    pub fn new(host: *mut u8, dev: DevicePtr, bytes: usize, layout: SlotLayout) -> Result<Self> {
        Self::with_settings(host, dev, bytes, layout, &legacy_env_weights_settings())
    }
}

impl ExpertArena {
    /// 2026-10-17: LEGACY. [`ExpertArena::alloc_with`] at [`legacy_env_weights_settings`].
    ///
    /// # Errors
    /// As [`ExpertArena::alloc_with`].
    pub fn alloc(
        gpu: &dyn GpuBackend,
        bytes: usize,
        layout: SlotLayout,
        device: bool,
        ring_slots: usize,
    ) -> Result<(Self, ExpertLru)> {
        let w = legacy_env_weights_settings();
        Self::alloc_with(gpu, bytes, layout, device, ring_slots, &w)
    }
}

#[cfg(feature = "cuda")]
impl crate::weight_tier_rdma::RdmaWeightLoader {
    /// 2026-10-17: LEGACY. [`Self::with_settings`] with no expert parallelism, at
    /// [`legacy_env_weights_settings`].
    pub fn new(peer_addr: String) -> Self {
        Self::with_settings(peer_addr, 0, 1, 0, &legacy_env_weights_settings())
    }

    /// 2026-10-17: LEGACY. [`Self::with_settings`] at [`legacy_env_weights_settings`].
    pub fn with_ep(
        peer_addr: String,
        ep_rank: usize,
        ep_world_size: usize,
        num_experts: usize,
    ) -> Self {
        let w = legacy_env_weights_settings();
        Self::with_settings(peer_addr, ep_rank, ep_world_size, num_experts, &w)
    }
}

impl RdmaLoraLoader {
    /// 2026-10-17: LEGACY. [`RdmaLoraLoader::with_settings`] at [`legacy_env_weights_settings`].
    pub fn new(peer_addr: String, adapter_id: String) -> Self {
        Self::with_settings(peer_addr, adapter_id, &legacy_env_weights_settings())
    }
}

#[cfg(test)]
#[path = "legacy_env_tests.rs"]
mod tests;
