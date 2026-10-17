// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `WeightsSettings`, the levers model-weights reads when it loads and streams
//! weights, projected once from the resolved [`InferenceServerSettings`] and passed to the
//! loader (ROTI). The RDMA fallback chains (`METRALE_WEIGHT_*` falling back to
//! `METRALE_EXPERT_*`) are resolved here, so a loader sees one device name and one GID per rail.
//!
//! Owner: settings.
//! Invariants: reads only `InferenceServerSettings`; every Text lever keeps the parse of the
//! reader it replaced.

use alloc::string::String;

use crate::resolve::{InferenceServerSettings, SettingsError};

/// 2026-10-17: Rail-0 RDMA device when no `*_RDMA_DEV` is set.
pub const RDMA_DEV_BUILT_IN: &str = "roceP2p1s0f1";
/// 2026-10-17: Rail-1 RDMA device when no `*_RAIL2_DEV` is set.
pub const RAIL2_DEV_BUILT_IN: &str = "rocep1s0f1";
/// 2026-10-17: RoCE GID index when no GID lever parses.
pub const RDMA_GID_BUILT_IN: u32 = 3;
/// 2026-10-17: `METRALE_DS41_EVICT_RANDOM_PCT` when unset or not a percentage.
pub const EVICT_RANDOM_PCT_BUILT_IN: usize = 8;

/// 2026-10-17: One verbs rail: the RDMA device and its GID index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rail {
    pub dev: String,
    pub gid: u32,
}

/// 2026-10-17: The levers model-weights reads at run time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightsSettings {
    /// 2026-10-17: `METRALE_GGUF_NATIVE_Q2`.
    pub gguf_native_q2: bool,
    /// 2026-10-17: `METRALE_GGUF_Q2_GROUP`: 64 only for exactly `64`, else 128.
    pub q2_group: usize,
    /// 2026-10-17: `METRALE_GGUF_FORCE_CPU`.
    pub gguf_force_cpu: bool,
    /// 2026-10-17: `METRALE_DS41_KEEP_PAGE_CACHE`.
    pub keep_page_cache: bool,
    /// 2026-10-17: `METRALE_DS41_DIRECT_READS`.
    pub direct_reads: bool,
    /// 2026-10-17: `METRALE_DS41_KQUANT_RESIDENT`: off only for exactly `0`.
    pub kquant_resident: bool,
    /// 2026-10-17: `METRALE_DS41_Q6K_HEAD`: off only for exactly `0`.
    pub q6k_head: bool,
    /// 2026-10-17: `METRALE_DS41_EVICT_RANDOM_PCT`, 0..=100.
    pub evict_random_pct: usize,
    /// 2026-10-17: Weight-tier rail 0 (`METRALE_WEIGHT_RDMA_*`, then `METRALE_EXPERT_RDMA_*`;
    /// an empty device name is skipped).
    pub weight_rail0: Rail,
    /// 2026-10-17: Weight-tier rail 1 when `METRALE_WEIGHT_DUAL_RAIL` is on
    /// (`METRALE_WEIGHT_RAIL2_*`, then `METRALE_EXPERT_RAIL2_*`).
    pub weight_rail1: Option<Rail>,
    /// 2026-10-17: The LoRA rail: device `METRALE_LORA_RDMA_DEV`, then the weight and expert
    /// devices (an empty name counts as set); GID from `METRALE_LORA_RDMA_GID` only.
    pub lora_rail: Rail,
}

impl WeightsSettings {
    /// 2026-10-17: The settings of an empty environment, no CLI and no recipe: the declared
    /// defaults.
    ///
    /// # Errors
    /// The resolver's error, should a declared default fail to parse.
    pub fn declared() -> Result<Self, SettingsError> {
        Ok(Self::from(&super::device::declared()?))
    }
}

/// 2026-10-17: The first set, non-empty value, else `default`.
fn first_nonempty(chain: &[Option<&str>], default: &str) -> String {
    chain
        .iter()
        .find_map(|v| v.filter(|v| !v.is_empty()))
        .unwrap_or(default)
        .into()
}

/// 2026-10-17: The first set value (empty included), else `default`.
fn first_set(chain: &[Option<&str>], default: &str) -> String {
    chain.iter().find_map(|v| *v).unwrap_or(default).into()
}

/// 2026-10-17: The first value that is a `u32` (others are skipped), else `default`.
fn first_u32(chain: &[Option<u32>], default: u32) -> u32 {
    chain.iter().find_map(|v| *v).unwrap_or(default)
}

/// 2026-10-18: A Text GID row as the old lenient parse read it.
fn text_u32(v: &Option<String>) -> Option<u32> {
    v.as_deref().and_then(|v| v.parse().ok())
}

/// 2026-10-18: An Int GID row; a value outside `u32` is skipped, as the old parse skipped it.
fn int_u32(v: i64) -> Option<u32> {
    u32::try_from(v).ok()
}

impl From<&InferenceServerSettings> for WeightsSettings {
    fn from(s: &InferenceServerSettings) -> Self {
        let (st, m, k) = (&s.storage, &s.model, &s.kernels);
        let not_zero = |v: &Option<String>| v.as_deref() != Some("0");
        Self {
            gguf_native_q2: m.gguf_native_q2,
            q2_group: if k.gguf_q2_group.as_deref() == Some("64") {
                64
            } else {
                128
            },
            gguf_force_cpu: s.runtime.gguf_force_cpu,
            keep_page_cache: s.cache.ds41_keep_page_cache,
            direct_reads: st.ds41_direct_reads,
            kquant_resident: not_zero(&m.ds41_kquant_resident),
            q6k_head: not_zero(&m.ds41_q6k_head),
            evict_random_pct: m
                .ds41_evict_random_pct
                .as_deref()
                .and_then(|v| v.parse().ok())
                .filter(|p| *p <= 100)
                .unwrap_or(EVICT_RANDOM_PCT_BUILT_IN),
            weight_rail0: Rail {
                dev: first_nonempty(
                    &[st.weight_rdma_dev.as_deref(), Some(&st.expert_rdma_dev)],
                    RDMA_DEV_BUILT_IN,
                ),
                gid: first_u32(
                    &[
                        st.weight_rdma_gid.and_then(int_u32),
                        int_u32(st.expert_rdma_gid),
                    ],
                    RDMA_GID_BUILT_IN,
                ),
            },
            weight_rail1: m.weight_dual_rail.then(|| Rail {
                dev: first_nonempty(
                    &[m.weight_rail2_dev.as_deref(), Some(&k.expert_rail2_dev)],
                    RAIL2_DEV_BUILT_IN,
                ),
                gid: first_u32(
                    &[text_u32(&m.weight_rail2_gid), int_u32(k.expert_rail2_gid)],
                    RDMA_GID_BUILT_IN,
                ),
            }),
            lora_rail: Rail {
                dev: first_set(
                    &[
                        st.lora_rdma_dev.as_deref(),
                        st.weight_rdma_dev.as_deref(),
                        Some(&st.expert_rdma_dev),
                    ],
                    RDMA_DEV_BUILT_IN,
                ),
                gid: first_u32(&[text_u32(&st.lora_rdma_gid)], RDMA_GID_BUILT_IN),
            },
        }
    }
}

#[cfg(test)]
#[path = "device_weights_tests.rs"]
mod tests;
