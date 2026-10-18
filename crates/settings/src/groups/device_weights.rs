// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `WeightsSettings`, the levers model-weights reads when it loads and streams
//! weights, projected once from the resolved [`InferenceServerSettings`] and passed to the
//! loader (ROTI). The RDMA fallback chains (`METRALE_WEIGHT_*` falling back to
//! `METRALE_EXPERT_*`) are resolved here, so a loader sees one device name and one GID per rail.
//! The built-in devices and GID are the declared defaults of the `METRALE_EXPERT_*` and
//! `METRALE_LORA_RDMA_GID` rows.
//!
//! Owner: settings.
//! Invariants: reads only `InferenceServerSettings`; a value the loader cannot use is a
//! `ProjectionOutOfRange` error, never a silent default.

use alloc::string::String;

use metrale_error::Error;

use crate::groups::device::{DeviceSettingsError, in_u32, out_of_range, row_default};
use crate::groups::kernels::KernelsSettings;
use crate::groups::storage::StorageSettings;
use crate::resolve::InferenceServerSettings;

/// 2026-10-17: The rail-0 device an empty `*_RDMA_DEV` value falls back to.
const RAIL0_DEV_DECLARED: &str = row_default(StorageSettings::LEVERS, "METRALE_EXPERT_RDMA_DEV");
/// 2026-10-17: The rail-1 device an empty `*_RAIL2_DEV` value falls back to.
const RAIL1_DEV_DECLARED: &str = row_default(KernelsSettings::LEVERS, "METRALE_EXPERT_RAIL2_DEV");

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
    /// 2026-10-17: `METRALE_GGUF_Q2_GROUP`: 64 or 128.
    pub q2_group: usize,
    /// 2026-10-17: `METRALE_GGUF_FORCE_CPU`.
    pub gguf_force_cpu: bool,
    /// 2026-10-17: `METRALE_DS41_KEEP_PAGE_CACHE`.
    pub keep_page_cache: bool,
    /// 2026-10-17: `METRALE_DS41_DIRECT_READS`.
    pub direct_reads: bool,
    /// 2026-10-17: `METRALE_DS41_KQUANT_RESIDENT`.
    pub kquant_resident: bool,
    /// 2026-10-17: `METRALE_DS41_Q6K_HEAD`.
    pub q6k_head: bool,
    /// 2026-10-17: `METRALE_DS41_EVICT_RANDOM_PCT`, 0..=100.
    pub evict_random_pct: usize,
    /// 2026-10-17: Weight-tier rail 0 (`METRALE_WEIGHT_RDMA_*`, then `METRALE_EXPERT_RDMA_*`;
    /// an empty device name is skipped, and the first GID set is the one used).
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
    /// A declared default that does not parse or project.
    pub fn declared() -> Result<Self, DeviceSettingsError> {
        Ok(Self::try_from(&super::device::declared()?)?)
    }
}

/// 2026-10-17: The first set, non-empty device name, else `declared`.
fn first_nonempty(chain: &[Option<&str>], declared: &str) -> String {
    chain
        .iter()
        .flatten()
        .copied()
        .find(|v| !v.is_empty())
        .unwrap_or(declared)
        .into()
}

impl TryFrom<&InferenceServerSettings> for WeightsSettings {
    type Error = Error;

    fn try_from(s: &InferenceServerSettings) -> Result<Self, Error> {
        let (st, m, k) = (&s.storage, &s.model, &s.kernels);
        let q2_group = match k.gguf_q2_group {
            64 => 64,
            128 => 128,
            v => return Err(out_of_range("METRALE_GGUF_Q2_GROUP", v, "64 or 128")),
        };
        let evict_random_pct = match usize::try_from(m.ds41_evict_random_pct) {
            Ok(p) if p <= 100 => p,
            _ => {
                return Err(out_of_range(
                    "METRALE_DS41_EVICT_RANDOM_PCT",
                    m.ds41_evict_random_pct,
                    "0..=100",
                ));
            }
        };
        let (weight_gid, weight_gid_env) = match st.weight_rdma_gid {
            Some(g) => (g, "METRALE_WEIGHT_RDMA_GID"),
            None => (st.expert_rdma_gid, "METRALE_EXPERT_RDMA_GID"),
        };
        let weight_rail1 = if m.weight_dual_rail {
            let (gid, env) = match m.weight_rail2_gid {
                Some(g) => (g, "METRALE_WEIGHT_RAIL2_GID"),
                None => (k.expert_rail2_gid, "METRALE_EXPERT_RAIL2_GID"),
            };
            Some(Rail {
                dev: first_nonempty(
                    &[m.weight_rail2_dev.as_deref(), Some(&k.expert_rail2_dev)],
                    RAIL1_DEV_DECLARED,
                ),
                gid: in_u32(env, gid)?,
            })
        } else {
            None
        };
        Ok(Self {
            gguf_native_q2: m.gguf_native_q2,
            q2_group,
            gguf_force_cpu: s.runtime.gguf_force_cpu,
            keep_page_cache: s.cache.ds41_keep_page_cache,
            direct_reads: st.ds41_direct_reads,
            kquant_resident: m.ds41_kquant_resident,
            q6k_head: m.ds41_q6k_head,
            evict_random_pct,
            weight_rail0: Rail {
                dev: first_nonempty(
                    &[st.weight_rdma_dev.as_deref(), Some(&st.expert_rdma_dev)],
                    RAIL0_DEV_DECLARED,
                ),
                gid: in_u32(weight_gid_env, weight_gid)?,
            },
            weight_rail1,
            lora_rail: Rail {
                dev: st
                    .lora_rdma_dev
                    .as_deref()
                    .or(st.weight_rdma_dev.as_deref())
                    .unwrap_or(&st.expert_rdma_dev)
                    .into(),
                gid: in_u32("METRALE_LORA_RDMA_GID", st.lora_rdma_gid)?,
            },
        })
    }
}

#[cfg(test)]
#[path = "device_weights_tests.rs"]
mod tests;
