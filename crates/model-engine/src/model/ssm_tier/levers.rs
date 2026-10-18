// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The SSM snapshot-tier levers, resolved once from the startup settings when
//! `build_ssm_snapshots` builds the snapshot pool.
//!
//! Owner: model-engine (SSM snapshot tier).
//! Invariants: the tier gate, the store selectors and the fingerprint read these fields,
//! never the environment, except for the three namespace values `fingerprint` still parses
//! (their settings rows are presence switches, which carry no value).

/// 2026-10-17: The resolved SSM tier levers. Raw text rows are kept verbatim, so an empty
/// value still counts as set where the old `var_os` readers counted it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SsmTierLevers {
    /// 2026-10-17: `cache.ssm_tier` (`METRALE_SSM_TIER`, any value): build a spill tier.
    pub(crate) enabled: bool,
    /// 2026-10-17: `cache.ssm_rdma_tier` (`METRALE_SSM_RDMA_TIER`), verbatim.
    pub(crate) rdma_tier: Option<String>,
    /// 2026-10-17: Peer arena slots: `cache.ssm_rdma_arena_slots`
    /// (`METRALE_SSM_RDMA_ARENA_SLOTS`) when it parses as a `usize`, else 512.
    pub(crate) rdma_arena_slots: usize,
    /// 2026-10-17: The `cache.ssm_swap` switch (`METRALE_SSM_SWAP`): paging mode.
    pub(crate) swap: bool,
    /// 2026-10-17: `cache.ssm_decode_tier` (`METRALE_SSM_DECODE_TIER`), verbatim.
    pub(crate) decode_tier: Option<String>,
    /// 2026-10-17: `runtime.ssm_decode_nvme_dir` (`METRALE_SSM_DECODE_NVME_DIR`), verbatim.
    pub(crate) decode_nvme_dir: Option<String>,
    /// 2026-10-17: `runtime.ssm_decode_rdma_tier` (`METRALE_SSM_DECODE_RDMA_TIER`), verbatim.
    pub(crate) decode_rdma_tier: Option<String>,
    /// 2026-10-17: `cache.ssm_decode_ring_roll` (`METRALE_SSM_DECODE_RING_ROLL`) is set.
    pub(crate) decode_ring_roll_set: bool,
    /// 2026-10-17: `cache.ssm_tier_unified` (`METRALE_SSM_TIER_UNIFIED`) is `1`, `true`, `on`
    /// or `yes` after trimming.
    pub(crate) unified: bool,
    /// 2026-10-17: `cache.ssm_tier_swap_dir` (`METRALE_SSM_TIER_SWAP_DIR`), verbatim.
    pub(crate) swap_dir: Option<String>,
    /// 2026-10-17: Unified hot-arena slots: the non-negative `cache.ssm_tier_slots`
    /// (`METRALE_SSM_TIER_SLOTS`), else 64, at least 1.
    pub(crate) hot_slots: usize,
    /// 2026-10-17: `cache.ssm_tier_disk_gb` (`METRALE_SSM_TIER_DISK_GB`), verbatim; parsed
    /// strictly by `disk_slots_from` on the unified arms only.
    pub(crate) disk_gb: Option<String>,
    /// 2026-10-17: `model.model_id` (`METRALE_MODEL_ID`), empty when unset.
    pub(crate) model_id: String,
    /// 2026-10-17: `cache.ssm_swap_ns` or `kernels.ssm_decode_ns` is present.
    pub(crate) ns_overridden: bool,
    /// 2026-10-17: `kernels.ssm_decode_client_id` is present.
    pub(crate) decode_client_id_set: bool,
}

impl SsmTierLevers {
    pub(crate) fn from_settings(s: &metrale_settings::InferenceServerSettings) -> Self {
        let c = &s.cache;
        Self {
            enabled: c.ssm_tier,
            rdma_tier: c.ssm_rdma_tier.clone(),
            rdma_arena_slots: c
                .ssm_rdma_arena_slots
                .as_deref()
                .and_then(|v| v.parse().ok())
                .unwrap_or(512),
            swap: c.ssm_swap,
            decode_tier: c.ssm_decode_tier.clone(),
            decode_nvme_dir: s.runtime.ssm_decode_nvme_dir.clone(),
            decode_rdma_tier: s.runtime.ssm_decode_rdma_tier.clone(),
            decode_ring_roll_set: c.ssm_decode_ring_roll.is_some(),
            unified: super::unified::unified_flag_truthy(c.ssm_tier_unified.as_deref()),
            swap_dir: c.ssm_tier_swap_dir.clone(),
            hot_slots: c
                .ssm_tier_slots
                .and_then(|v| usize::try_from(v).ok())
                .unwrap_or(64)
                .max(1),
            disk_gb: c.ssm_tier_disk_gb.clone(),
            model_id: s.model.model_id.clone().unwrap_or_default(),
            ns_overridden: c.ssm_swap_ns || s.kernels.ssm_decode_ns,
            decode_client_id_set: s.kernels.ssm_decode_client_id,
        }
    }

    /// 2026-10-17: The tier-request env vars found set, in the order the old gate listed them.
    pub(crate) fn requested_tier_vars(&self) -> Vec<&'static str> {
        [
            (self.enabled, "METRALE_SSM_TIER"),
            (self.rdma_tier.is_some(), "METRALE_SSM_RDMA_TIER"),
            (self.swap, "METRALE_SSM_SWAP"),
            (self.decode_tier.is_some(), "METRALE_SSM_DECODE_TIER"),
            (self.decode_ring_roll_set, "METRALE_SSM_DECODE_RING_ROLL"),
        ]
        .into_iter()
        .filter_map(|(set, var)| set.then_some(var))
        .collect()
    }

    /// 2026-10-17: The peer address for the spill tier, when non-empty.
    pub(crate) fn rdma_peer(&self) -> Option<&str> {
        non_empty(self.rdma_tier.as_deref())
    }

    /// 2026-10-17: Whether a non-empty disk budget is set.
    pub(crate) fn disk_gb_requested(&self) -> bool {
        non_empty(self.disk_gb.as_deref()).is_some()
    }
}

pub(crate) fn non_empty(v: Option<&str>) -> Option<&str> {
    v.filter(|s| !s.is_empty())
}

#[cfg(test)]
#[path = "levers_tests.rs"]
mod tests;
