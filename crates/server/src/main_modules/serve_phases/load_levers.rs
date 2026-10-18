// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: [`LoadLevers`]: the levers the model-load phases read (weight loader, reserve
//! preflight, KV sizing, topology, media policy, EP worker guard), resolved once from the load's
//! settings at the top of `load_engine` (ROTI), so a malformed value stops the load before any
//! model work (PCND) and no phase reads the environment (SBIO).
//!
//! Owner: server startup (`met serve`).
//! Invariants: every field is a function of the settings alone; the only defaults are the
//! named constants below, taken when the lever is unset.

use anyhow::Result;
use metrale_settings::InferenceServerSettings;

use crate::lever_value::{narrow, narrow_opt, text_int, text_switch};

/// 2026-09-26: Default context length of the KV floor, in tokens per sequence. The floor keeps
/// the decode-ring fit from giving the ring the room a KV cache needs to admit
/// `--max-batch-size` sequences. `METRALE_KV_FLOOR_TOKENS=<n>` overrides it; 0 disables the
/// floor.
pub(crate) const DEFAULT_KV_FLOOR_TOKENS: usize = 4096;

/// 2026-10-17: What one model load's phases are configured with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LoadLevers {
    /// 2026-10-17: `METRALE_ALLOW_SPEC_MISMATCH`: an EP worker without speculative flags may
    /// load a checkpoint that carries MTP weights (`maybe_run_ep_worker`).
    pub(crate) allow_spec_mismatch: bool,
    /// 2026-10-17: `METRALE_SSM_RESERVE_RING_FULL` (presence): reserve the full decode-rollback
    /// ring whatever the speculative flags (`decode_ring::requested_slots`).
    pub(crate) ssm_reserve_ring_full: bool,
    /// 2026-10-17: `METRALE_KV_FLOOR_TOKENS`, or [`DEFAULT_KV_FLOOR_TOKENS`] when unset.
    pub(crate) kv_floor_tokens: usize,
    /// 2026-10-17: `METRALE_GDN_FUSED_CONV`: refused under an FP16 h-state
    /// (`ssm_h_fp16_preconditions`).
    pub(crate) gdn_fused_conv: bool,
    /// 2026-10-17: `METRALE_MAX_BATCH_TOKENS`: a raise of the batch-token budget
    /// (`resolve_prefill_budget`, which ignores one below its default).
    pub(crate) max_batch_tokens: Option<usize>,
    /// 2026-10-17: `--vision-max-pixels` / `METRALE_VISION_MAX_PIXELS`; `None` for 0, which
    /// leaves the checkpoint's own cap (`resolve_vision_max_pixels`).
    pub(crate) vision_max_pixels: Option<usize>,
    /// 2026-10-17: `--no-fast-load` / `METRALE_FAST_LOAD`: the O_DIRECT weight loader.
    pub(crate) fast_load: bool,
    /// 2026-10-17: `--fast-load-prefetch-shards` / `METRALE_FAST_LOAD_PREFETCH_SHARDS`.
    pub(crate) fast_load_prefetch_shards: bool,
    /// 2026-10-17: `METRALE_KV_PAGING` selects the peer-paging KV backend
    /// (`metrale_storage::kv_paging::ns::kv_paging_selected`), which `--indexer-cache-dtype`
    /// is checked against.
    pub(crate) kv_paging: bool,
    /// 2026-10-17: `METRALE_GLM53_PREFIX_CACHE_UNPROVEN` and `METRALE_GLM_DSA_INDEXER_FLAT`:
    /// the prefix-cache gate's validation override and the GLM DSA indexer layout.
    pub(crate) prefix_cache_gate: metrale_config::PrefixCacheGate,
}

impl LoadLevers {
    /// 2026-10-17: This load's values, read from `settings`.
    ///
    /// # Errors
    /// A lever set to a value its reader's grammar rejects: a switch that is not on/off, a
    /// count that is not a non-negative integer, or a `METRALE_KV_PAGING` the storage crate
    /// refuses.
    pub(crate) fn from_settings(settings: &InferenceServerSettings) -> Result<Self> {
        let vision: usize = narrow(
            "METRALE_VISION_MAX_PIXELS",
            settings.model.vision_max_pixels,
        )?;
        Ok(Self {
            allow_spec_mismatch: text_switch(
                "METRALE_ALLOW_SPEC_MISMATCH",
                settings.speculative.allow_spec_mismatch.as_deref(),
            )? == Some(true),
            ssm_reserve_ring_full: settings.cache.ssm_reserve_ring_full,
            kv_floor_tokens: text_int(
                "METRALE_KV_FLOOR_TOKENS",
                settings.cache.kv_floor_tokens.as_deref(),
            )?
            .unwrap_or(DEFAULT_KV_FLOOR_TOKENS),
            gdn_fused_conv: settings.kernels.gdn_fused_conv,
            max_batch_tokens: narrow_opt(
                "METRALE_MAX_BATCH_TOKENS",
                settings.scheduler.max_batch_tokens,
            )?,
            vision_max_pixels: (vision > 0).then_some(vision),
            fast_load: settings.storage.fast_load,
            fast_load_prefetch_shards: settings.storage.fast_load_prefetch_shards,
            kv_paging: metrale_storage::kv_paging::ns::kv_paging_selected(
                settings.runtime.kv_paging.as_deref(),
            )?,
            prefix_cache_gate: metrale_config::PrefixCacheGate::from_settings(&settings.cache),
        })
    }
}

#[cfg(test)]
#[path = "load_levers_tests.rs"]
mod tests;
