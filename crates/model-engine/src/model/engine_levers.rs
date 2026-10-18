// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Engine-local levers whose rule is more than a plain settings field, resolved
//! once from the startup settings when `TransformerModel::new` builds the model.
//!
//! Owner: model-engine.
//! Invariants:
//! - Built once per model; nothing on the decode, prefill or verify paths reads the
//!   environment or re-applies these rules.

use metrale_settings::InferenceServerSettings;

use super::prefill_zero::{KvZeroBlocks, parse_kv_zero_blocks};
use super::trait_impl::lm_head_batched::{bf16_batch_gemv_from_value, m16_tc_n_tile_from_value};

/// 2026-10-17: The resolved engine levers; see [`EngineLevers::from_settings`] for each rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EngineLevers {
    /// 2026-10-17: `METRALE_LM_HEAD_M16_TC_NTILE`: the tensor-core head's N tile.
    pub(crate) lm_head_m16_tc_n_tile: u32,
    /// 2026-10-17: `METRALE_LMHEAD_BATCH_GEMV`: the BF16 head's batched GEMV is on.
    pub(crate) lmhead_batch_gemv: bool,
    /// 2026-10-17: `METRALE_KV_ZERO_BLOCKS`: how a prefill chunk zeroes its new blocks.
    pub(crate) kv_zero_blocks: KvZeroBlocks,
    /// 2026-10-17: `METRALE_NO_BORROW_STREAK_LIMIT`: the graph-borrow streak guard is on.
    pub(crate) borrow_streak_limit: bool,
    /// 2026-10-17: `METRALE_REDZONE` / `METRALE_REDZONE_EVERY`: scan the guard bands every
    /// `n`-th decode step; 0 never scans.
    pub(crate) redzone_every: usize,
    /// 2026-10-17: `METRALE_GLM_VISION_CHECK`: every rank copies its spliced vision rows to
    /// the host to digest them, and a worker whose digest differs from rank 0's returns an
    /// error.
    pub(crate) glm_vision_check: bool,
}

impl EngineLevers {
    /// 2026-10-17: Apply each lever's rule to its settings field. An unrecognised
    /// `METRALE_KV_ZERO_BLOCKS` is the one error.
    pub(crate) fn from_settings(s: &InferenceServerSettings) -> anyhow::Result<Self> {
        Ok(Self {
            lm_head_m16_tc_n_tile: m16_tc_n_tile_from_value(
                s.kernels.lm_head_m16_tc_ntile.as_deref(),
            ),
            lmhead_batch_gemv: bf16_batch_gemv_from_value(s.scheduler.lmhead_batch_gemv.as_deref()),
            kv_zero_blocks: parse_kv_zero_blocks(s.cache.kv_zero_blocks.as_deref())
                .map_err(anyhow::Error::msg)?,
            // 2026-10-17: The guard is on unless the kill switch holds a non-empty value.
            borrow_streak_limit: s
                .scheduler
                .no_borrow_streak_limit
                .as_deref()
                .is_none_or(str::is_empty),
            // 2026-10-17: Off unless `METRALE_REDZONE` is set; then every
            // `METRALE_REDZONE_EVERY`-th step, 1 when that is unset or negative.
            redzone_every: match s.runtime.redzone {
                None => 0,
                Some(_) => s
                    .runtime
                    .redzone_every
                    .map_or(1, |v| usize::try_from(v).unwrap_or(1)),
            },
            glm_vision_check: matches!(
                s.model
                    .glm_vision_check
                    .as_deref()
                    .map(|v| v.trim().to_ascii_lowercase())
                    .as_deref(),
                Some("1" | "true" | "yes" | "on")
            ),
        })
    }
}

#[cfg(test)]
#[path = "engine_levers_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "engine_env_equivalence_tests.rs"]
mod env_equivalence_tests;
