// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The two KV-completeness gates: whether the radix prefix cache and the
//! `--swap-space-gb` swap-out may treat a sequence's KV blocks as its whole state.
//!
//! Owner: config.
//! Invariants:
//! - Without `METRALE_GLM53_PREFIX_CACHE_UNPROVEN`, the two gates give the same answer for
//!   every model but `glm5_next` / `glm5_next_text` with a paged indexer cache.
//! - That switch opens only the prefix-cache gate, and only for `glm5_next` / `glm5_next_text`.
//! - 2026-10-09: A paged GLM indexer cache ([`PrefixCacheGate::glm_indexer_paged`]) opens the
//!   prefix-cache gate for `glm5_next` / `glm5_next_text` and leaves the swap-out gate closed.
//! - 2026-10-17: Both inputs arrive as a [`PrefixCacheGate`] resolved once from the settings;
//!   nothing here reads the environment.

use metrale_settings::groups::cache::CacheSettings;

use super::ModelConfig;

/// 2026-10-17: The two lever inputs of the prefix-cache gate, read once from the resolved
/// settings (ROTI): the GLM-5.3 validation switch and the GLM DSA indexer layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixCacheGate {
    /// 2026-10-17: `METRALE_GLM53_PREFIX_CACHE_UNPROVEN` (presence, `0` included).
    pub validation_override: bool,
    /// 2026-10-17: GLM-5.3's DSA indexer rows live in the KV blocks: on unless
    /// `METRALE_GLM_DSA_INDEXER_FLAT` is on.
    pub glm_indexer_paged: bool,
}

impl PrefixCacheGate {
    /// 2026-10-17: The gate of the cache group's declared defaults: no validation override, the
    /// indexer paged. `met circuit memory` plans with it, because the levers of the serve it
    /// models live in that serve's environment, not the tool's. A server test checks it against
    /// [`Self::from_settings`] over an empty environment.
    pub const DECLARED: Self = Self {
        validation_override: false,
        glm_indexer_paged: true,
    };

    /// 2026-10-17: The gate `cache` configures.
    pub fn from_settings(cache: &CacheSettings) -> Self {
        Self {
            validation_override: cache.glm53_prefix_cache_unproven,
            glm_indexer_paged: !cache.glm_dsa_indexer_flat,
        }
    }
}

impl ModelConfig {
    /// 2026-09-26: Whether every byte of a sequence's per-layer state is held in its KV blocks.
    ///
    /// False for `glm5_next` / `glm5_next_text` (the DSA indexer rows, `Glm5NextDsaState`) and
    /// for `deepseek_v4` when any `compress_ratios` entry is non-zero.
    fn per_sequence_state_is_kv_complete(&self) -> bool {
        match self.model_type.as_str() {
            "glm5_next" | "glm5_next_text" => false,
            "deepseek_v4" => self.compress_ratios.iter().all(|&ratio| ratio == 0),
            _ => true,
        }
    }

    /// 2026-09-26: Whether the radix prefix cache captures every state needed to resume this
    /// model exactly. `preflight_reserve` and `build_prefix_cache` both read it.
    ///
    /// `gate.validation_override` (`METRALE_GLM53_PREFIX_CACHE_UNPROVEN`) also opens it for
    /// `glm5_next`. `kv_only_swap_out_is_safe` ignores that switch: the swap image cannot
    /// carry the DSA rows (see there).
    pub fn kv_only_prefix_cache_is_safe(&self, gate: PrefixCacheGate) -> bool {
        self.kv_only_prefix_cache_is_safe_given(gate.validation_override, gate.glm_indexer_paged)
    }

    /// 2026-10-09: The gate with its two inputs as arguments; the override opens only
    /// `glm5_next` / `glm5_next_text`.
    ///
    /// A paged GLM indexer cache keeps the indexer rows in the KV blocks, which the radix tree
    /// shares; the KDA recurrent and conv state are `LinearAttention` SSM pool state, which a
    /// hit restores from a snapshot like any hybrid's (and a hit without one recomputes); the
    /// mHC streams carry nothing between tokens. So with the cache paged, the KV blocks plus
    /// the SSM snapshot are the whole state.
    pub(crate) fn kv_only_prefix_cache_is_safe_given(
        &self,
        validation_override: bool,
        glm_indexer_paged: bool,
    ) -> bool {
        let glm = matches!(self.model_type.as_str(), "glm5_next" | "glm5_next_text");
        self.per_sequence_state_is_kv_complete()
            || (glm && (validation_override || glm_indexer_paged))
    }

    /// 2026-09-26: Whether a sequence may be swapped out to the `--swap-space-gb` pool and
    /// restored from it; `resolve_swap_space_gb` reads it. `save_sequence_state_dispatch`
    /// writes the KV blocks and the `SsmLayerState` of each `LayerType::LinearAttention`
    /// layer and nothing else, so any other per-sequence state would not survive the swap.
    pub fn kv_only_swap_out_is_safe(&self) -> bool {
        self.per_sequence_state_is_kv_complete()
    }
}
