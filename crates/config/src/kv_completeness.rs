// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The two KV-completeness gates: whether the radix prefix cache and the
//! `--swap-space-gb` swap-out may treat a sequence's KV blocks as its whole state.
//!
//! Owner: config.
//! Invariants:
//! - Without `METRALE_GLM53_PREFIX_CACHE_UNPROVEN`, the two gates give the same answer for
//!   every model but `glm5_next` / `glm5_next_text` with a paged indexer cache.
//! - That switch opens only the prefix-cache gate, and only for `glm5_next` / `glm5_next_text`.
//! - 2026-10-09: A paged GLM indexer cache ([`glm_dsa_indexer_paged`]) opens the prefix-cache
//!   gate for `glm5_next` / `glm5_next_text` and leaves the swap-out gate closed.

use super::ModelConfig;

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
    /// `METRALE_GLM53_PREFIX_CACHE_UNPROVEN` (any value, `0` included) also opens it for
    /// `glm5_next`. `kv_only_swap_out_is_safe` ignores that switch: the swap image cannot
    /// carry the DSA rows (see there).
    pub fn kv_only_prefix_cache_is_safe(&self) -> bool {
        self.kv_only_prefix_cache_is_safe_with(Self::glm53_prefix_cache_validation_env())
    }

    /// 2026-09-26: Env-free core of [`Self::kv_only_prefix_cache_is_safe`]; the override opens
    /// only `glm5_next` / `glm5_next_text`.
    pub(crate) fn kv_only_prefix_cache_is_safe_with(&self, validation_override: bool) -> bool {
        self.kv_only_prefix_cache_is_safe_given(validation_override, glm_dsa_indexer_paged())
    }

    /// 2026-10-09: Env-free core with the indexer mode as an argument.
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

    fn glm53_prefix_cache_validation_env() -> bool {
        std::env::var_os("METRALE_GLM53_PREFIX_CACHE_UNPROVEN").is_some()
    }

    /// 2026-09-26: Whether a sequence may be swapped out to the `--swap-space-gb` pool and
    /// restored from it; `resolve_swap_space_gb` reads it. `save_sequence_state_dispatch`
    /// writes the KV blocks and the `SsmLayerState` of each `LayerType::LinearAttention`
    /// layer and nothing else, so any other per-sequence state would not survive the swap.
    pub fn kv_only_swap_out_is_safe(&self) -> bool {
        self.per_sequence_state_is_kv_complete()
    }
}

/// 2026-10-09: Whether GLM-5.3's text-stack DSA layers keep their indexer rows in the KV
/// pool's blocks (paged) rather than a per-sequence buffer reserved at `--max-seq-len`. Paged
/// unless `METRALE_GLM_DSA_INDEXER_FLAT=1`. The loader, the serve's per-sequence reserve and
/// the prefix-cache gate all read this one function.
pub fn glm_dsa_indexer_paged() -> bool {
    std::env::var("METRALE_GLM_DSA_INDEXER_FLAT").as_deref() != Ok("1")
}
