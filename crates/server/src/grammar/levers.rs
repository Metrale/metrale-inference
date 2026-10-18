// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: [`GrammarLevers`]: the grammar engine's switches, resolved once per model load
//! from the load's settings and carried on [`super::GrammarEngine`].
//!
//! Owner: server (grammar).
//! Invariants:
//! - 2026-10-17: No grammar code reads the environment (ROTI). A compile reads its switches
//!   from the engine it runs on; the engine got them from the settings when it was built.

use metrale_settings::InferenceServerSettings;

use super::mask_cache::cache_enabled_from;
use super::prewarm::overlap_from_env;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrammarLevers {
    /// 2026-10-17: `METRALE_GRAMMAR_ALLOW_EMPTY_VALUE`: an XML parameter value may be empty.
    pub allow_empty_value: bool,
    /// 2026-10-17: `METRALE_GRAMMAR_FORCE_CLOSE`: once a value holds the whole close prefix,
    /// only the close may follow (the ladder drops its deepest arm).
    pub force_close: bool,
    /// 2026-10-17: `METRALE_GRAMMAR_VALUE_HARDEN`: a value's `rest` repeats at most
    /// `VALUE_REST_MAX_REPEAT` times instead of without bound.
    pub value_harden: bool,
    /// 2026-10-17: `METRALE_TOOL_SHORT_TRIGGER`: qwen3_coder with triggers has the one
    /// trigger `<tool_call>` instead of one `<tool_call>\n<function=NAME>` per tool.
    pub short_trigger: bool,
    /// 2026-10-17: Off when `METRALE_GRAMMAR_ASYNC_PREWARM` is `0|false|off|no`: the mask
    /// prewarm runs on the request's thread instead of its own.
    pub async_prewarm: bool,
    /// 2026-10-17: Off when `METRALE_GRAMMAR_CACHE` is `0|false|off|no`: no on-disk mask
    /// cache is loaded or saved.
    pub mask_cache: bool,
    /// 2026-10-17: `METRALE_GRAMMAR_CACHE_DIR`: where the mask cache lives instead of under
    /// the model directory (a blank value is ignored, see `mask_cache::snapshot_path`).
    pub mask_cache_dir: Option<String>,
}

impl GrammarLevers {
    /// 2026-10-17: Every switch as it resolves with its variable unset.
    pub const UNSET: Self = Self {
        allow_empty_value: false,
        force_close: false,
        value_harden: false,
        short_trigger: false,
        async_prewarm: true,
        mask_cache: true,
        mask_cache_dir: None,
    };

    /// 2026-10-17: The switches one load's settings select. The two `Text` levers keep the
    /// readers they had (`overlap_from_env`, `cache_enabled_from`), so their values parse as
    /// before.
    pub fn from_settings(settings: &InferenceServerSettings) -> Self {
        let sampling = &settings.sampling;
        Self {
            allow_empty_value: sampling.grammar_allow_empty_value,
            force_close: sampling.grammar_force_close,
            value_harden: sampling.grammar_value_harden,
            short_trigger: settings.scheduler.tool_short_trigger,
            async_prewarm: overlap_from_env(sampling.grammar_async_prewarm.as_deref()),
            mask_cache: cache_enabled_from(sampling.grammar_cache.as_deref()),
            mask_cache_dir: sampling.grammar_cache_dir.clone(),
        }
    }
}
