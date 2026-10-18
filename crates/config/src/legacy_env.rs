// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: LEGACY: the config crate's last process-environment reads, kept for callers that
//! do not receive the resolved settings yet: the model loaders (`glm_dsa_indexer_paged`,
//! `glm_vision_enabled`), the LoRA and model-config parsers they call, and the lm-head setup.
//! Each `*_from` form takes the variable lookup and reproduces the old reader exactly;
//! `legacy_env_tests` compares it with the settings field over the same values and lists every
//! difference. The server reads the settings fields instead ([`crate::PrefixCacheGate`]).
//!
//! Owner: config.
//! Invariants: the only module of this crate that reads the process environment; it goes away
//! once its callers take the resolved values.

/// 2026-10-17: One variable of the process environment; a missing or non-UTF-8 value is `None`,
/// as `std::env::var(..).ok()` was at each old reader.
fn process(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// 2026-10-09: Whether GLM-5.3's text-stack DSA layers keep their indexer rows in the KV
/// pool's blocks (paged) rather than a per-sequence buffer reserved at `--max-seq-len`. Paged
/// unless `METRALE_GLM_DSA_INDEXER_FLAT=1`. 2026-10-17: LEGACY, for the model loader; the serve
/// reads `PrefixCacheGate::glm_indexer_paged`.
pub fn glm_dsa_indexer_paged() -> bool {
    glm_dsa_indexer_paged_from(process)
}

pub(crate) fn glm_dsa_indexer_paged_from(var: impl Fn(&str) -> Option<String>) -> bool {
    var("METRALE_GLM_DSA_INDEXER_FLAT").as_deref() != Some("1")
}

/// 2026-09-26: Whether GLM-5.3's vision tower is enabled for this process
/// (`METRALE_GLM_VISION`, see [`crate::glm_vision_enabled_from`]); off by default.
///
/// The glm5_next parser (which leaves `config.vision` `None` when off) and
/// `Glm5NextWeightLoader::binds_vision_encoder` both call it, so they cannot disagree.
pub fn glm_vision_enabled() -> bool {
    glm_vision_enabled_env(process)
}

pub(crate) fn glm_vision_enabled_env(var: impl Fn(&str) -> Option<String>) -> bool {
    crate::glm_vision_enabled_from(var("METRALE_GLM_VISION").as_deref())
}

/// 2026-09-26: `METRALE_LORA_ALLOW_PARTIAL` (`1` or `true`, any case), read once per process:
/// load an adapter that names target modules the engine cannot apply, skipping those. Off by
/// default: a partly applied adapter cannot be told apart from a bad one.
pub(crate) fn allow_partial_targets() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| allow_partial_targets_from(process))
}

pub(crate) fn allow_partial_targets_from(var: impl Fn(&str) -> Option<String>) -> bool {
    var("METRALE_LORA_ALLOW_PARTIAL").is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// 2026-09-26: `METRALE_SOFTCAP_OVERRIDE`: the Gemma-4 final-logit softcap for an A/B, when it
/// parses as `f32`; an unparsable value is ignored.
pub(crate) fn softcap_override() -> Option<f32> {
    softcap_override_from(process)
}

pub(crate) fn softcap_override_from(var: impl Fn(&str) -> Option<String>) -> Option<f32> {
    var("METRALE_SOFTCAP_OVERRIDE").and_then(|v| v.parse::<f32>().ok())
}

/// 2026-09-26: `METRALE_GEMMA4_LMHEAD_NVFP4=1` quantizes a dense Gemma-4 lm-head (the rollback
/// of the BF16 default).
pub(crate) fn gemma4_lmhead_nvfp4() -> bool {
    gemma4_lmhead_nvfp4_from(process)
}

pub(crate) fn gemma4_lmhead_nvfp4_from(var: impl Fn(&str) -> Option<String>) -> bool {
    var("METRALE_GEMMA4_LMHEAD_NVFP4").as_deref() == Some("1")
}
