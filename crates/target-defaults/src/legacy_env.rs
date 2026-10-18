// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: LEGACY: the one process-environment read the target-defaults crate makes at run time,
//! kept so production stays byte-identical until the dispatch receives the resolved
//! `DeviceSettings::attn_decode_gqa_pack` (the 2c-device-wire commit deletes this file).
//!
//! Owner: target-defaults.
//! Invariants: [`legacy_env_gqa_pack_from`] equals
//! `resolve_gqa_pack(DECODE_GQA_PACK_DECLARED, DeviceSettings::attn_decode_gqa_pack)` for every
//! value (`legacy_env_tests.rs`).

use std::env::VarError;

use crate::attn_splitk::{DECODE_GQA_PACK_DECLARED, resolve_gqa_pack};

/// 2026-10-17: The lever.
const GQA_PACK: &str = "METRALE_ATTN_DECODE_GQA_PACK";

/// 2026-10-17: Whether the packed decode kernels are armed, read from `var`.
pub fn legacy_env_gqa_pack_from(var: impl Fn(&str) -> Result<String, VarError>) -> bool {
    resolve_gqa_pack(DECODE_GQA_PACK_DECLARED, var(GQA_PACK).ok().as_deref())
}

/// 2026-10-17: [`legacy_env_gqa_pack_from`] over the process environment, resolved once: a fixed
/// answer keeps a captured CUDA graph consistent with later launches.
pub fn legacy_env_gqa_pack_enabled() -> bool {
    static ARMED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ARMED.get_or_init(|| legacy_env_gqa_pack_from(|k| std::env::var(k)))
}

#[cfg(test)]
#[path = "legacy_env_tests.rs"]
mod tests;
