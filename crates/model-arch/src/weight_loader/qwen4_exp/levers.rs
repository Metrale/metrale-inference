// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The qwen4_exp loader levers, resolved from the startup settings once per load.
//!
//! Owner: model-arch weight loader (qwen4_exp).
//! Invariants: `load_layers` resolves this once before its layer loop; each field reads one
//! settings field with the parse its old reader applied.

use metrale_settings::InferenceServerSettings;

use super::super::shared_levers::grouped_cutlass;

/// 2026-10-17: The PLE scratch rows when `METRALE_PLE_MAX_TOKENS` is unset or unparseable.
const PLE_MAX_TOKENS: usize = 2048;
/// 2026-10-17: The PLE row-cache slots when `METRALE_PLE_CACHE_SLOTS` is unset or unparseable.
const PLE_CACHE_SLOTS: usize = 65536;

fn parsed_or(v: Option<&str>, fallback: usize) -> usize {
    v.and_then(|v| v.parse().ok()).unwrap_or(fallback)
}

/// 2026-10-17: The levers the qwen4_exp loader and its builders read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Qwen4ExpLevers {
    /// 2026-10-17: `METRALE_PLE_MAX_TOKENS`: the largest prefill chunk the PLE scratch holds.
    pub(super) ple_max_tokens: usize,
    /// 2026-10-17: `METRALE_PLE_CACHE_SLOTS`: rows the PLE row cache holds.
    pub(super) ple_cache_slots: usize,
    /// 2026-10-17: `METRALE_QWEN4EXP_NO_PLE`: no PLE layer is loaded (bisection only).
    pub(super) no_ple: bool,
    /// 2026-10-17: `METRALE_QWEN4EXP_BF16_GDN`: GDN projections stay BF16.
    pub(super) bf16_gdn: bool,
    /// 2026-10-17: `METRALE_HOLO_MOE_GROUPED_CUTLASS` (`shared_levers::grouped_cutlass`).
    pub(super) grouped_cutlass: bool,
    /// 2026-10-17: `METRALE_QSA_DISABLE`: no QSA indexer is attached.
    pub(super) qsa_disable: bool,
}

impl Qwen4ExpLevers {
    /// 2026-10-17: Resolve from the startup settings.
    pub(super) fn from_settings(s: &InferenceServerSettings) -> Self {
        Self {
            ple_max_tokens: parsed_or(s.scheduler.ple_max_tokens.as_deref(), PLE_MAX_TOKENS),
            ple_cache_slots: parsed_or(s.cache.ple_cache_slots.as_deref(), PLE_CACHE_SLOTS),
            no_ple: s.model.qwen4exp_no_ple,
            bf16_gdn: s.kernels.qwen4exp_bf16_gdn,
            grouped_cutlass: grouped_cutlass(s),
            qsa_disable: s.runtime.qsa_disable,
        }
    }
}

#[cfg(test)]
#[path = "levers_tests.rs"]
mod tests;
