// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The QSA indexer's prefill-selection levers, resolved once from the startup
//! settings ([`QsaLevers::from_settings`]) and carried on `AttnLevers::qsa`. The attention
//! layer passes them to [`QsaIndexer::prefill_select`](super::QsaIndexer::prefill_select).
//!
//! Owner: model-layers (QSA).
//! Invariants: `from_settings(&defaults)` equals [`QsaLevers::declared`]
//! (`the_declared_defaults_are_the_old_unset_values`).

use metrale_settings::InferenceServerSettings;

/// 2026-10-18: The prefill-selection levers. Each field names its variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QsaLevers {
    /// 2026-09-25: On unless `METRALE_QSA_NO_PREFILL_SELECT` is switched on; off makes
    /// `prefill_select` a no-op.
    pub prefill_select: bool,
    /// 2026-09-25: `METRALE_QSA_S2_DIAG`: keep the dense context of the last row before the
    /// overwrite, and log cosine(dense, selected) after.
    pub s2_diag: bool,
    /// 2026-09-25: `METRALE_QSA_SCORE_SCALAR`: the scalar row scorer even where the target
    /// ships the tensor-core one.
    pub score_scalar: bool,
}

impl QsaLevers {
    /// 2026-10-18: The levers of the settings resolved once at startup.
    pub fn from_settings(s: &InferenceServerSettings) -> Self {
        Self {
            prefill_select: !s.scheduler.qsa_no_prefill_select,
            s2_diag: s.telemetry.qsa_s2_diag,
            score_scalar: s.runtime.qsa_score_scalar,
        }
    }

    /// 2026-10-18: What [`Self::from_settings`] returns at the declared defaults.
    pub const fn declared() -> Self {
        Self {
            prefill_select: true,
            s2_diag: false,
            score_scalar: false,
        }
    }
}

/// 2026-10-18: [`QsaLevers::declared`], so `AttnLevers` can keep deriving `Default`.
impl Default for QsaLevers {
    fn default() -> Self {
        Self::declared()
    }
}

#[cfg(test)]
#[path = "qsa_levers_tests.rs"]
mod tests;
