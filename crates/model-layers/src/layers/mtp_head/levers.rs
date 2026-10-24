// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The MTP head's dispatch, sizing and diagnostic levers, resolved once from the
//! startup settings ([`MtpLevers::from_settings`]) and carried on `ModelLevers::mtp`. `new`
//! copies them onto the head (`MtpHead::levers`), so the projection dispatch and the drafter
//! prefill, which have no forward context, read the value resolved at startup.
//!
//! Owner: model-layers (MTP head).
//! Invariants:
//! - `from_settings(&defaults)` equals [`MtpLevers::declared`]
//!   (`the_declared_defaults_are_the_old_unset_values`).
//! - `METRALE_NO_DRAFTER_SMALL_M_TIER` keeps its reader's text predicate: exactly `1`.
//! - `METRALE_PROPOSE_META_STRIDE` keeps its reader's parse
//!   (`batch_caps::parse_stride_override`): trimmed, an unsigned byte count aligned and clamped,
//!   ignored when empty or unparseable.

use metrale_settings::InferenceServerSettings;

/// 2026-10-18: The MTP head's levers. Each field names its variable and the reader it
/// replaced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MtpLevers {
    /// 2026-09-25: `METRALE_MTP_PREFILL_PROFILE`: per-phase wall time of the drafter prefill
    /// (`prefill.rs`); each profiled phase synchronizes the stream.
    pub prefill_profile: bool,
    /// 2026-09-25: `METRALE_NO_DRAFTER_SMALL_M_TIER` is exactly `1`: the batched-GEMV tier is
    /// off (`row_dispatch::drafter_row_kernel`).
    pub small_m_tier_off: bool,
    /// 2026-09-25: `METRALE_MTP_KV_GEMV` is present at any value: the small-N projections stay
    /// on the per-row GEMV loop (`row_dispatch::drafter_row_kernel`).
    pub kv_gemv_pinned: bool,
    /// 2026-09-25: `METRALE_PROPOSE_META_STRIDE`, parsed, aligned and clamped; `None` keeps the
    /// computed stride (`batch_caps::propose_meta_stride`).
    pub propose_meta_stride: Option<usize>,
    /// 2026-09-25: On unless `METRALE_NO_MTP_LMHEAD_TGEMM` is present at any value: the
    /// transposed LM head for the tile GEMM is kept (`new.rs`).
    pub lmhead_tgemm: bool,
}

impl MtpLevers {
    /// 2026-10-18: The levers of the settings resolved once at startup.
    pub fn from_settings(s: &InferenceServerSettings) -> Self {
        let sp = &s.speculative;
        Self {
            prefill_profile: s.telemetry.mtp_prefill_profile,
            small_m_tier_off: sp.no_drafter_small_m_tier.as_deref() == Some("1"),
            kv_gemv_pinned: sp.mtp_kv_gemv,
            propose_meta_stride: super::batch_caps::parse_stride_override(
                sp.propose_meta_stride.as_deref(),
            ),
            lmhead_tgemm: !sp.no_mtp_lmhead_tgemm,
        }
    }

    /// 2026-10-18: What [`Self::from_settings`] returns at the declared defaults.
    pub const fn declared() -> Self {
        Self {
            prefill_profile: false,
            small_m_tier_off: false,
            kv_gemv_pinned: false,
            propose_meta_stride: None,
            lmhead_tgemm: true,
        }
    }
}

/// 2026-10-18: [`MtpLevers::declared`], so `ModelLevers` can keep deriving `Default`.
impl Default for MtpLevers {
    fn default() -> Self {
        Self::declared()
    }
}

#[cfg(test)]
#[path = "levers_tests.rs"]
mod tests;
