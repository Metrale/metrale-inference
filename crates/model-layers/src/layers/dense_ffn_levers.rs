// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The dense FFN's dispatch levers, resolved once from the startup settings
//! ([`DenseFfnLevers::from_settings`]) and carried on `ModelLevers::ffn`. `new_with_activation`
//! copies the batch16 opt-in and the m16 tile onto the layer; the W8A8 prefill gate reads
//! `w8a16_only` from the forward context.
//!
//! Owner: model-layers (dense FFN).
//! Invariants:
//! - `from_settings(&defaults)` equals [`DenseFfnLevers::declared`]
//!   (`the_declared_defaults_are_the_old_unset_values`).
//! - `m16_tc_n_tile` is the wide tile at exactly `METRALE_FFN_M16_TC_NTILE=64` and the 32-wide
//!   tile at any other value or none, without failing the boot.

use metrale_settings::InferenceServerSettings;

use crate::layers::ops::{W8A16_GEMM_M16_N_TILE, W8A16_GEMM_M16_N_TILE_WIDE};

/// 2026-10-18: The dense FFN's levers. Each field names its variable and the reader it replaced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseFfnLevers {
    /// 2026-09-25: `METRALE_FFN_BATCH16` arms the 5..=32-row batch16 decode tier
    /// (`batch16_decode::batch16_plan`).
    pub batch16: bool,
    /// 2026-09-25: `METRALE_FFN_W8A16_ONLY` is present at any value: no projection takes the
    /// W8A8 prefill arm (`w8a8_prefill::w8a8_prefill_selected`).
    pub w8a16_only: bool,
    /// 2026-09-25: The m16 tier's CTA N width for the FFN arm (`m16_tc::m16_tc_kernel`).
    pub m16_tc_n_tile: u32,
}

impl DenseFfnLevers {
    /// 2026-10-18: The levers of the settings resolved once at startup.
    pub fn from_settings(s: &InferenceServerSettings) -> Self {
        let k = &s.kernels;
        Self {
            batch16: s.scheduler.ffn_batch16,
            w8a16_only: k.ffn_w8a16_only,
            m16_tc_n_tile: match k.ffn_m16_tc_ntile.as_deref() {
                Some("64") => W8A16_GEMM_M16_N_TILE_WIDE,
                _ => W8A16_GEMM_M16_N_TILE,
            },
        }
    }

    /// 2026-10-18: What [`Self::from_settings`] returns at the declared defaults.
    pub const fn declared() -> Self {
        Self {
            batch16: false,
            w8a16_only: false,
            m16_tc_n_tile: W8A16_GEMM_M16_N_TILE,
        }
    }
}

/// 2026-10-18: [`DenseFfnLevers::declared`], so `ModelLevers` can keep deriving `Default`.
impl Default for DenseFfnLevers {
    fn default() -> Self {
        Self::declared()
    }
}

#[cfg(test)]
#[path = "dense_ffn_levers_tests.rs"]
mod tests;
