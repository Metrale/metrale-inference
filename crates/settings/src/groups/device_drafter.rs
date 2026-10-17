// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `DrafterSettings`, the levers metrale-speculative (the MTP adaptive rung and
//! gate, the DFlash gamma resolver) reads at run time, projected once from the resolved
//! [`InferenceServerSettings`] and passed to each controller when it is built (ROTI).
//!
//! A tuning override is `None` when unset, and the controller keeps its compiled constant; the
//! constants stay with the controllers they tune.
//!
//! Owner: settings.
//! Invariants: reads only `InferenceServerSettings`; every Text lever keeps the parse of the
//! reader it replaced.

use crate::groups::device::{DeviceSettingsError, ProjectionError, declared, non_negative};
use crate::resolve::InferenceServerSettings;

/// 2026-10-17: `METRALE_SPEC_ENTRY_PIN` when unset or not a `u32`.
pub const SPEC_ENTRY_PIN_BUILT_IN: u32 = 8;

/// 2026-10-17: MTP adaptive-rung overrides (`METRALE_MTP_RUNG_*`). A value that is not
/// positive is ignored, as the old reader ignored it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MtpRungTuning {
    pub enter: Option<f64>,
    pub leave: Option<f64>,
    pub alpha: Option<f64>,
    pub alpha_slow: Option<f64>,
    pub probe_ticks: Option<f64>,
    pub p1_trigger: Option<f64>,
}

/// 2026-10-17: DFlash gamma-resolver overrides (`METRALE_DFLASH_RUNG_*`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DflashRungTuning {
    pub multi: Option<usize>,
    pub narrow: Option<usize>,
    pub wide: Option<usize>,
    pub dwell: Option<usize>,
    pub enter: Option<u32>,
    pub leave: Option<u32>,
}

/// 2026-10-17: The levers metrale-speculative reads at run time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrafterSettings {
    /// 2026-10-17: `METRALE_MTP_STATIC_RUNG` (any value): pin the static ladder.
    pub mtp_static_rung: bool,
    /// 2026-10-17: An explicit ladder (`--mtp-k-ladder`, carried in the CLI layer, or
    /// `METRALE_MTP_K_LADDER` with any value) also pins the static ladder.
    pub mtp_ladder_set: bool,
    pub mtp_rung: MtpRungTuning,
    /// 2026-10-17: `METRALE_DFLASH_STATIC_GAMMA` (any value).
    pub dflash_static_gamma: bool,
    /// 2026-10-17: `METRALE_DFLASH_GAMMA_RESOLVER` (any value): keep the resolver on even with an
    /// explicit `--dflash-gamma`.
    pub dflash_gamma_resolver: bool,
    pub dflash_rung: DflashRungTuning,
    /// 2026-10-17: `METRALE_MTP_GATE_REPROBE`, tokens.
    pub mtp_gate_reprobe: usize,
    /// 2026-10-17: `METRALE_MTP_GATE_REFRESH`, tokens.
    pub mtp_gate_refresh: usize,
    /// 2026-10-17: `METRALE_SPEC_ENTRY_PIN`, tokens; 0 disables the pin.
    pub spec_entry_pin_tokens: u32,
}

fn positive(v: Option<f64>) -> Option<f64> {
    v.filter(|v| *v > 0.0)
}

fn width(env: &'static str, v: Option<i64>) -> Result<Option<usize>, ProjectionError> {
    v.map(|v| non_negative(env, v)).transpose()
}

fn text_u32(v: Option<&str>) -> Option<u32> {
    v.and_then(|v| v.parse().ok())
}

impl DrafterSettings {
    /// 2026-10-17: [`declared`], projected: what an empty environment, no CLI and no recipe give.
    ///
    /// # Errors
    /// A declared default that does not parse or project.
    pub fn declared() -> Result<Self, DeviceSettingsError> {
        Ok(Self::try_from(&declared()?)?)
    }
}

impl TryFrom<&InferenceServerSettings> for DrafterSettings {
    type Error = ProjectionError;

    fn try_from(s: &InferenceServerSettings) -> Result<Self, ProjectionError> {
        let sp = &s.speculative;
        Ok(Self {
            mtp_static_rung: sp.mtp_static_rung,
            mtp_ladder_set: sp.mtp_k_ladder.is_some(),
            mtp_rung: MtpRungTuning {
                enter: positive(sp.mtp_rung_enter),
                leave: positive(sp.mtp_rung_leave),
                alpha: positive(sp.mtp_rung_alpha),
                alpha_slow: positive(sp.mtp_rung_alpha_slow),
                probe_ticks: positive(sp.mtp_rung_probe_ticks),
                p1_trigger: positive(sp.mtp_rung_p1_trigger),
            },
            dflash_static_gamma: sp.dflash_static_gamma,
            dflash_gamma_resolver: sp.dflash_gamma_resolver,
            dflash_rung: DflashRungTuning {
                multi: width("METRALE_DFLASH_RUNG_MULTI", sp.dflash_rung_multi)?,
                narrow: width("METRALE_DFLASH_RUNG_NARROW", sp.dflash_rung_narrow)?,
                wide: width("METRALE_DFLASH_RUNG_WIDE", sp.dflash_rung_wide)?,
                dwell: width("METRALE_DFLASH_RUNG_DWELL", sp.dflash_rung_dwell)?,
                enter: text_u32(sp.dflash_rung_enter.as_deref()),
                leave: text_u32(sp.dflash_rung_leave.as_deref()),
            },
            mtp_gate_reprobe: non_negative("METRALE_MTP_GATE_REPROBE", sp.mtp_gate_reprobe)?,
            mtp_gate_refresh: non_negative("METRALE_MTP_GATE_REFRESH", sp.mtp_gate_refresh)?,
            spec_entry_pin_tokens: text_u32(sp.spec_entry_pin.as_deref())
                .unwrap_or(SPEC_ENTRY_PIN_BUILT_IN),
        })
    }
}

#[cfg(test)]
#[path = "device_drafter_tests.rs"]
mod tests;
