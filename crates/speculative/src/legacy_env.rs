// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: LEGACY. Every process-environment read left in metrale-speculative, kept so that
//! the entry points the server still calls (`AdaptiveRung::from_env`, `DflashRung::configure`,
//! `MtpGate::new`, `mtp_gate::entry_pin_tokens_from_env`) keep their current meaning until the
//! server passes a resolved [`DrafterSettings`]. [`legacy_env_drafter_settings_from`] reproduces
//! the deleted readers' parses over an injected lookup, and `legacy_env_tests.rs` proves it equals
//! the `DrafterSettings` projection except for the listed deviations. The 2c-device-wire commit
//! deletes this module.
//!
//! Owner: speculative.
//! Invariants: the only module in metrale-speculative that reads the process environment.

use std::env::VarError;
use std::str::FromStr;

use metrale_settings::groups::device_drafter::{
    DflashRungTuning, DrafterSettings, MtpRungTuning, SPEC_ENTRY_PIN_BUILT_IN,
};

use crate::adaptive_rung::AdaptiveRung;
use crate::dflash_rung::DflashRung;
use crate::mtp_gate::MtpGate;

/// 2026-10-17: The deleted `mtp_gate::reprobe_tokens` default (`METRALE_MTP_GATE_REPROBE`).
const MTP_GATE_REPROBE_DEFAULT: usize = 256;
/// 2026-10-17: The deleted `mtp_gate::serial_refresh_tokens` default (`METRALE_MTP_GATE_REFRESH`).
const MTP_GATE_REFRESH_DEFAULT: usize = 1024;

/// 2026-10-17: The process-environment lookup the `legacy_env_*` readers are given in production.
fn process_var(name: &str) -> Result<String, VarError> {
    std::env::var(name)
}

/// 2026-10-17: The deleted `std::env::var_os(name).is_some()` reads: set to any value, `0` and
/// the empty string included.
fn present(var: &impl Fn(&str) -> Result<String, VarError>, name: &str) -> bool {
    !matches!(var(name), Err(VarError::NotPresent))
}

/// 2026-10-17: The deleted `env_usize`/`env_u32`/entry-pin parses, before their defaults.
fn parsed<T: FromStr>(var: &impl Fn(&str) -> Result<String, VarError>, name: &str) -> Option<T> {
    var(name).ok().and_then(|v| v.parse().ok())
}

/// 2026-10-17: The deleted `adaptive_rung::tunable`, as an override: `None` (the constant) unless
/// the value parses as a finite positive `f64`.
fn tunable(var: &impl Fn(&str) -> Result<String, VarError>, name: &str) -> Option<f64> {
    parsed::<f64>(var, name).filter(|v| v.is_finite() && *v > 0.0)
}

/// 2026-10-17: `DrafterSettings` as the deleted speculative readers parsed them: the presence
/// pins, the overrides ignored when they did not parse, the gate intervals and entry pin
/// defaulting on a bad value. Infallible, as they were. `mtp_ladder_set` here is only the
/// `METRALE_MTP_K_LADDER` presence; [`legacy_env_drafter_settings`] adds `--mtp-k-ladder`.
pub fn legacy_env_drafter_settings_from(
    var: impl Fn(&str) -> Result<String, VarError>,
) -> DrafterSettings {
    DrafterSettings {
        mtp_static_rung: present(&var, "METRALE_MTP_STATIC_RUNG"),
        mtp_ladder_set: present(&var, "METRALE_MTP_K_LADDER"),
        mtp_rung: MtpRungTuning {
            enter: tunable(&var, "METRALE_MTP_RUNG_ENTER"),
            leave: tunable(&var, "METRALE_MTP_RUNG_LEAVE"),
            alpha: tunable(&var, "METRALE_MTP_RUNG_ALPHA"),
            alpha_slow: tunable(&var, "METRALE_MTP_RUNG_ALPHA_SLOW"),
            probe_ticks: tunable(&var, "METRALE_MTP_RUNG_PROBE_TICKS"),
            p1_trigger: tunable(&var, "METRALE_MTP_RUNG_P1_TRIGGER"),
        },
        dflash_static_gamma: present(&var, "METRALE_DFLASH_STATIC_GAMMA"),
        dflash_gamma_resolver: present(&var, "METRALE_DFLASH_GAMMA_RESOLVER"),
        dflash_rung: DflashRungTuning {
            multi: parsed(&var, "METRALE_DFLASH_RUNG_MULTI"),
            narrow: parsed(&var, "METRALE_DFLASH_RUNG_NARROW"),
            wide: parsed(&var, "METRALE_DFLASH_RUNG_WIDE"),
            dwell: parsed(&var, "METRALE_DFLASH_RUNG_DWELL"),
            enter: parsed(&var, "METRALE_DFLASH_RUNG_ENTER"),
            leave: parsed(&var, "METRALE_DFLASH_RUNG_LEAVE"),
        },
        mtp_gate_reprobe: parsed(&var, "METRALE_MTP_GATE_REPROBE")
            .unwrap_or(MTP_GATE_REPROBE_DEFAULT),
        mtp_gate_refresh: parsed(&var, "METRALE_MTP_GATE_REFRESH")
            .unwrap_or(MTP_GATE_REFRESH_DEFAULT),
        spec_entry_pin_tokens: parsed(&var, "METRALE_SPEC_ENTRY_PIN")
            .unwrap_or(SPEC_ENTRY_PIN_BUILT_IN),
    }
}

/// 2026-10-17: [`legacy_env_drafter_settings_from`] on the process environment, with
/// `mtp_ladder_set` also on for a published `--mtp-k-ladder`, as the deleted
/// `RungParams::from_env` read it (`mtp_ladder_pinned`).
pub fn legacy_env_drafter_settings() -> DrafterSettings {
    let mut d = legacy_env_drafter_settings_from(process_var);
    d.mtp_ladder_set |= metrale_model_layers::speculative::mtp_ladder_pinned();
    d
}

impl AdaptiveRung {
    /// 2026-10-17: LEGACY. [`AdaptiveRung::with_settings`] at [`legacy_env_drafter_settings`].
    pub fn from_env() -> Self {
        Self::with_settings(&legacy_env_drafter_settings())
    }
}

impl DflashRung {
    /// 2026-10-17: LEGACY. [`DflashRung::configure_with_settings`] at
    /// [`legacy_env_drafter_settings`].
    pub fn configure(&self, cap_k: usize, explicit_flag: bool, woa_available: bool) {
        self.configure_with_settings(
            cap_k,
            explicit_flag,
            woa_available,
            &legacy_env_drafter_settings(),
        );
    }
}

impl MtpGate {
    /// 2026-10-17: LEGACY. [`MtpGate::with_settings`] at [`legacy_env_drafter_settings`].
    pub fn new(num_drafts: usize) -> Self {
        Self::with_settings(num_drafts, &legacy_env_drafter_settings())
    }
}

/// 2026-10-17: LEGACY. `METRALE_SPEC_ENTRY_PIN` as the scheduler stores it in
/// `SchedLevers::spec_entry_pin_tokens`: a `u32`, 8 when unset or unparseable.
pub fn entry_pin_tokens_from_env() -> u32 {
    legacy_env_drafter_settings_from(process_var).spec_entry_pin_tokens
}

#[cfg(test)]
#[path = "legacy_env_tests.rs"]
mod tests;
