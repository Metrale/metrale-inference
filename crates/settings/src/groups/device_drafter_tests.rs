// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `DrafterSettings` properties that need no old reader. The equivalence with the
//! readers it replaces is proven where those readers live, against their `legacy_env_*` form
//! (speculative `legacy_env_tests.rs`).
//!
//! Owner: settings.
//! Invariants: never reads or writes the process environment.

use super::{DflashRungTuning, DrafterSettings, MtpRungTuning, SPEC_ENTRY_PIN_BUILT_IN};

#[test]
fn declared_settings_are_no_pin_no_override_and_the_built_in_intervals() {
    assert_eq!(
        DrafterSettings::declared().expect("declared defaults resolve"),
        DrafterSettings {
            mtp_static_rung: false,
            mtp_ladder_set: false,
            mtp_rung: MtpRungTuning::default(),
            dflash_static_gamma: false,
            dflash_gamma_resolver: false,
            dflash_rung: DflashRungTuning::default(),
            mtp_gate_reprobe: 256,
            mtp_gate_refresh: 1024,
            spec_entry_pin_tokens: SPEC_ENTRY_PIN_BUILT_IN,
        }
    );
}
