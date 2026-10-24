// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `DrafterSettings` properties: the declared settings and the projection errors.
//!
//! Owner: settings.
//! Invariants: never reads or writes the process environment.

use metrale_error::ErrorCode;

use super::{DflashRungTuning, DrafterSettings, MtpRungTuning};
use crate::groups::equivalence::resolved;

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
            spec_entry_pin_tokens: 8,
        }
    );
}

#[test]
fn a_pin_or_width_the_controller_type_cannot_hold_is_projection_out_of_range() {
    for (env, raw) in [
        ("METRALE_SPEC_ENTRY_PIN", "-1"),
        ("METRALE_SPEC_ENTRY_PIN", "4294967296"),
        ("METRALE_MTP_GATE_REPROBE", "-1"),
        ("METRALE_DFLASH_RUNG_WIDE", "-1"),
    ] {
        let s = resolved(&[(env, raw)]).expect("a valid Int");
        let e = DrafterSettings::try_from(&s).expect_err("out of range");
        assert_eq!(e.code(), ErrorCode::ProjectionOutOfRange, "{env}={raw}");
    }
    let s = resolved(&[("METRALE_SPEC_ENTRY_PIN", "4294967295")]).expect("a valid Int");
    assert_eq!(
        DrafterSettings::try_from(&s).map(|d| d.spec_entry_pin_tokens),
        Ok(u32::MAX)
    );
}
