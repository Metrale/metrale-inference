// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The legacy GQA-pack reader against the settings projection.
//!
//! Owner: target-defaults.
//! Invariants: never reads or writes the process environment.

use metrale_settings::groups::device::DeviceSettings;
use metrale_settings::groups::equivalence::{VALUES, check_values, var_one};

use super::*;

#[test]
fn gqa_pack_projection_resolves_like_the_legacy_reader() {
    let values = [VALUES.as_slice(), &[Some("on"), Some("no"), Some(" 1")]].concat();
    check_values(
        GQA_PACK,
        &values,
        |v| legacy_env_gqa_pack_from(var_one(GQA_PACK, v)),
        |s| {
            DeviceSettings::try_from(s).map(|d| {
                resolve_gqa_pack(DECODE_GQA_PACK_DECLARED, d.attn_decode_gqa_pack.as_deref())
            })
        },
        &[],
    );
}
