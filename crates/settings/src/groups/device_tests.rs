// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `DeviceSettings` properties: the declared defaults and the projection errors.
//!
//! Owner: settings.
//! Invariants: never reads or writes the process environment.

use metrale_error::{Error, ErrorCode};

use super::{DeviceSettings, row_default};
use crate::groups::equivalence::resolved;
use crate::groups::runtime::RuntimeSettings;

fn device(pairs: &[(&str, &str)]) -> Result<DeviceSettings, Error> {
    DeviceSettings::try_from(&resolved(pairs).expect("every value here is a valid Int"))
}

#[test]
fn a_value_the_device_type_cannot_hold_is_projection_out_of_range_naming_the_lever() {
    for (env, raw) in [
        ("METRALE_REDZONE", "-1"),
        ("METRALE_REDZONE_FILL", "256"),
        ("METRALE_REDZONE_FILL", "-1"),
        ("METRALE_REDZONE_MIN_IDX", "-1"),
        ("METRALE_REDZONE_TRACE_IDX", "-1"),
        ("METRALE_CUTLASS_WORKSPACE_MB", "-1"),
    ] {
        let e = device(&[(env, raw)]).expect_err("out of range");
        assert_eq!(e.code(), ErrorCode::ProjectionOutOfRange, "{env}={raw}");
        assert!(
            alloc::format!("{e}").starts_with(env),
            "{env}={raw}: the message names the lever: {e}"
        );
    }
}

#[test]
fn the_byte_and_workspace_rows_hold_the_built_in_values() {
    let d = device(&[]).expect("declared defaults project");
    assert_eq!(d.redzone.fill, 0xEE);
    assert_eq!(d.cutlass_workspace_bytes, 512 * 1024 * 1024);
    let d = device(&[
        ("METRALE_REDZONE_FILL", "255"),
        ("METRALE_CUTLASS_WORKSPACE_MB", "0"),
    ])
    .expect("in range");
    assert_eq!((d.redzone.fill, d.cutlass_workspace_bytes), (255, 0));
}

#[test]
fn redzone_bytes_round_up_to_sixteen() {
    let d = device(&[("METRALE_REDZONE", "17")]).expect("in range");
    assert_eq!(d.redzone.bytes, 32);
}

#[test]
fn row_default_reads_the_declared_default_of_the_named_row() {
    assert_eq!(
        row_default(RuntimeSettings::LEVERS, "METRALE_REDZONE_FILL"),
        "238"
    );
}

#[test]
fn declared_is_the_projection_of_an_empty_environment() {
    assert_eq!(
        DeviceSettings::declared().expect("declared defaults project"),
        device(&[]).expect("empty environment projects")
    );
}
