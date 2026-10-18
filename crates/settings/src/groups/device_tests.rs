// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `DeviceSettings` properties that need no old reader. The equivalence with the
//! readers it replaces is proven where those readers live, against their `legacy_env_*` forms
//! (gpu-runtime and kernels `legacy_env_tests.rs`).
//!
//! Owner: settings.
//! Invariants: never reads or writes the process environment.

use super::{DeviceSettings, ProjectionError};
use crate::groups::equivalence::resolved;

fn device(s: &crate::InferenceServerSettings) -> Result<DeviceSettings, ProjectionError> {
    DeviceSettings::try_from(s)
}

#[test]
fn a_negative_redzone_is_a_projection_error_naming_the_lever() {
    let s = resolved(&[("METRALE_REDZONE", "-1")]).expect("-1 is a valid Int");
    match device(&s) {
        Err(ProjectionError::OutOfRange { env, .. }) => assert_eq!(env, "METRALE_REDZONE"),
        other => panic!("expected OutOfRange, got {other:?}"),
    }
}

#[test]
fn declared_is_the_projection_of_an_empty_environment() {
    let empty = resolved(&[]).expect("declared defaults resolve");
    assert_eq!(
        DeviceSettings::declared().expect("declared defaults project"),
        device(&empty).expect("empty environment projects")
    );
}
