// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The in-memory resolve the device-side projection tests build their settings with.
//!
//! Owner: settings.
//! Invariants: never reads or writes the process environment.

use metrale_io_host_mem::MapEnv;

use crate::resolve::InferenceServerSettings;
use crate::sources::{CliLayer, RecipeLayer};
use crate::{SettingsError, resolve};

/// 2026-10-17: Every runtime lever resolved from `pairs` alone (no CLI, no recipe).
///
/// # Errors
/// The resolver's error for a malformed value.
pub fn resolved(pairs: &[(&str, &str)]) -> Result<InferenceServerSettings, SettingsError> {
    let env = MapEnv::new(pairs.iter().copied());
    resolve(&env, &CliLayer::empty(), &RecipeLayer::empty())
}
