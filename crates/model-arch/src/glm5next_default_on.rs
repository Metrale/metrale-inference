// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The rule for a GLM lever that is on by default but cannot apply on some load: a
//! missing parent lever, a missing kernel, a shape the kernel does not take, another model type.
//! Left at its default it runs inert (the path the lever-off build takes) and says so once per
//! process; set explicitly (CLI, recipe or environment) its reader keeps its refusal where the
//! decision is made from the settings.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Inert means the lever-off path: never a third path.
//! - Each lever is reported at most once per process.

use std::sync::Mutex;

use metrale_settings::InferenceServerSettings;
use metrale_settings::provenance::Source;

/// 2026-10-22: Whether `env` was given a value (CLI, recipe or environment) rather than left at
/// its declared default.
pub fn explicit(s: &InferenceServerSettings, env: &str) -> bool {
    s.provenance
        .source_of(env)
        .is_some_and(|src| src != Source::Default)
}

/// 2026-10-22: One line per lever per process: `env` is on but runs inert here, because `why`.
pub fn report_inert(env: &'static str, why: &str) {
    static SEEN: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&env) {
        seen.push(env);
        tracing::info!("{env}: on, but inert on this load ({why}); the lever-off path runs");
    }
}

/// 2026-10-22: A child lever `env` (value `on`) that runs only under `parent` (`parent_on`): on
/// where the parent is; where it is not, off (inert, reported) when the child was left at its
/// default, and still on when it was set, so its reader refuses it as before.
pub fn under_parent(
    s: &InferenceServerSettings,
    env: &'static str,
    on: bool,
    parent_on: bool,
    parent: &str,
) -> bool {
    if !on || parent_on {
        return on;
    }
    if explicit(s, env) {
        return true;
    }
    report_inert(env, &format!("needs {parent}"));
    false
}
