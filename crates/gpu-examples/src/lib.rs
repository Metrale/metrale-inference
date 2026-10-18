// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Holds the GPU microtests, parity gates and benches that used to live in
//! the `examples/` directory of `crates/model-arch`. The library is the examples' shared
//! startup: the settings a process runs under, resolved the way a serve resolves them.
//!
//! Owner: gpu-examples.
//! Invariants: not a default workspace member; build it with `-p metrale-gpu-examples`.

use metrale_settings::InferenceServerSettings;
use metrale_settings::sources::{CliLayer, RecipeLayer};

/// 2026-10-17: The settings an example runs under: the process environment, as a serve resolves
/// it (no CLI or recipe layer).
///
/// # Errors
/// The first lever whose value does not parse under its rule.
pub fn process_settings() -> anyhow::Result<InferenceServerSettings> {
    Ok(metrale_settings::resolve(
        &metrale_io_host_std::ProcessEnv,
        &CliLayer::empty(),
        &RecipeLayer::empty(),
    )?)
}

/// 2026-10-17: The GLM layer levers of [`process_settings`].
///
/// # Errors
/// As [`process_settings`], or a GLM row width out of range.
pub fn glm_levers() -> anyhow::Result<metrale_model_arch::glm5next_layer::GlmLevers> {
    metrale_model_arch::glm5next_layer::GlmLevers::from_settings(&process_settings()?)
}

/// 2026-10-17: The mHC `mix` scratch bound at [`glm_levers`]'s prefill width, the token rows the
/// text-stack loader allocates each site's `mix` and `done` for.
///
/// # Errors
/// As [`glm_levers`].
pub fn mhc_mix_tokens() -> anyhow::Result<usize> {
    Ok(metrale_model_arch::glm5next_mhc::mhc_mix_max_tokens(
        glm_levers()?.prefill_rows,
    ))
}

/// 2026-10-17: The `layers::ops` launcher levers of [`process_settings`].
///
/// # Errors
/// As [`process_settings`], and what `OpsLevers::from_settings` refuses.
pub fn ops_levers() -> anyhow::Result<metrale_model_layers::layers::ops::OpsLevers> {
    metrale_model_layers::layers::ops::OpsLevers::from_settings(&process_settings()?)
}
