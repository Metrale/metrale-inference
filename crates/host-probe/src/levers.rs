// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The harness's lever values, resolved once by the binary and passed down.
//!
//! [`BenchLevers`] is the projection of the resolved settings that the benchmark harness reads:
//! where the artifact store and the warm cargo dirs live, the gate commit override, the Python
//! override, the agentic request switches and the hardware precheck switches. The binary builds
//! it at its entry point ([`BenchLevers::resolve`], or [`BenchLevers::from_settings`] when it
//! already holds the serve settings) and hands it to the executor and the CLI paths; nothing
//! reads the process environment for these levers.
//!
//! It lives here, not in metrale-bench, because `met serve` without the `bench` feature still
//! resolves it at boot and finds the recipe store from it ([`crate::artifacts`]).
//! metrale-bench re-exports this module as `metrale_bench::levers`.
//!
//! Owner: host-probe.
//! Invariants:
//! - Text and path levers keep their value verbatim, `""` included; each reader keeps its own
//!   rule for an empty value.
//! - The environment is read only through the [`Env`] port the caller supplies.

use anyhow::{Context, Result};
use metrale_settings::Env;
use metrale_settings::groups::harness::HarnessSettings;
use metrale_settings::provenance::Provenance;
use metrale_settings::sources::{CliLayer, RecipeLayer, Sources};
use metrale_settings::{InferenceServerSettings, resolve};

/// 2026-10-17: The value of `METRALE_AGENTIC_SAMPLING` that drops the greedy pins.
pub const MODEL_CARD_SAMPLING: &str = "model-card";

/// 2026-10-17: The harness levers of one process; see the module doc.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BenchLevers {
    /// 2026-10-17: `METRALE_PYTHON`: when set, the only interpreter `find_python` tries.
    pub python: Option<String>,
    /// 2026-10-17: `METRALE_GATE_SHA`: the commit a gate record names when the checkout has no
    /// git metadata. `git_sha` refuses a blank value.
    pub gate_sha: Option<String>,
    /// 2026-10-17: `METRALE_HOME`: the artifact store root. `MetraleHome` refuses `""`.
    pub home: Option<String>,
    /// 2026-10-17: `HOME`, the base of `$HOME/.metrale` and `$HOME/.cargo/metrale-warm-*`.
    pub user_home: Option<String>,
    /// 2026-10-17: `METRALE_AGENTIC_PRESERVE_THINKING`.
    pub preserve_thinking: bool,
    /// 2026-10-17: `METRALE_AGENTIC_SAMPLING` is exactly [`MODEL_CARD_SAMPLING`]: the agentic
    /// requests carry no temperature or seed.
    pub model_card_sampling: bool,
    /// 2026-10-17: `METRALE_WARM_TARGET_DIR`; `""` counts as unset.
    pub warm_target_dir: Option<String>,
    /// 2026-10-17: `METRALE_WARM_TEMPLATE_DIR`; `""` counts as unset.
    pub warm_template_dir: Option<String>,
    /// 2026-10-17: `METRALE_NO_HW_PRECHECK` under [`hw_switch`]: a precheck refusal becomes a
    /// warning.
    pub hw_kill_switch: bool,
    /// 2026-10-17: `METRALE_HW_TEMP_GATE` under [`hw_switch`]: the absolute temperature
    /// ceilings refuse instead of warn.
    pub hw_temp_gate: bool,
}

impl BenchLevers {
    /// 2026-10-17: Resolve the runtime settings and the harness group from `env` alone (no CLI
    /// or recipe layer), then project them. For a process that is not a serve.
    ///
    /// # Errors
    /// The first malformed lever in any group, or a `HOME` that is not UTF-8.
    pub fn resolve(env: &impl Env) -> Result<Self> {
        let settings = resolve(env, &CliLayer::empty(), &RecipeLayer::empty())
            .map_err(|e| anyhow::anyhow!("settings: {e}"))?;
        let harness = resolve_harness(env)?;
        Self::from_settings(&settings, &harness, env)
    }

    /// 2026-10-17: Project settings the caller already resolved; `env` supplies only `HOME`.
    ///
    /// # Errors
    /// A `HOME` that is not UTF-8.
    pub fn from_settings(
        settings: &InferenceServerSettings,
        harness: &HarnessSettings,
        env: &impl Env,
    ) -> Result<Self> {
        let user_home = env.var("HOME").context("reading HOME")?;
        Ok(Self {
            python: settings.runtime.python.clone(),
            gate_sha: settings.kernels.gate_sha.clone(),
            home: harness.home.clone(),
            user_home,
            preserve_thinking: settings.sampling.agentic_preserve_thinking,
            model_card_sampling: settings.server.agentic_sampling.as_deref()
                == Some(MODEL_CARD_SAMPLING),
            warm_target_dir: settings.storage.warm_target_dir.clone(),
            warm_template_dir: harness.warm_template_dir.clone(),
            hw_kill_switch: hw_switch(settings.runtime.no_hw_precheck.as_deref()),
            hw_temp_gate: hw_switch(settings.sampling.hw_temp_gate.as_deref()),
        })
    }
}

/// 2026-10-17: The `harness` group from `env` alone; the one resolver of that group.
///
/// # Errors
/// The first harness lever that fails to resolve.
pub fn resolve_harness(env: &impl Env) -> Result<HarnessSettings> {
    let (cli, recipe) = (CliLayer::empty(), RecipeLayer::empty());
    HarnessSettings::resolve(
        &Sources {
            env,
            cli: &cli,
            recipe: &recipe,
        },
        &mut Provenance::default(),
    )
    .map_err(|e| anyhow::anyhow!("harness settings: {e}"))
}

/// 2026-09-26: `1`/`true`/`yes`, trimmed and case-insensitive. Anything else, including an
/// empty value, is off. The grammar of the two hardware precheck switches.
#[must_use]
pub fn hw_switch(value: Option<&str>) -> bool {
    value.is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
}
