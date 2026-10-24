// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Startup gate that refuses an SSM snapshot tier on a model with no recurrent state.
//!
//! Owner: model-engine SSM tier.
//! Invariants:
//! - With no tier var requested, or on a model where
//!   [`ModelConfig::has_recurrent_state`] is true, the gate returns `Ok(())`.
//! - Otherwise it returns an error that names the model type and every var found set.

use anyhow::{Result, bail};
use metrale_config::ModelConfig;

use super::SsmTierLevers;

/// 2026-09-25: Fail when any tier var is requested ([`SsmTierLevers::requested_tier_vars`]:
/// any value of a presence or text row, or the `METRALE_SSM_SWAP` switch on) and the model
/// has no recurrent state. The namespace overrides `METRALE_SSM_SWAP_NS` and
/// `METRALE_SSM_DECODE_NS` select no tier and are not listed. `TransformerModel::new` calls
/// it before it builds the SSM snapshot pool, so before any tier store or peer connect.
pub(crate) fn ensure_ssm_tier_capability(
    config: &ModelConfig,
    levers: &SsmTierLevers,
) -> Result<()> {
    ensure_ssm_tier_capability_from(config, &levers.requested_tier_vars())
}

/// 2026-09-25: Env-free core of [`ensure_ssm_tier_capability`]; `set_vars` is the
/// list of tier vars found set.
pub(crate) fn ensure_ssm_tier_capability_from(
    config: &ModelConfig,
    set_vars: &[&str],
) -> Result<()> {
    if set_vars.is_empty() || config.has_recurrent_state() {
        return Ok(());
    }
    bail!(
        "model '{}' has no recurrent state (num_ssm_layers=0) — the SSM snapshot \
         tier cannot be populated by this model. Unset {} for this serve (fleet-wide \
         env files now need per-model curation; a tier request on an incapable model \
         was previously a SILENT no-op and is now a startup error).",
        config.model_type,
        set_vars.join(", "),
    );
}

#[cfg(test)]
#[path = "capability_tests.rs"]
mod tests;
