// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Boot refusal of speculative decoding over Mamba-2 layers.
//!
//! Owner: server startup (`met serve`).
//! Invariants: a serve with a speculative proposer never reaches weight load for a model
//! with Mamba-2 layers.

use anyhow::{Result, bail};
use metrale_config::ModelConfig;

use crate::cli;

/// 2026-09-29: Refuse every speculative proposer for a model with Mamba-2 layers.
///
/// Verify rollback does not cover Mamba-2 state: the checkpoint and rollback copies are
/// sized with the GatedDeltaNet formula (`linear_*` fields, all 0 for Nemotron-H, so 0
/// bytes), and the Mamba-2 decode writes no per-token h/conv intermediates. A rejected draft
/// would leave the recurrent state advanced past the accepted tokens and every later token
/// wrong, with no error. Refused until that rollback exists.
pub(super) fn refuse_speculation_over_mamba2(
    args: &cli::ServeArgs,
    config: &ModelConfig,
) -> Result<()> {
    if args.speculative_proposer_requested() && config.has_mamba2_layers() {
        bail!(
            "speculative decoding is not supported for `{}`: it has {} Mamba-2 layers, and \
             speculative verify cannot roll Mamba-2 state back after a rejected draft (the \
             rollback copies 0 bytes of it), so output would silently diverge. Serve without \
             --speculative, --self-speculative, --ngram-speculative and --dflash.",
            config.model_type,
            config.num_ssm_layers(),
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "mamba2_spec_tests.rs"]
mod tests;
