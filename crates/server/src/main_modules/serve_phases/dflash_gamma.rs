// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The DFlash drafter as the serve finds it (which checkpoint, its parsed config)
//! and the γ the serve runs, resolved once before the preflight reserve so the reserve, the
//! pools the build sizes and the scheduler read one value (`resolve_dflash_gamma`).
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - `load_dflash_drafter` finds the drafter through [`dflash_drafter_dir`] and
//!   [`read_dflash_config`], the functions [`apply_dflash_gamma`] resolves γ from.

use std::path::PathBuf;

use anyhow::{Context, Result};
use metrale_model_arch::weight_loader::DflashConfig;

use crate::cli;

/// 2026-09-30: The MODEL.toml `[dflash]` drafter of the serve's kernel target.
pub(crate) fn model_default_drafter(ptx_set: &metrale_kernels::TargetPtxSet) -> Option<String> {
    ptx_set.dflash.as_ref().map(|d| d.draft_model.to_string())
}

/// 2026-09-30: The drafter checkpoint: `--draft-model`, else `model_default` (MODEL.toml
/// `[dflash]`, [`model_default_drafter`]), resolved in the local cache.
pub(super) fn dflash_drafter_dir(
    args: &cli::ServeArgs,
    model_default: Option<String>,
) -> Result<PathBuf> {
    let drafter_id = args.draft_model.clone().or(model_default).context(
        "--dflash set but no drafter HF id provided: pass --draft-model <ID> \
             or use a target whose MODEL.toml has a [dflash] section",
    )?;
    tracing::info!("DFlash: resolving drafter '{drafter_id}'");
    crate::model_resolver::resolve_model_dir(&drafter_id, args.cache_dir.as_deref())
        .context("Failed to resolve DFlash drafter checkpoint")
}

/// 2026-09-30: The drafter's parsed `config.json`.
pub(super) fn read_dflash_config(drafter_dir: &std::path::Path) -> Result<DflashConfig> {
    let json = std::fs::read_to_string(drafter_dir.join("config.json")).with_context(|| {
        format!(
            "Failed to read drafter config.json at {}",
            drafter_dir.display()
        )
    })?;
    metrale_model_arch::weight_loader::dflash_loader::parse_dflash_config(&json)
}

/// 2026-09-30: Resolve this serve's γ into `args.dflash_gamma_resolved`: the pinned
/// `--dflash-gamma`, else `default_dflash_gamma` of the drafter's `effective_block_size`. A
/// no-op without `--dflash`.
pub(crate) fn apply_dflash_gamma(
    args: &mut cli::ServeArgs,
    model_default: Option<String>,
) -> Result<()> {
    if !args.dflash {
        return Ok(());
    }
    let block = match args.dflash_gamma {
        Some(_) => None,
        None => Some(
            read_dflash_config(&dflash_drafter_dir(args, model_default)?)?.effective_block_size(),
        ),
    };
    let gamma =
        metrale_model_layers::layers::qwen3_ssm::resolve_dflash_gamma(args.dflash_gamma, block);
    tracing::info!(
        "DFlash γ = {gamma} ({})",
        if args.dflash_gamma.is_some() {
            "--dflash-gamma"
        } else {
            "the drafter's block size + 2"
        }
    );
    args.dflash_gamma_resolved = Some(gamma);
    Ok(())
}

#[cfg(test)]
#[path = "dflash_gamma_tests.rs"]
mod dflash_gamma_tests;
