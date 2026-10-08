// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: `--spec-cost-model measured`'s boot-time resolution: load the table and
//! calibration the operator pointed at, recompute this serve's own key (box class, circuit plan
//! digests, drafter weights), and refuse to start on any mismatch rather than plan from a stale
//! measurement.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - `Ok(None)` only when `--spec-cost-model` is `off`; every other outcome is `Ok(Some(_))` or
//!   an `Err` that names the mismatch, never a silently-ignored one.
//! - `validate_serve_args` already refused `measured` without its table, calibration, recipe id
//!   and slack; the `.context(...)` calls below are defence in depth for a caller that skips it
//!   (`met circuit diff`), not the primary error message an operator sees.

use anyhow::{Context, Result, bail};
use metrale_speculative::spec_cost::{
    AcceptanceCalibration, CostTable, DrafterKey, SpecCostState, TableKey,
};

use crate::cli::serve_args_spec_cost::{ServeSpecCostArgs, SpecCostModel};

/// 2026-10-04: `args`' resolved state, or `None` when the mode is off.
///
/// `drafter_weights_sha256` is this checkpoint's drafter key
/// (`WeightStore::drafter_weights_sha256`, computed while the weights were still in the store).
/// `mtp_vocab` and `mtp_quantization` are the serve's own flags, so the check is against what
/// THIS serve actually runs, not what the operator typed on the calibration run. The drafter has
/// no context toggle today (every serve reads prompt context); `context: true` reflects that one
/// behaviour, not a restated default — it must change if a toggle is ever added.
pub(crate) fn resolve(
    args: &ServeSpecCostArgs,
    drafter_weights_sha256: Option<&str>,
    mtp_vocab: u32,
    mtp_quantization: &str,
) -> Result<Option<SpecCostState>> {
    if args.spec_cost_model != SpecCostModel::Measured {
        return Ok(None);
    }
    let table_path = args
        .spec_cost_table
        .as_deref()
        .context("--spec-cost-model measured without --spec-cost-table")?;
    let cal_path = args
        .spec_cost_calibration
        .as_deref()
        .context("--spec-cost-model measured without --spec-cost-calibration")?;
    let recipe = args
        .spec_cost_recipe
        .as_deref()
        .context("--spec-cost-model measured without --spec-cost-recipe")?;
    let slack = args
        .spec_cost_slack
        .context("--spec-cost-model measured without --spec-cost-slack")?;

    let table_text = std::fs::read_to_string(table_path)
        .with_context(|| format!("reading {}", table_path.display()))?;
    let table = CostTable::parse(&table_text).map_err(anyhow::Error::msg)?;
    let cal_text = std::fs::read_to_string(cal_path)
        .with_context(|| format!("reading {}", cal_path.display()))?;
    let calibration = AcceptanceCalibration::parse(&cal_text).map_err(anyhow::Error::msg)?;

    let box_class = metrale_bench::hardware::Hardware::probe().gate_key();
    let plan_digests = metrale_model_layers::circuit_exec::spec_key::spec_cost_plan_digests(recipe)
        .with_context(|| format!("--spec-cost-recipe {recipe}"))?;
    let serve_key = TableKey {
        schema: metrale_speculative::spec_cost::SCHEMA,
        box_class,
        recipe: recipe.to_string(),
        plan_digests,
    };
    let mismatches = table.key.check(&serve_key);
    if !mismatches.is_empty() {
        bail!(
            "--spec-cost-table {} does not match this serve ({mismatches:?}); re-measure: met \
             benchmark spec-cost-table --recipe {recipe} --box-class {} ...",
            table_path.display(),
            serve_key.box_class,
        );
    }

    let Some(weights_sha256) = drafter_weights_sha256 else {
        bail!("--spec-cost-model measured needs a drafter (this checkpoint has no mtp.* weights)");
    };
    let serve_drafter = DrafterKey {
        weights_sha256: weights_sha256.to_string(),
        vocab: mtp_vocab as usize,
        quantization: mtp_quantization.to_string(),
        context: true,
    };
    if calibration.drafter != serve_drafter {
        bail!(
            "--spec-cost-calibration {} was fitted on a different drafter ({:?}), this serve's \
             is {serve_drafter:?}; re-measure: met benchmark spec-cost-table --recipe {recipe} ...",
            cal_path.display(),
            calibration.drafter,
        );
    }

    Ok(Some(SpecCostState {
        table,
        calibration,
        slack,
    }))
}

#[cfg(test)]
#[path = "spec_cost_boot_tests.rs"]
mod tests;
