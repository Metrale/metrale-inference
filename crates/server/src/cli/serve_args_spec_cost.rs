// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: `--spec-cost-model`, `--spec-cost-table`, `--spec-cost-calibration` and
//! `--spec-cost-slack`: the measured speculative-depth planner, flattened into
//! `ServeSchedulingArgs` after `--mtp-dcut-ratio`. It replaces `--mtp-k-ladder` and D-Cut
//! (`validate.rs` refuses them together) and is off by default.
//!
//! Owner: server CLI.
//! Invariants: the `///` text on the struct's fields is the `--help` output and
//! carries no date.

use clap::{Args, ValueEnum};

/// 2026-10-04: `--spec-cost-model`'s value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SpecCostModel {
    Off,
    Measured,
}

#[derive(Args, Debug, Clone, PartialEq)]
pub struct ServeSpecCostArgs {
    /// Plan MTP draft depth per batch width from a measured cost table (expected accepted
    /// tokens per joule), in place of `--mtp-k-ladder` and D-Cut (default: off). Requires
    /// `--spec-cost-table`, `--spec-cost-calibration` and `--spec-cost-slack`; refused together
    /// with `--mtp-k-ladder`, `--mtp-dcut-ratio` or `--draft-confidence-stop`. The serve computes
    /// its own circuit plan digests and drafter weights key at boot and refuses to start if the
    /// table or calibration was measured under a different one.
    #[arg(long, value_enum, default_value_t = SpecCostModel::Off)]
    pub spec_cost_model: SpecCostModel,

    /// The `met benchmark spec-cost-table --out` file this serve plans from. Required by
    /// `--spec-cost-model measured` (no implicit default path).
    #[arg(long, value_name = "PATH")]
    pub spec_cost_table: Option<std::path::PathBuf>,

    /// The matching `met benchmark spec-cost-table --calibration-out` file. Required by
    /// `--spec-cost-model measured` (no implicit default path).
    #[arg(long, value_name = "PATH")]
    pub spec_cost_calibration: Option<std::path::PathBuf>,

    /// The recipe id (as `met benchmark spec-cost-table --recipe` was given) this serve's
    /// command line is believed to match. Required by `--spec-cost-model measured`: this serve
    /// does not infer a recipe id from its own flags, so the match is asserted here and the boot
    /// check is only as good as this string — the table's key (circuit plan digests) is what
    /// actually catches a drift.
    #[arg(long, value_name = "ID")]
    pub spec_cost_recipe: Option<String>,

    /// How far below depth 1's tokens/ms the measured planner may trade for a better tokens per
    /// joule, in `0..1` (0.0 = never slower than depth 1). Required by `--spec-cost-model
    /// measured`: no implicit default, since silently picking one would hide a speed/energy
    /// trade the operator did not ask for.
    #[arg(long, value_name = "EPSILON")]
    pub spec_cost_slack: Option<f64>,
}
