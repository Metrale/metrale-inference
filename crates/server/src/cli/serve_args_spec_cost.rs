// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: `--spec-cost-model`, `--spec-cost-table`, `--spec-cost-calibration` and
//! `--spec-cost-slack`: the measured speculative-depth planner, flattened into
//! `ServeSchedulingArgs` after `--mtp-dcut-ratio`. It replaces `--mtp-k-ladder` and D-Cut
//! (`validate.rs` refuses them together) and is off by default. 2026-10-10: and
//! `--spec-objective`, the speculation controller's objective, which has no default.
//!
//! Owner: server CLI.
//! Invariants: the `///` text on the struct's fields is the `--help` output and
//! carries no date.

use clap::{Args, ValueEnum};

/// 2026-10-10: `--spec-objective`'s value: what the speculation controller maximises
/// (`metrale_speculative::spec_ctl::decide::Objective`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SpecObjective {
    Latency,
    Throughput,
    Energy,
}

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

    /// What the speculation controller maximises when it chooses, each step, plain decode or a
    /// draft depth: `latency` (the slowest stream's tokens per ms), `throughput` (all streams'
    /// tokens per ms) or `energy` (tokens per joule, no slower than one draft by more than
    /// `--spec-cost-slack`; needs `--spec-cost-model measured`, the only cost source with
    /// joules). Required whenever MTP or DFlash runs without `--mtp-gate force`; no default.
    #[arg(long, value_enum)]
    pub spec_objective: Option<SpecObjective>,
}

impl ServeSpecCostArgs {
    /// 2026-10-10: `--spec-objective` as the controller's objective. `energy` takes
    /// `--spec-cost-slack` as its floor slack against one draft (`validate_serve_args` refuses
    /// `energy` without `--spec-cost-model measured`, which requires the slack).
    pub fn objective(&self) -> Option<metrale_speculative::spec_ctl::decide::Objective> {
        use metrale_speculative::spec_ctl::decide::{FloorRef, Objective};
        self.spec_objective.map(|o| match o {
            SpecObjective::Latency => Objective::Latency,
            SpecObjective::Throughput => Objective::Throughput,
            SpecObjective::Energy => Objective::Energy {
                slack: self
                    .spec_cost_slack
                    .expect("validate_serve_args: --spec-objective energy needs --spec-cost-slack"),
                floor: FloorRef::Depth(1),
            },
        })
    }
}
