// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: `--spec-cost-model measured`'s validation, split out of `validate.rs` verbatim to
//! keep it under the 500-line cap.
//!
//! Owner: server CLI.
//! Invariants: the `///` text on each violation is the operator-facing message; see
//! `validate.rs` for the overall contract.

use super::ServeArgs;
use super::validate::violation::Violation;

/// 2026-10-10: `--spec-cost-model measured`'s checks and `--spec-objective`'s.
pub(super) fn check(args: &ServeArgs, v: &mut Vec<Violation>) {
    check_objective(args, v);
    check_measured(args, v);
}

/// 2026-10-10: `--spec-objective` is required whenever the speculation controller runs (MTP
/// or DFlash without `--mtp-gate force`, which resolves `METRALE_MTP_GATE_FORCE` and
/// `--hermetic` too); a recipe's objective under a forcing override is simply unused. `energy`
/// needs the only cost source that measures joules.
fn check_objective(args: &ServeArgs, v: &mut Vec<Violation>) {
    use crate::cli::serve_args_spec_cost::{SpecCostModel, SpecObjective};
    let forced = crate::scheduler::levers::resolve_mtp_gate_force(args.mtp_gate_force());
    let controlled = (args.speculative || args.dflash) && !forced;
    if controlled && args.spec_cost.spec_objective.is_none() {
        v.push(Violation::new(
            "speculation is on without --spec-objective.",
            "each step the speculation controller chooses plain decode or a draft depth by an \
             objective; picking one silently would choose between latency, throughput and \
             energy for the operator.",
            "pass --spec-objective latency|throughput|energy, or --mtp-gate force to run \
             every eligible step speculatively.",
        ));
    }
    if args.spec_cost.spec_objective == Some(SpecObjective::Energy)
        && args.spec_cost.spec_cost_model != SpecCostModel::Measured
    {
        v.push(Violation::new(
            "--spec-objective energy is set without --spec-cost-model measured.",
            "the energy objective needs joules per step, and only the measured cost table \
             carries them.",
            "add --spec-cost-model measured (with its table, calibration and slack), or pick \
             latency or throughput.",
        ));
    }
}

/// 2026-10-04: `--spec-cost-model measured` needs its table, calibration and slack (PCND: no
/// implicit default path or epsilon), needs MTP, and replaces the static K-ladder and D-Cut
/// rather than composing with them.
fn check_measured(args: &ServeArgs, v: &mut Vec<Violation>) {
    if args.spec_cost.spec_cost_model != crate::cli::serve_args_spec_cost::SpecCostModel::Measured {
        return;
    }
    if args.spec_cost.spec_cost_table.is_none() {
        v.push(Violation::new(
            "--spec-cost-model measured is set without --spec-cost-table.",
            "the measured planner plans from a table measured by `met benchmark \
             spec-cost-table`; there is no implicit default path.",
            "pass --spec-cost-table <path>, or drop --spec-cost-model.",
        ));
    }
    if args.spec_cost.spec_cost_calibration.is_none() {
        v.push(Violation::new(
            "--spec-cost-model measured is set without --spec-cost-calibration.",
            "the planner needs the drafter's confidence-to-acceptance calibration from the \
             same `met benchmark spec-cost-table` run; there is no implicit default path.",
            "pass --spec-cost-calibration <path>, or drop --spec-cost-model.",
        ));
    }
    if args.spec_cost.spec_cost_recipe.is_none() {
        v.push(Violation::new(
            "--spec-cost-model measured is set without --spec-cost-recipe.",
            "the boot check needs the recipe id the table was measured under, to recompute \
             its circuit plan digests; this serve does not infer a recipe id from its own \
             flags.",
            "pass --spec-cost-recipe <id>, or drop --spec-cost-model.",
        ));
    }
    match args.spec_cost.spec_cost_slack {
        None => v.push(Violation::new(
            "--spec-cost-model measured is set without --spec-cost-slack.",
            "slack trades speed for energy (0.0 = never slower than depth 1); a silent \
             default would pick that trade for the operator.",
            "pass --spec-cost-slack <epsilon> (0.0 keeps depth 1's speed), or drop \
             --spec-cost-model.",
        )),
        Some(s) if !(0.0..1.0).contains(&s) => v.push(Violation::new(
            format!("--spec-cost-slack {s} is outside [0, 1)."),
            "slack is the fraction of depth 1's tokens/ms the planner may give up; 1.0 or \
             more would admit an arbitrarily slow depth.",
            "pass a value from 0 (inclusive) to 1 (exclusive).",
        )),
        Some(_) => {}
    }
    if !args.speculative || args.dflash {
        v.push(Violation::new(
            "--spec-cost-model measured needs --speculative (MTP) and is not used with \
             --dflash.",
            "the planner chooses MTP draft depth per batch width; without MTP there is no \
             depth to choose, and a DFlash drafter proposes its whole block in one pass.",
            "add --speculative, or drop --spec-cost-model.",
        ));
    }
    if args.mtp_shape.mtp_k_ladder.is_some() {
        v.push(Violation::new(
            "--spec-cost-model measured is set together with --mtp-k-ladder.",
            "the measured planner replaces the static K-ladder; the two cannot both choose \
             the propose depth.",
            "drop --mtp-k-ladder, or drop --spec-cost-model.",
        ));
    }
    if args.mtp_shape.mtp_dcut_ratio.is_some() {
        v.push(Violation::new(
            "--spec-cost-model measured is set together with --mtp-dcut-ratio.",
            "the measured planner replaces D-Cut's per-sequence pruning with its own \
             tokens-per-joule search; the two cannot both choose the verify depth.",
            "drop --mtp-dcut-ratio, or drop --spec-cost-model.",
        ));
    }
    if args.mtp_draft.draft_confidence_stop.is_some() {
        v.push(Violation::new(
            "--spec-cost-model measured is set together with --draft-confidence-stop.",
            "both choose how many of a sequence's drafts are verified; the two cannot both \
             decide it.",
            "drop --draft-confidence-stop, or drop --spec-cost-model.",
        ));
    }
}
