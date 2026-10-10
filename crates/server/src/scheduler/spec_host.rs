// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The scheduler's host for the per-step speculation decision
//! (`metrale_speculative::spec_ctl::batch::BatchSpec`): which depths a step may run, how the
//! controller is built from the serve's settings, and the snapshot it reports. The decision
//! itself, and its cost and acceptance models, live in `spec_ctl`; this module only binds them
//! to the scheduler (SBIO: the clock is read by the caller, `core/lane_decode.rs`).
//!
//! Owner: scheduler.
//! Invariants:
//! - The allowed depths always include plain decode (0) and the depth the speculative step
//!   would run; with a pinned DFlash depth they are exactly those two.
//! - Built only when MTP or DFlash runs without `--mtp-gate force`, with an objective
//!   (`validate_serve_args` refuses a missing one; this module fails fast if it slips by).

use metrale_speculative::spec_ctl::accept::{AcceptParams, ColdPrior, SeedWeight};
use metrale_speculative::spec_ctl::batch::BatchSpec;
use metrale_speculative::spec_ctl::calib::Calibration;
use metrale_speculative::spec_ctl::controller::{ControllerConfig, SpecController};
use metrale_speculative::spec_ctl::cost::{CostModel, CostSource};
use metrale_speculative::spec_ctl::decide::Margins;
use metrale_speculative::spec_ctl::online::OnlineTable;
use metrale_speculative::spec_ctl::reprobe::ReprobePolicy;
use metrale_speculative::spec_ctl::source::{DraftCost, DraftKind, DraftSource};

use crate::scheduler::config::SpecPolicy;

/// 2026-10-10: Weight of a measured step in an online cost cell: the MTP gate's per-window
/// EWMA weight, which this decision replaced.
const ONLINE_ALPHA: f64 = 0.3;
/// 2026-10-10: Observations after which an online cost cell is measured again: the gate's
/// plain-decode re-probe interval (256 tokens at one stream).
const ONLINE_STALE_STEPS: u64 = 256;
/// 2026-10-10: Plain decode must beat speculation by this fraction: the gate's switch margin.
const SUSPEND_MARGIN: f64 = 0.05;
/// 2026-10-10: Plain-decoded tokens after which a suspended batch re-probes speculation (the
/// gate's re-probe interval), for a window of 16 speculative steps (its probe window).
const RESUME_AFTER_TOKENS: u32 = 256;
const PROBE_STEPS: u32 = 16;
/// 2026-10-10: Acceptance memory (about 20 observations), the serve-wide prior's weight, and
/// the chained cold prior's weight: the replay-tested configuration (`spec_ctl` replay tests).
const ACCEPT: AcceptParams = AcceptParams {
    decay: 0.95,
    prior_weight: 4.0,
    cold_weight: 8.0,
    seed: SeedWeight::Unit,
};
/// 2026-10-10: Exploration one deeper every 32 decisions, backing off to 512 while the
/// choice holds; a re-probe softens a stream's counts to a quarter.
const REPROBE: ReprobePolicy = ReprobePolicy {
    explore_every: Some(32),
    explore_max: 512,
    resume_after_tokens: Some(RESUME_AFTER_TOKENS),
    probe_steps: PROBE_STEPS,
    soften: 0.25,
};
/// 2026-10-10: Weight of a measured step in the measured table's calibration.
const MEASURED_CALIBRATION_ALPHA: f64 = 0.1;

/// 2026-10-10: The batch decision for this serve: costs from the measured table when
/// `--spec-cost-model measured` loaded one, else measured online; `max_k` is the drafter's
/// deepest proposal.
pub(crate) fn build(
    policy: &SpecPolicy,
    measured: Option<&metrale_speculative::spec_cost::CostTable>,
    max_k: usize,
    dflash: bool,
) -> BatchSpec {
    let objective = policy
        .objective
        .expect("validate_serve_args requires --spec-objective whenever the controller runs");
    let (source, calib) = match measured {
        Some(t) => (
            CostSource::Measured(t.clone()),
            Calibration::new(MEASURED_CALIBRATION_ALPHA),
        ),
        None => (
            CostSource::Online(OnlineTable::new(ONLINE_ALPHA, ONLINE_STALE_STEPS)),
            Calibration::new(0.0),
        ),
    };
    let cfg = ControllerConfig {
        source: DraftSource {
            kind: if dflash {
                DraftKind::DFlash
            } else {
                DraftKind::Mtp
            },
            max_k,
            prefix_stable: true,
            // 2026-10-10: Unused by the measured and online sources, which price whole steps.
            draft_cost: DraftCost::Free,
        },
        accept: ACCEPT,
        objective,
        margins: Margins {
            deeper: 0.0,
            shallower: 0.0,
            suspend: SUSPEND_MARGIN,
        },
        reprobe: REPROBE,
    };
    tracing::info!(
        "speculation controller: {objective:?}, costs {}, depths {}",
        if measured.is_some() {
            "from the measured table"
        } else {
            "measured online"
        },
        if dflash && !policy.dflash_depth_pinned {
            format!("0..={max_k}")
        } else {
            "plain or the step's depth".to_string()
        }
    );
    BatchSpec::new(SpecController::new(
        cfg,
        CostModel { source, calib },
        ColdPrior::Chained,
    ))
}

/// 2026-10-10: The depths a step may run: plain decode and `spec_k` (the depth the
/// speculative step runs: the MTP ladder/rung/planner depth, or the pinned DFlash depth); an
/// unpinned DFlash drafter may run any depth up to `spec_k`.
pub(crate) fn allowed(dflash: bool, dflash_depth_pinned: bool, spec_k: usize) -> Vec<usize> {
    if dflash && !dflash_depth_pinned {
        (0..=spec_k).collect()
    } else {
        vec![0, spec_k]
    }
}

/// 2026-10-10: The snapshot's view of the decision: the mode of the last step and the
/// delivered tokens per second.
pub(crate) fn snapshot(h: &BatchSpec) -> (metrale_speculative::snapshot::MtpModeSnap, f32) {
    use metrale_speculative::snapshot::MtpModeSnap;
    let mode = if h.probing() {
        MtpModeSnap::Probing
    } else if h.last_k() == Some(0) {
        MtpModeSnap::Serial
    } else {
        MtpModeSnap::Mtp
    };
    (mode, h.delivered_tps() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowed_depths_always_hold_plain_decode_and_the_step_depth() {
        assert_eq!(allowed(false, false, 3), vec![0, 3]);
        assert_eq!(allowed(true, true, 7), vec![0, 7]);
        assert_eq!(allowed(true, false, 3), vec![0, 1, 2, 3]);
    }
}
