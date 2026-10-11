// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04 (moved 2026-10-10 from `spec_cost::plan`): `--spec-cost-model measured` as a
//! configuration of the controller: the cost source is the measured table
//! (`CostSource::Measured`), the acceptance comes from the drafter's calibration, and the
//! objective is [`Objective::Energy`] with the floor at depth 1 and `--spec-cost-slack`. It
//! chooses how many drafts are proposed and verified, never which token is emitted.
//!
//! - [`propose_depth`]: before drafting, the uniform depth for width `n` (see [`choose`]).
//!   Above the serve's own `mtp_max_seqs` (the width beyond which the scheduler plain-decodes
//!   every sequence regardless, `sched.levers.mtp_max_seqs`) it returns `0` by construction:
//!   the scheduler will not dispatch a draft there (2026-10-04, found measuring the published
//!   throughput recipe at C=64/128: with the default cap of 32,
//!   `metrale_sched_phase_seconds_count{phase="step_mtp"}` fired once across a 91 s,
//!   32129-token run).
//! - [`sequence_depths`]: after drafting, how many of each sequence's drafts to verify
//!   ([`cut_depths`]): a row's marginal cost is verify cost only (the drafts are paid for).
//!   The pilot measured verify cost as non-monotone in rows per sequence (n=1 cost MORE at 3
//!   rows than at 4 on the dense 27B, 2026-10-04), so every reachable row count is scanned.
//!
//! Owner: speculative.
//! Invariants:
//! - Pure and deterministic: table, calibration, width and confidences in; no clock.
//! - Ties keep the shallower depth or the earlier sequence (slot order).

use super::chain::chain_sum;
use super::cost::{table_batch, table_cost};
use super::decide::{Candidate, FloorRef, Margins, Objective, choose, cut_depths};
use crate::spec_cost::{AcceptanceCalibration, CostTable};

fn objective(slack: f64) -> Objective {
    Objective::Energy {
        slack,
        floor: FloorRef::Depth(1),
    }
}

/// 2026-10-04: The propose depth for a batch of `n` sequences under `slack` (in `0..1`).
/// `mtp_max_seqs` is the serve's dispatch cap (SSOT: `sched.levers.mtp_max_seqs`); above it the
/// answer is `0` without consulting the table. Within the cap the search can also return `0`
/// on its own terms (a draft row that costs a whole step); a caller that does not suspend MTP
/// from the energy preference alone re-checks the cap to tell the two apart (`mtp_step.rs`).
pub fn propose_depth(
    table: &CostTable,
    cal: &AcceptanceCalibration,
    n: usize,
    slack: f64,
    mtp_max_seqs: usize,
) -> usize {
    if n > mtp_max_seqs {
        return 0;
    }
    let cands: Vec<Candidate> = (0..=table.max_k())
        .map(|k| {
            let e = chain_sum(|j| cal.prior(j), k, 1.0);
            Candidate {
                k,
                tokens: n as f64 * e,
                slowest: e,
                cost: table_cost(table, n, k).expect("k <= max_k"),
            }
        })
        .collect();
    choose(&objective(slack), &Margins::NONE, None, &cands).expect("max_k >= 0: one candidate")
}

/// 2026-10-04: Drafts to verify per sequence. `confidences[i]` is sequence `i`'s draft top-1
/// log-probabilities in draft order (empty with `known_drafts[i] > 0`: drafts without
/// confidences, which take the calibration's position priors).
pub fn sequence_depths(
    table: &CostTable,
    cal: &AcceptanceCalibration,
    confidences: &[&[f32]],
    known_drafts: &[usize],
    k_max: usize,
    row_budget: usize,
    slack: f64,
) -> Vec<usize> {
    let n = confidences.len();
    assert_eq!(n, known_drafts.len(), "one draft count per sequence");
    let k_max = k_max.min(table.max_k());
    let rate = |i: usize, j: usize| -> f64 {
        confidences[i]
            .get(j - 1)
            .map_or_else(|| cal.prior(j), |&lp| cal.p_given_lp(lp))
    };
    cut_depths(
        &objective(slack),
        rate,
        known_drafts,
        k_max,
        row_budget,
        |rows| table_batch(table, n, rows, k_max),
    )
}

#[cfg(test)]
#[path = "measured_tests.rs"]
mod tests;
