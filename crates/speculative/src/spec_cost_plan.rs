// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: The measured planner of `--spec-cost-model measured`: draft depth by expected
//! accepted tokens per joule, with tok/s as a constraint. Pure and deterministic: its output
//! depends only on the table, the calibration, the batch width and the drafter's confidences;
//! no clock is read. It chooses how many drafts are proposed and verified, never which token is
//! emitted, so greedy output is unchanged on row-invariant tiers.
//!
//! - [`propose_depth`]: before drafting, the depth `K` for width `n` that maximises
//!   E[tokens] / E[joules] over `0..=max_k`, among the depths whose E[tokens] / E[ms] is at
//!   least `(1 - slack)` times depth 1's. E[tokens] per sequence is `1 + Σ_{j<=K} Π_{t<=j}
//!   prior(t)`; a step costs `verify(n, K) + draft(n, K)` from the table.
//! - [`sequence_depths`]: after drafting, how many of each sequence's drafts to verify. Every
//!   sequence starts at one draft (none when `K = 0`); a row is added greedily to the sequence
//!   whose next draft has the most expected accepted tokens (its chained P(accept) from its
//!   confidences, the position prior where one is missing), while the batch's tokens per
//!   joule rises and the time constraint holds. The drafts are already paid for, so a row's
//!   marginal cost is verify cost only; the table is uniform in depth, so a batch of `R` rows
//!   costs the width's cells interpolated in `R / n`.
//!
//! Owner: speculative.
//! Invariants:
//! - [`sequence_depths`] never gives a sequence more drafts than it has, more than `k_max`, or
//!   the batch more than `row_budget` draft rows; a batch with more sequences than
//!   `row_budget` verifies no drafts at all (a plain decode step).
//! - Ties keep the shallower choice ([`propose_depth`]) or the earlier sequence (the caller's
//!   order, which is slot order).

use super::{AcceptanceCalibration, Cell, CostTable};

/// 2026-10-04: Expected tokens per sequence of a chain of `k` drafts at the position priors.
fn prior_tokens(cal: &AcceptanceCalibration, k: usize) -> f64 {
    let mut run = 1.0;
    let mut total = 1.0;
    for j in 1..=k {
        run *= cal.prior(j);
        total += run;
    }
    total
}

/// 2026-10-04: The propose depth for a batch of `n` sequences: see the module doc. `slack` is
/// `--spec-cost-slack`, in `0..1`. A table without depth 1 has no reference rate, so its
/// constraint is against depth 0.
pub fn propose_depth(
    table: &CostTable,
    cal: &AcceptanceCalibration,
    n: usize,
    slack: f64,
) -> usize {
    let at = |k: usize| -> (f64, Cell) {
        let c = table.cell(n, k).expect("k <= max_k");
        (n as f64 * prior_tokens(cal, k), c)
    };
    let reference = {
        let (tok, c) = at(table.max_k().min(1));
        tok / c.step_ms()
    };
    let mut best: Option<(usize, f64)> = None;
    for k in 0..=table.max_k() {
        let (tok, c) = at(k);
        if tok / c.step_ms() < (1.0 - slack) * reference {
            continue;
        }
        let per_joule = tok / c.step_j();
        if best.is_none_or(|(_, b)| per_joule > b) {
            best = Some((k, per_joule));
        }
    }
    best.map_or(table.max_k().min(1), |(k, _)| k)
}

/// 2026-10-04: The verify cost of `rows` draft rows across `n` sequences, interpolated between
/// the width's uniform-depth cells; the drafting cost of depth `k_max` is sunk and added whole.
fn batch_cost(table: &CostTable, n: usize, rows: usize, k_max: usize) -> (f64, f64) {
    let depth = rows as f64 / n as f64;
    let (lo, hi) = (depth.floor() as usize, depth.ceil() as usize);
    let (a, b) = (
        table.cell(n, lo.min(table.max_k())).expect("in range"),
        table.cell(n, hi.min(table.max_k())).expect("in range"),
    );
    let t = depth - lo as f64;
    let draft = table.cell(n, k_max).expect("k_max <= max_k");
    (
        a.verify_ms + (b.verify_ms - a.verify_ms) * t + draft.draft_ms,
        a.verify_j + (b.verify_j - a.verify_j) * t + draft.draft_j,
    )
}

/// 2026-10-04: Drafts to verify per sequence: see the module doc. `drafts[i]` is sequence
/// `i`'s draft confidences (top-1 log-probabilities, in draft order; its length is how many
/// drafts it has; an empty slice with `known_drafts[i] > 0` means drafts without confidences).
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
    let cap: Vec<usize> = known_drafts.iter().map(|&d| d.min(k_max)).collect();
    // 2026-10-04: P(draft j of sequence i accepted | drafts before it accepted).
    let p = |i: usize, j: usize| -> f64 {
        confidences[i]
            .get(j - 1)
            .map_or_else(|| cal.prior(j), |&lp| cal.p_given_lp(lp))
    };
    let mut depth: Vec<usize> = cap.iter().map(|&c| c.min(1)).collect();
    let mut chain: Vec<f64> = (0..n)
        .map(|i| if depth[i] == 1 { p(i, 1) } else { 1.0 })
        .collect();
    let mut tokens: f64 = n as f64
        + (0..n)
            .filter(|&i| depth[i] == 1)
            .map(|i| chain[i])
            .sum::<f64>();
    let mut rows: usize = depth.iter().sum();
    if n == 0 || rows > row_budget {
        return vec![0; n];
    }
    let reference = {
        let (ms, _) = batch_cost(table, n, rows, k_max);
        tokens / ms
    };
    loop {
        if rows >= row_budget {
            break;
        }
        // 2026-10-04: The sequence whose next draft adds the most expected tokens; the first
        // on a tie.
        let next = (0..n)
            .filter(|&i| depth[i] < cap[i])
            .map(|i| (i, chain[i] * p(i, depth[i] + 1)))
            .fold(None::<(usize, f64)>, |best, (i, gain)| match best {
                Some((_, g)) if g >= gain => best,
                _ => Some((i, gain)),
            });
        let Some((i, gain)) = next else { break };
        let (_, j0) = batch_cost(table, n, rows, k_max);
        let (ms1, j1) = batch_cost(table, n, rows + 1, k_max);
        let t1 = tokens + gain;
        if t1 / j1 <= tokens / j0 || t1 / ms1 < (1.0 - slack) * reference {
            break;
        }
        depth[i] += 1;
        chain[i] *= p(i, depth[i]);
        tokens = t1;
        rows += 1;
    }
    depth
}

#[cfg(test)]
#[path = "spec_cost_plan_tests.rs"]
mod tests;
