// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: The measured planner of `--spec-cost-model measured`: draft depth by expected
//! accepted tokens per joule, with tok/s as a constraint. Pure and deterministic: its output
//! depends only on the table, the calibration, the batch width and the drafter's confidences;
//! no clock is read. It chooses how many drafts are proposed and verified, never which token is
//! emitted, so greedy output is unchanged on row-invariant tiers.
//!
//! - [`propose_depth`]: before drafting, the depth `K` for width `n` that maximises
//!   `E[tokens] / E[joules]` over `0..=max_k`, among the depths whose `E[tokens] / E[ms]` is at
//!   least `(1 - slack)` times depth 1's. `E[tokens]` per sequence is `1 + Σ_{j<=K} Π_{t<=j}
//!   prior(t)`; a step costs `verify(n, K) + draft(n, K)` from the table. Above the serve's own
//!   `mtp_max_seqs` (the width beyond which the scheduler plain-decodes every sequence
//!   regardless, `sched.levers.mtp_max_seqs`), it returns `0` by construction: there is nothing
//!   to search, since the scheduler will not dispatch a draft there no matter what this picks
//!   (2026-10-04, found measuring the published throughput recipe at C=64/128: with the
//!   default cap of 32, `metrale_sched_phase_seconds_count{phase="step_mtp"}` fired once across
//!   a 91 s, 32129-token run — the batch plain-decoded almost throughout).
//! - [`sequence_depths`]: after drafting, how many of each sequence's drafts to verify. Every
//!   sequence starts at one draft (none when `K = 0`). Which sequence a further row goes to is
//!   still decided greedily (by chained P(accept): a sequence's own marginal token gain is
//!   non-increasing in its depth, since it is a running product of probabilities `<= 1`, so for
//!   any total row count the globally-largest-gain-first order is the tokens-maximising
//!   allocation — an exchange argument, not an approximation). But the total number of rows `R`
//!   the batch spends is chosen by scanning every reachable `R`, not by stopping at the first one
//!   whose tokens per joule does not improve: the pilot measured verify cost as non-monotone in
//!   rows per sequence (n=1 cost MORE at 3 rows than at 4, on the dense 27B, 2026-10-04), so a
//!   row that does not pay can still be worth adding when a cheaper row lies beyond it. The
//!   drafts are already paid for, so a row's marginal cost is verify cost only; the table is
//!   uniform in depth, so a batch of `R` rows costs the width's cells interpolated in `R / n`.
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
/// constraint is against depth 0. `mtp_max_seqs` is the serve's own dispatch cap
/// (`sched.levers.mtp_max_seqs`, SSOT — never a restated default here); `n > mtp_max_seqs`
/// returns `0` without consulting the table (see the module doc). Within the cap the search can
/// also return `0` on its own terms (a draft row that costs a whole step — see the
/// `no_drafting_when_it_is_neither_faster_nor_cheaper` test) — the two `0`s mean different
/// things to a caller that is not wired to suspend MTP from the table's energy preference
/// alone: it must re-check the cap itself to tell them apart (`mtp_step.rs` does).
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
    let chain: Vec<f64> = (0..n)
        .map(|i| if depth[i] == 1 { p(i, 1) } else { 1.0 })
        .collect();
    let base_tokens: f64 = n as f64
        + (0..n)
            .filter(|&i| depth[i] == 1)
            .map(|i| chain[i])
            .sum::<f64>();
    let base_rows: usize = depth.iter().sum();
    if n == 0 || base_rows > row_budget {
        return vec![0; n];
    }
    // 2026-10-04: Every row beyond a sequence's mandatory first draft, with its marginal
    // expected-token gain, in draft order per sequence. A sequence's own gains are
    // non-increasing (see the module doc), so sorting all of them together, across every
    // sequence, and taking a prefix of length `R - base_rows` is the tokens-maximising way to
    // reach any total row count `R >= base_rows`.
    struct Extra {
        seq: usize,
        depth: usize,
        gain: f64,
    }
    let mut extra = Vec::new();
    for i in 0..n {
        let mut run = chain[i];
        for j in (depth[i] + 1)..=cap[i] {
            run *= p(i, j);
            extra.push(Extra {
                seq: i,
                depth: j,
                gain: run,
            });
        }
    }
    extra.sort_by(|a, b| {
        b.gain
            .partial_cmp(&a.gain)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let reference = {
        let (ms, _) = batch_cost(table, n, base_rows, k_max);
        base_tokens / ms
    };
    // 2026-10-04: The best total row count is found by scanning every reachable `R`, not by
    // stopping at the first `R` whose tokens per joule does not improve: the table's cost is
    // not guaranteed monotone in rows (see the module doc), so a row that does not pay can
    // still be worth taking when a cheaper one lies beyond it. Ties keep the shallower `R`.
    let max_take = (row_budget - base_rows).min(extra.len());
    let mut best_rows = base_rows;
    let mut best_per_joule = {
        let (_, j) = batch_cost(table, n, base_rows, k_max);
        base_tokens / j
    };
    let mut tokens = base_tokens;
    for take in 1..=max_take {
        tokens += extra[take - 1].gain;
        let rows = base_rows + take;
        let (ms, j) = batch_cost(table, n, rows, k_max);
        if tokens / ms < (1.0 - slack) * reference {
            continue;
        }
        let per_joule = tokens / j;
        if per_joule > best_per_joule {
            best_rows = rows;
            best_per_joule = per_joule;
        }
    }
    for e in &extra[..(best_rows - base_rows)] {
        depth[e.seq] = e.depth;
    }
    depth
}

#[cfg(test)]
#[path = "spec_cost_plan_tests.rs"]
mod tests;
