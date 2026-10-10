// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The decision: which candidate depth (or row count) a step runs, under one
//! explicit objective and switch margins. Every K decision in the engine goes through
//! [`choose`]; the per-sequence row allocation goes through [`cut_depths`], which builds its
//! candidates and calls [`choose`] too.
//!
//! Owner: speculative.
//! Invariants:
//! - Pure and deterministic: the answer depends only on the arguments.
//! - Ties keep the shallower candidate, except that a challenger which reaches the incumbent's
//!   value times `1 + margin` replaces it ("reaching the threshold switches").
//! - Under [`Objective::Energy`] a candidate below `(1 - slack)` times the floor candidate's
//!   tokens/ms, or without a joule cost, is infeasible; with none feasible the floor candidate
//!   is returned.
//!
//! Objectives (no default; the recipe or the cost mode names one):
//! - `Latency`: the slowest stream's expected tokens per ms (at one stream, Throughput).
//! - `Throughput`: all streams' expected tokens per ms.
//! - `Energy`: expected tokens per joule, subject to a tokens/ms floor relative to the
//!   candidate named by [`FloorRef`].

use super::cost::StepCost;

/// 2026-10-10: Which candidate's tokens/ms the energy floor is relative to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloorRef {
    /// 2026-10-10: The candidate at this depth (clamped to the deepest candidate). Depth 1 is
    /// `--spec-cost-model measured`'s reference.
    Depth(usize),
    /// 2026-10-10: The fastest candidate.
    Best,
}

/// 2026-10-10: What a step maximises.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Objective {
    Latency,
    Throughput,
    Energy { slack: f64, floor: FloorRef },
}

/// 2026-10-10: Margins a challenger must beat the incumbent by, by direction; `suspend` is the
/// margin plain decode (depth 0) needs over the best speculative candidate whatever the
/// incumbent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Margins {
    pub deeper: f64,
    pub shallower: f64,
    pub suspend: f64,
}

impl Margins {
    pub const NONE: Self = Self {
        deeper: 0.0,
        shallower: 0.0,
        suspend: 0.0,
    };
}

/// 2026-10-10: One option: depth `k` (or, in [`cut_depths`], a total row count), the expected
/// tokens of all streams, of the slowest stream, and the step's cost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    pub k: usize,
    pub tokens: f64,
    pub slowest: f64,
    pub cost: StepCost,
}

impl Candidate {
    fn rate(&self) -> f64 {
        self.tokens / self.cost.ms
    }
}

fn floor_index(cands: &[Candidate], floor: FloorRef) -> usize {
    match floor {
        FloorRef::Depth(d) => {
            let top = cands.iter().map(|c| c.k).max().unwrap_or(0);
            cands.iter().position(|c| c.k == d.min(top)).unwrap_or(0)
        }
        FloorRef::Best => {
            cands
                .iter()
                .enumerate()
                .fold((0, f64::MIN), |b, (i, c)| {
                    if c.rate() > b.1 { (i, c.rate()) } else { b }
                })
                .0
        }
    }
}

/// 2026-10-10: The candidate's value under `obj`, `None` when infeasible.
fn value(obj: &Objective, c: &Candidate, floor_rate: f64) -> Option<f64> {
    match *obj {
        Objective::Latency => Some(c.slowest / c.cost.ms),
        Objective::Throughput => Some(c.rate()),
        Objective::Energy { slack, .. } => {
            if c.rate() < (1.0 - slack) * floor_rate {
                return None;
            }
            c.cost.j.map(|j| c.tokens / j)
        }
    }
}

/// 2026-10-10: The chosen candidate's `k` among `cands` (ascending in `k`); `None` when empty.
/// `incumbent` is the depth the caller runs now, when its margins are hysteresis.
pub fn choose(
    obj: &Objective,
    margins: &Margins,
    incumbent: Option<usize>,
    cands: &[Candidate],
) -> Option<usize> {
    let floor = match obj {
        Objective::Energy { floor, .. } => floor_index(cands, *floor),
        _ => 0,
    };
    let floor_rate = cands.get(floor)?.rate();
    let mut best: Option<(usize, f64)> = None;
    for c in cands {
        let Some(v) = value(obj, c, floor_rate) else {
            continue;
        };
        let is_inc = incumbent == Some(c.k);
        let m = match incumbent {
            _ if is_inc => 0.0,
            _ if c.k == 0 => margins.suspend,
            Some(i) if c.k > i => margins.deeper,
            Some(_) => margins.shallower,
            None => 0.0,
        };
        let a = v / (1.0 + m);
        let replace = match best {
            None => true,
            Some((bk, ba)) => a > ba || (a == ba && incumbent == Some(bk) && !is_inc),
        };
        if replace {
            best = Some((c.k, a));
        }
    }
    Some(best.map_or(cands[floor].k, |b| b.0))
}

/// 2026-10-10: Drafts to verify per sequence when drafts are already proposed. Every sequence
/// keeps one draft (none when it holds none or `k_max` is 0); further rows go out in order of
/// marginal expected-token gain, which is the tokens-maximising order for every total row
/// count (each sequence's gains are non-increasing). The total row count is then chosen by
/// [`choose`] over every reachable count, not by stopping at the first that does not pay
/// (measured costs are not monotone in rows). `rate(i, j)` is P(draft `j` of sequence `i`
/// accepted | earlier accepted); `batch(rows)` is the batch's cost at that many draft rows.
///
/// Never more drafts than a sequence holds, than `k_max`, or than `row_budget` rows in total;
/// a batch whose mandatory first drafts exceed the budget verifies none (a plain step).
#[allow(clippy::too_many_arguments)]
pub fn cut_depths(
    obj: &Objective,
    rate: impl Fn(usize, usize) -> f64,
    known_drafts: &[usize],
    k_max: usize,
    row_budget: usize,
    batch: impl Fn(usize) -> Option<StepCost>,
) -> Vec<usize> {
    let n = known_drafts.len();
    let cap: Vec<usize> = known_drafts.iter().map(|&d| d.min(k_max)).collect();
    let mut depth: Vec<usize> = cap.iter().map(|&c| c.min(1)).collect();
    let chain: Vec<f64> = (0..n)
        .map(|i| if depth[i] == 1 { rate(i, 1) } else { 1.0 })
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
    struct Extra {
        seq: usize,
        depth: usize,
        gain: f64,
    }
    let mut extra = Vec::new();
    for i in 0..n {
        let mut run = chain[i];
        for j in (depth[i] + 1)..=cap[i] {
            run *= rate(i, j);
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
    let max_take = (row_budget - base_rows).min(extra.len());
    let mut per_seq: Vec<f64> = (0..n)
        .map(|i| 1.0 + if depth[i] == 1 { chain[i] } else { 0.0 })
        .collect();
    let mut cands = Vec::with_capacity(max_take + 1);
    let mut tokens = base_tokens;
    for take in 0..=max_take {
        if take > 0 {
            let e = &extra[take - 1];
            tokens += e.gain;
            per_seq[e.seq] += e.gain;
        }
        let rows = base_rows + take;
        let Some(cost) = batch(rows) else { break };
        let slowest = per_seq.iter().copied().fold(f64::INFINITY, f64::min);
        cands.push(Candidate {
            k: rows,
            tokens,
            slowest,
            cost,
        });
    }
    // 2026-10-10: A depth floor names a row count here: every sequence at that depth.
    let obj = match *obj {
        Objective::Energy {
            slack,
            floor: FloorRef::Depth(d),
        } => Objective::Energy {
            slack,
            floor: FloorRef::Depth(cap.iter().map(|&c| c.min(d)).sum()),
        },
        o => o,
    };
    let rows = choose(&obj, &Margins::NONE, None, &cands).unwrap_or(base_rows);
    for e in &extra[..(rows - base_rows)] {
        depth[e.seq] = e.depth;
    }
    depth
}

#[cfg(test)]
#[path = "decide_tests.rs"]
mod tests;
