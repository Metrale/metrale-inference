// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Critical pairs of the fusion rules (`met circuit rules --critical-pairs`).
//!
//! The fuser only ever groups nodes no group holds yet, so applying rules terminates. By
//! Newman's lemma the plan is then independent of the order rules are tried in exactly when
//! every critical pair is joinable: two matches over a circuit with no node claimed whose node
//! sets overlap, resolved both ways (one claimed first, then the normal greedy), end in the same
//! plan, or at least the same cost. A pair that does not is one where rule `priority` alone
//! decides the plan, the only places a greedy can be beaten.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - Pure: the caller resolves a match (fuses with it seeded, [`crate::fuser::fuse_seeded`]) and
//!   costs the plan; this module only enumerates, memoises and classifies.
//! - Each match is resolved at most once: "A first" does not depend on which B it overlaps.
//! - Pairs are deduplicated by the caller's signature of each match (a layer-independent
//!   name of the site): one representative pair per signature pair is resolved, and the
//!   number of pairs it stands for is kept ([`CriticalPair::occurrences`]).
//! - Output order is deterministic: by the representative's position in the match list.

use crate::prelude::*;

use alloc::collections::{BTreeMap, BTreeSet};

use crate::fuser::Match;

/// 2026-10-23: A resolved plan as the report compares it.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    /// 2026-10-23: The plan digest ([`crate::digest::plan_digest`]).
    pub digest: String,
    /// 2026-10-23: Its cost (the headroom floor, microseconds per step).
    pub cost: f64,
    /// 2026-10-23: Nodes only a placeholder (no kernel of the class) covers.
    pub placeholders: usize,
    /// 2026-10-23: Kernel launches per step ([`crate::fuser::FusionPlan::launches`]): the floor
    /// carries no launch cost, so a same-cost pair can still differ here.
    pub launches: u64,
}

/// 2026-10-23: How a critical pair joins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Joinability {
    /// 2026-10-23: Both orders give the same plan digest.
    Identical,
    /// 2026-10-23: Different plans, same cost and the same placeholders.
    SameCost,
    /// 2026-10-23: Different plans and different costs: rule priority decides.
    PriorityDecided,
    /// 2026-10-23: Only one order gives a plan the class can run in full (the other is refused,
    /// or leaves more nodes to placeholders): the pair has one legal resolution.
    Forced,
}

impl Joinability {
    /// 2026-10-23: The report's spelling.
    pub fn name(self) -> &'static str {
        match self {
            Joinability::Identical => "joinable-identical",
            Joinability::SameCost => "joinable-same-cost",
            Joinability::PriorityDecided => "priority-decided",
            Joinability::Forced => "forced",
        }
    }
}

/// 2026-10-23: Which of the pair the plan the rules produce today contains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Pick {
    /// 2026-10-23: The first match.
    A,
    /// 2026-10-23: The second match.
    B,
    /// 2026-10-23: Neither (a third group holds part of the overlap).
    Neither,
}

/// 2026-10-23: One critical pair, resolved both ways.
#[derive(Debug, Clone, PartialEq)]
pub struct CriticalPair {
    /// 2026-10-23: The match the greedy tries first (higher priority, or same rule earlier).
    pub a: Match,
    /// 2026-10-23: The other.
    pub b: Match,
    /// 2026-10-23: `a` claimed first, then the greedy; `Err` when that plan cannot be formed.
    pub a_first: Result<Outcome, String>,
    /// 2026-10-23: `b` claimed first, then the greedy.
    pub b_first: Result<Outcome, String>,
    /// 2026-10-23: What today's plan picks.
    pub current: Pick,
    /// 2026-10-23: The class.
    pub class: Joinability,
    /// 2026-10-23: Overlapping pairs with this pair's signatures (layers), this one included.
    pub occurrences: u64,
    /// 2026-10-23: Today's plan's cost, which a pick of neither side is judged against.
    pub current_cost: f64,
}

impl CriticalPair {
    /// 2026-10-23: Today's plan takes the dearer side of a priority-decided pair, or the
    /// refused / gap-leaving side of a forced one.
    pub fn picks_dearer(&self) -> bool {
        let (mine, other) = match self.current {
            Pick::A => (&self.a_first, &self.b_first),
            Pick::B => (&self.b_first, &self.a_first),
            Pick::Neither => {
                let best = [&self.a_first, &self.b_first]
                    .into_iter()
                    .filter_map(|o| o.as_ref().ok())
                    .filter(|o| o.placeholders == 0)
                    .map(|o| o.cost)
                    .fold(f64::INFINITY, f64::min);
                return self.class == Joinability::PriorityDecided
                    && best.is_finite()
                    && self.current_cost > best
                    && !same_cost(self.current_cost, best);
            }
        };
        match self.class {
            Joinability::PriorityDecided => match (mine, other) {
                (Ok(m), Ok(o)) => m.cost > o.cost && !same_cost(m.cost, o.cost),
                _ => false,
            },
            Joinability::Forced => !legal_over(mine, other),
            Joinability::Identical | Joinability::SameCost => false,
        }
    }
}

impl CriticalPair {
    /// 2026-10-23: For a same-cost pair, today's side launches more kernels than the other: a
    /// candidate once the cost carries launches.
    pub fn picks_more_launches(&self) -> bool {
        let (mine, other) = match self.current {
            Pick::A => (&self.a_first, &self.b_first),
            Pick::B => (&self.b_first, &self.a_first),
            Pick::Neither => return false,
        };
        self.class == Joinability::SameCost
            && matches!((mine, other), (Ok(m), Ok(o)) if m.launches > o.launches)
    }
}

/// 2026-10-23: Relative tolerance under which two costs are the same (float summation order).
pub const COST_TOLERANCE: f64 = 1e-9;

fn same_cost(a: f64, b: f64) -> bool {
    (a - b).abs() <= COST_TOLERANCE * a.abs().max(b.abs()).max(1.0)
}

/// 2026-10-23: `mine` is at least as legal as `other`: formed, and with no more placeholders.
fn legal_over(mine: &Result<Outcome, String>, other: &Result<Outcome, String>) -> bool {
    match (mine, other) {
        (Ok(m), Ok(o)) => m.placeholders <= o.placeholders,
        (Ok(_), Err(_)) => true,
        (Err(_), _) => false,
    }
}

/// 2026-10-23: Classify a pair from its two resolutions.
pub fn classify(a: &Result<Outcome, String>, b: &Result<Outcome, String>) -> Joinability {
    match (a, b) {
        (Ok(x), Ok(y)) if x.digest == y.digest => Joinability::Identical,
        (Ok(x), Ok(y)) if x.placeholders == y.placeholders => {
            if same_cost(x.cost, y.cost) {
                Joinability::SameCost
            } else {
                Joinability::PriorityDecided
            }
        }
        _ => Joinability::Forced,
    }
}

/// 2026-10-23: Index pairs `(i, j)`, `i < j`, of matches sharing a node.
pub fn overlapping(matches: &[Match]) -> Vec<(usize, usize)> {
    let mut by_node: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, m) in matches.iter().enumerate() {
        for &n in &m.nodes {
            by_node.entry(n).or_default().push(i);
        }
    }
    let mut out: Vec<(usize, usize)> = by_node
        .values()
        .flat_map(|ms| {
            ms.iter()
                .enumerate()
                .flat_map(move |(k, &i)| ms[k + 1..].iter().map(move |&j| (i.min(j), i.max(j))))
        })
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// 2026-10-23: The critical pairs of `matches` (in the greedy's order, [`crate::fuser::
/// match_instances`]). `current` is today's plan's groups as matches and its cost; `signature`
/// names a match independently of its layer; `resolve` fuses with one match seeded and costs the
/// result.
pub fn critical_pairs<K: Ord + Clone, E>(
    matches: &[Match],
    (current, current_cost): (&[Match], f64),
    signature: impl Fn(&Match) -> K,
    mut resolve: impl FnMut(&Match) -> Result<Result<Outcome, String>, E>,
) -> Result<Vec<CriticalPair>, E> {
    let mut reps: BTreeMap<(K, K), ((usize, usize), u64)> = BTreeMap::new();
    for (i, j) in overlapping(matches) {
        let key = (signature(&matches[i]), signature(&matches[j]));
        reps.entry(key).or_insert(((i, j), 0)).1 += 1;
    }
    let mut ordered: Vec<((usize, usize), u64)> = reps.into_values().collect();
    ordered.sort_unstable();
    let current: BTreeSet<&Match> = current.iter().collect();
    let mut memo: BTreeMap<usize, Result<Outcome, String>> = BTreeMap::new();
    let mut out = Vec::with_capacity(ordered.len());
    for ((i, j), occurrences) in ordered {
        for k in [i, j] {
            if let alloc::collections::btree_map::Entry::Vacant(e) = memo.entry(k) {
                e.insert(resolve(&matches[k])?);
            }
        }
        let (a_first, b_first) = (memo[&i].clone(), memo[&j].clone());
        let current = if current.contains(&matches[i]) {
            Pick::A
        } else if current.contains(&matches[j]) {
            Pick::B
        } else {
            Pick::Neither
        };
        out.push(CriticalPair {
            a: matches[i].clone(),
            b: matches[j].clone(),
            class: classify(&a_first, &b_first),
            a_first,
            b_first,
            current,
            occurrences,
            current_cost,
        });
    }
    Ok(out)
}

#[cfg(test)]
#[path = "critical_tests.rs"]
mod critical_tests;
