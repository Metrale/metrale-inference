// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The critical-pair report (`kernels/circuits/plans/critical/CRITICAL-PAIRS.md`)
//! and its shrink-only allowlist (`kernels/circuits/CRITICAL_PAIRS.toml`).
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - Deterministic: plans sorted by model, device, mode and rows; pairs in each plan in the
//!   greedy's order; aggregates in rule-id order. The same inputs render the same bytes.
//! - The gate keys on the rule pair (ids sorted), the unit a priority edit changes. A rule pair
//!   is gated when, in any plan, it is priority-decided, or forced with today's plan on the side
//!   that is refused or leaves gaps. Each gated pair needs an allowlist entry with a reason; an
//!   entry no plan gates any more is stale and must be removed; the list never grows past the
//!   caller's ceiling.

use crate::prelude::*;

use alloc::collections::{BTreeMap, BTreeSet};
use core::fmt::Write as _;

use crate::critical::{CriticalPair, Joinability, Outcome, Pick};
use crate::venn::Run;

/// 2026-10-23: One pair of one plan, named for the report.
#[derive(Debug, Clone, PartialEq)]
pub struct PairRow {
    /// 2026-10-23: `block [locals]` of the first match.
    pub site_a: String,
    /// 2026-10-23: Of the second.
    pub site_b: String,
    /// 2026-10-23: The pair.
    pub pair: CriticalPair,
}

impl PairRow {
    fn gated(&self) -> bool {
        self.pair.class == Joinability::PriorityDecided
            || (self.pair.class == Joinability::Forced && self.pair.picks_dearer())
    }

    fn key(&self) -> (String, String) {
        let (a, b) = (&self.pair.a.rule, &self.pair.b.rule);
        (a.min(b).clone(), a.max(b).clone())
    }
}

/// 2026-10-23: The critical pairs of one plan.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanPairs {
    /// 2026-10-23: The model's label.
    pub model: String,
    /// 2026-10-23: Device id.
    pub device: String,
    /// 2026-10-23: Mode and rows.
    pub run: Run,
    /// 2026-10-23: Matches over the free circuit.
    pub matches: usize,
    /// 2026-10-23: Today's plan's floor, microseconds per step.
    pub floor_us: f64,
    /// 2026-10-23: The pairs.
    pub rows: Vec<PairRow>,
}

/// 2026-10-23: One allowlist entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Allowed {
    /// 2026-10-23: The rule pair, ids sorted.
    pub rules: (String, String),
    /// 2026-10-23: The one-line reason the priority is the intended resolution.
    pub why: String,
}

/// 2026-10-23: Parse the allowlist: `[[pair]] rules = ["a", "b"] why = "..."`.
pub fn parse_allowlist(text: &str) -> Result<Vec<Allowed>, String> {
    let t: toml::Table = toml::from_str(text).map_err(|e| e.to_string())?;
    let Some(pairs) = t.get("pair") else {
        return Ok(Vec::new());
    };
    let pairs = pairs.as_array().ok_or("`pair` is not an array of tables")?;
    pairs
        .iter()
        .map(|p| {
            let ids: Vec<&str> = p
                .get("rules")
                .and_then(|r| r.as_array())
                .map(|r| r.iter().filter_map(|v| v.as_str()).collect())
                .unwrap_or_default();
            let why = p.get("why").and_then(|w| w.as_str()).unwrap_or_default();
            match ids.as_slice() {
                [a, b] if !why.trim().is_empty() => Ok(Allowed {
                    rules: ((*a.min(b)).into(), (*a.max(b)).into()),
                    why: why.into(),
                }),
                [_, _] => Err(format!("pair {ids:?} has no `why`")),
                _ => Err(format!("a pair names {} rules, not 2", ids.len())),
            }
        })
        .collect()
}

/// 2026-10-23: What the gate refuses: gated pairs without an entry, stale entries, duplicate
/// entries and a list past `ceiling`. Empty: the gate passes.
pub fn gate(plans: &[PlanPairs], allowed: &[Allowed], ceiling: usize) -> Vec<String> {
    let gated: BTreeSet<(String, String)> = plans
        .iter()
        .flat_map(|p| p.rows.iter().filter(|r| r.gated()).map(PairRow::key))
        .collect();
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for a in allowed {
        if !seen.insert(a.rules.clone()) {
            out.push(format!("duplicate entry {:?}", a.rules));
        }
        if !gated.contains(&a.rules) {
            out.push(format!(
                "stale entry {:?}: no plan's priority decides it any more; remove it and lower the ceiling",
                a.rules
            ));
        }
    }
    for k in gated.iter().filter(|k| !seen.contains(*k)) {
        out.push(format!(
            "rule pair {k:?} is decided by priority alone and has no allowlist entry (add one with a one-line `why`, or make the pair joinable)"
        ));
    }
    if allowed.len() > ceiling {
        out.push(format!(
            "the allowlist holds {} entries, past its ceiling {ceiling}",
            allowed.len()
        ));
    }
    out
}

fn cost(o: &Result<Outcome, String>) -> String {
    match o {
        Ok(o) if o.placeholders > 0 => format!("{:.3} (+{} gap)", o.cost, o.placeholders),
        Ok(o) => format!("{:.3}", o.cost),
        Err(_) => "refused".into(),
    }
}

fn pick(p: Pick) -> &'static str {
    match p {
        Pick::A => "A",
        Pick::B => "B",
        Pick::Neither => "neither",
    }
}

const CLASSES: [Joinability; 4] = [
    Joinability::Identical,
    Joinability::SameCost,
    Joinability::PriorityDecided,
    Joinability::Forced,
];

/// 2026-10-23: Render the report. `command` is the line that regenerates it.
pub fn render(plans: &[PlanPairs], allowed: &[Allowed], command: &str) -> String {
    let mut plans: Vec<&PlanPairs> = plans.iter().collect();
    plans.sort_by(|a, b| {
        (&a.model, &a.device, a.run.mode, a.run.rows)
            .cmp(&(&b.model, &b.device, b.run.mode, b.run.rows))
    });
    let mut s = String::new();
    let _ = writeln!(
        s,
        "# Critical pairs of the fusion rules\n\nGenerated by `{command}`; checked by the same command with `--check`. Do not edit.\n\nA critical pair is two matches of the class's rules, over the circuit with no node claimed, that share a node. Each is resolved both ways: one match claimed first, then the normal greedy. Classes: `joinable-identical` (same plan digest), `joinable-same-cost` (same headroom floor), `priority-decided` (different floor: rule priority alone picks the plan), `forced` (only one order gives a full plan on the class's kernels). Pairs are deduplicated across layers by the site signature (rule, and each node's block, local id, layer kind and weight format); `sites` counts representatives, `occurrences` the layer instances they stand for. A floor is the step's headroom floor in microseconds, Σ max(bytes / bandwidth, FLOPs / peak) over its launch sites.\n"
    );
    let mut by_class: BTreeMap<Joinability, (u64, u64)> = BTreeMap::new();
    let (mut cheaper, mut dearer, mut neither) = (0u64, 0u64, 0u64);
    for r in plans.iter().flat_map(|p| &p.rows) {
        let e = by_class.entry(r.pair.class).or_default();
        e.0 += 1;
        e.1 += r.pair.occurrences;
        if r.pair.class == Joinability::PriorityDecided {
            match (r.pair.current, r.pair.picks_dearer()) {
                (Pick::Neither, _) => neither += 1,
                (_, true) => dearer += 1,
                (_, false) => cheaper += 1,
            }
        }
    }
    let _ = writeln!(
        s,
        "## Summary\n\n{} plans.\n\n| class | sites | occurrences |\n|---|---:|---:|",
        plans.len()
    );
    for c in CLASSES {
        let (n, o) = by_class.get(&c).copied().unwrap_or_default();
        let _ = writeln!(s, "| {} | {n} | {o} |", c.name());
    }
    let _ = writeln!(
        s,
        "\nPriority-decided sites where today's plan takes the cheaper side: {cheaper}; the dearer side: {dearer}; neither side: {neither}.\n"
    );
    let rows = || plans.iter().flat_map(|p| &p.rows);
    let launches = |o: &Result<Outcome, String>| o.as_ref().map(|o| o.launches).ok();
    let mut by_launch = [0u64; 3];
    for r in rows().filter(|r| r.pair.class == Joinability::SameCost) {
        let (a, b) = (launches(&r.pair.a_first), launches(&r.pair.b_first));
        let (mine, other) = match r.pair.current {
            Pick::A => (a, b),
            Pick::B => (b, a),
            Pick::Neither => (None, None),
        };
        match (a != b, mine.zip(other)) {
            (false, _) => {}
            (true, Some((m, o))) if m <= o => by_launch[0] += 1,
            (true, Some(_)) => by_launch[1] += 1,
            (true, None) => by_launch[2] += 1,
        }
    }
    let _ = writeln!(
        s,
        "Same-cost sites whose two plans launch a different number of kernels (the floor carries no launch cost): {}; today's plan takes the side with fewer launches at {}, more at {}, neither side at {}.\n",
        by_launch.iter().sum::<u64>(),
        by_launch[0],
        by_launch[1],
        by_launch[2]
    );
    let forced = rows().filter(|r| r.pair.class == Joinability::Forced);
    let refused = forced
        .clone()
        .filter(|r| r.pair.a_first.is_err() || r.pair.b_first.is_err())
        .count();
    let _ = writeln!(
        s,
        "Forced sites: {} with one order refused (a seed whose rule that plan refuses, or a node left uncovered), {} with one order leaving more nodes to placeholders.\n",
        refused,
        forced.count() - refused
    );
    let _ = writeln!(
        s,
        "## Per plan\n\n| model | device | run | matches | floor µs | identical | same-cost | priority-decided | forced | dearer |\n|---|---|---|---:|---:|---:|---:|---:|---:|---:|"
    );
    for p in &plans {
        let count = |c: Joinability| p.rows.iter().filter(|r| r.pair.class == c).count();
        let _ = writeln!(
            s,
            "| {} | {} | {} n={} | {} | {:.3} | {} | {} | {} | {} | {} |",
            p.model,
            p.device,
            p.run.mode.name(),
            p.run.rows,
            p.matches,
            p.floor_us,
            count(Joinability::Identical),
            count(Joinability::SameCost),
            count(Joinability::PriorityDecided),
            count(Joinability::Forced),
            p.rows.iter().filter(|r| r.pair.picks_dearer()).count()
        );
    }
    let why: BTreeMap<&(String, String), &str> =
        allowed.iter().map(|a| (&a.rules, a.why.as_str())).collect();
    let mut gated: BTreeMap<(String, String), (u64, u64)> = BTreeMap::new();
    for r in plans.iter().flat_map(|p| &p.rows).filter(|r| r.gated()) {
        let e = gated.entry(r.key()).or_default();
        e.0 += 1;
        e.1 += u64::from(r.pair.picks_dearer());
    }
    let _ = writeln!(
        s,
        "\n## Gated rule pairs\n\n| rule pair | sites | dearer | allowlisted: why |\n|---|---:|---:|---|"
    );
    for ((a, b), (n, d)) in &gated {
        let w = why
            .get(&(a.clone(), b.clone()))
            .copied()
            .unwrap_or("**missing**");
        let _ = writeln!(s, "| `{a}` × `{b}` | {n} | {d} | {w} |");
    }
    let _ = writeln!(
        s,
        "\n## Gated sites\n\nA is the match the greedy tries first. Δ = B-first − A-first floor, µs.\n\n| model | device | run | class | A | B | A-first | B-first | Δ | today | dearer | layers |\n|---|---|---|---|---|---|---:|---:|---:|---|---|---:|"
    );
    for p in &plans {
        for r in p.rows.iter().filter(|r| r.gated()) {
            let delta = match (&r.pair.a_first, &r.pair.b_first) {
                (Ok(a), Ok(b)) => format!("{:+.3}", b.cost - a.cost),
                _ => "–".into(),
            };
            let _ = writeln!(
                s,
                "| {} | {} | {} n={} | {} | `{}` {} | `{}` {} | {} | {} | {delta} | {} | {} | {} |",
                p.model,
                p.device,
                p.run.mode.name(),
                p.run.rows,
                r.pair.class.name(),
                r.pair.a.rule,
                r.site_a,
                r.pair.b.rule,
                r.site_b,
                cost(&r.pair.a_first),
                cost(&r.pair.b_first),
                pick(r.pair.current),
                if r.pair.picks_dearer() { "yes" } else { "no" },
                r.pair.occurrences
            );
        }
    }
    let _ = writeln!(
        s,
        "\n## Same floor, today launches more\n\nSame-cost sites where today's plan takes the side with more kernel launches: the candidates once the cost carries a launch term.\n\n| model | device | run | A | B | A-first launches | B-first launches | today | layers |\n|---|---|---|---|---|---:|---:|---|---:|"
    );
    for p in &plans {
        for r in p.rows.iter().filter(|r| r.pair.picks_more_launches()) {
            let l = |o: &Result<Outcome, String>| o.as_ref().map_or(0, |o| o.launches);
            let _ = writeln!(
                s,
                "| {} | {} | {} n={} | `{}` {} | `{}` {} | {} | {} | {} | {} |",
                p.model,
                p.device,
                p.run.mode.name(),
                p.run.rows,
                r.pair.a.rule,
                r.site_a,
                r.pair.b.rule,
                r.site_b,
                l(&r.pair.a_first),
                l(&r.pair.b_first),
                pick(r.pair.current),
                r.pair.occurrences
            );
        }
    }
    s
}

#[cfg(test)]
#[path = "critical_report_tests.rs"]
mod critical_report_tests;
