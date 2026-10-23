// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Per-rank plans of a multi-device deployment (`[instance.parallel]`): every rank's circuit fused at
//! the report runs and the draft head, and the report section that shows where the ranks' plans
//! differ (a rank whose slice takes another kernel, a site whose launches change with the slice).
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - Each rank is planned exactly as the whole-model circuit is ([`super::plan::fuse_on`]), with the
//!   same class, policy and rules; only its dims differ.
//! - The section lists every site whose kernels differ between any two ranks, so "the ranks run
//!   the same kernels" is a statement the report makes, not an assumption.

use crate::prelude::*;

use alloc::collections::{BTreeMap, BTreeSet};
use core::fmt::Write as _;

use super::HwError;
use super::plan::{Planned, Resolved, fuse_on, report_runs};
use crate::fuser::Policy;
use crate::ir::{Circuit, Section};
use crate::rules::Mode;
use crate::venn::Run;

/// 2026-10-23: One rank's plans, in [`rank_runs`] order.
#[derive(Debug, Clone)]
pub struct RankPlans {
    /// 2026-10-23: The rank.
    pub rank: u64,
    /// 2026-10-23: Its plans.
    pub plans: Vec<Planned>,
}

/// 2026-10-23: The runs each rank is planned at: the report runs, then the one-row draft head when
/// the circuit has one (every speculative step runs it beside the verify).
pub fn rank_runs(c: &Circuit) -> Vec<Run> {
    let mut runs = report_runs();
    if c.blocks.iter().any(|b| b.section == Section::Draft) {
        runs.push(Run {
            mode: Mode::Draft,
            rows: 1,
        });
    }
    runs
}

/// 2026-10-23: Every rank circuit of `ranks` fused at [`rank_runs`] under `policy` on `r`.
pub fn plan_ranks(
    r: &Resolved,
    ranks: &[Circuit],
    policy: &Policy,
) -> Result<Vec<RankPlans>, HwError> {
    ranks
        .iter()
        .zip(0u64..)
        .map(|(c, rank)| {
            let plans = rank_runs(c)
                .into_iter()
                .map(|run| fuse_on(r, c, policy, run))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| HwError::Plan(format!("rank {rank}: {e}")))?;
            Ok(RankPlans { rank, plans })
        })
        .collect()
}

/// 2026-10-23: Per site of `p` (template block and member locals), the multiset of what its groups
/// launch: `rule: kernels` to the number of groups.
fn sites(c: &Circuit, p: &Planned) -> BTreeMap<String, BTreeMap<String, u64>> {
    let mut out: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
    for g in &p.plan.groups {
        let what = format!("{}: {}", g.rule, super::render::group_kernels(g));
        *out.entry(super::render::group_site(c, g))
            .or_default()
            .entry(what)
            .or_insert(0) += 1;
    }
    out
}

/// 2026-10-23: The "Parallel ranks" section of the report; nothing for a single-device model.
pub(super) fn section(
    s: &mut String,
    whole: &Circuit,
    ranks: &[Circuit],
    layout: Option<&crate::parallel::Layout>,
    plans: &[RankPlans],
) {
    let Some(layout) = layout.filter(|_| !ranks.is_empty()) else {
        return;
    };
    let _ = writeln!(
        s,
        "## Parallel ranks\n\nServed over {} ranks; each rank's circuit is the one circuit at its slice of \
         every split part, its collectives derived (the circuit's `[parallel]` spec, \
         kernels/circuits/INSTANCES.toml `[instance.parallel]`, crates/circuit/src/parallel.rs). Parts \
         (degree, group): {}. Dims not listed are replicated.\n",
        ranks.len(),
        layout
            .parts
            .iter()
            .map(|(d, p)| {
                let g: Vec<String> = p.group.iter().map(u64::to_string).collect();
                format!("`{d}` {} {{{}}}", p.degree, g.join(","))
            })
            .collect::<Vec<_>>()
            .join(", ")
    );
    let split: BTreeSet<&String> = ranks
        .iter()
        .flat_map(|c| c.dims.iter().filter(|(k, v)| whole.dims.get(*k) != Some(v)))
        .map(|(k, _)| k)
        .collect();
    let head: Vec<String> = (0..ranks.len()).map(|r| format!("rank {r}")).collect();
    let _ = writeln!(s, "| dim | whole | {} |", head.join(" | "));
    let _ = writeln!(s, "|---|---:|{}", "---:|".repeat(ranks.len()));
    for d in &split {
        let vals: Vec<String> = ranks.iter().map(|c| c.dims[*d].to_string()).collect();
        let _ = writeln!(s, "| {d} | {} | {} |", whole.dims[*d], vals.join(" | "));
    }
    let _ = writeln!(s);
    let Some(first) = plans.first() else {
        return;
    };
    let _ = writeln!(s, "| run | launches per rank | sites that differ |\n|---|---|---:|");
    let mut differing = Vec::new();
    for (i, p0) in first.plans.iter().enumerate() {
        let per: Vec<BTreeMap<String, BTreeMap<String, u64>>> = plans
            .iter()
            .zip(ranks)
            .map(|(rp, c)| sites(c, &rp.plans[i]))
            .collect();
        let keys: BTreeSet<&String> = per.iter().flat_map(|m| m.keys()).collect();
        let mut n = 0;
        for k in keys {
            let cells: Vec<Option<&BTreeMap<String, u64>>> = per.iter().map(|m| m.get(k)).collect();
            if cells.windows(2).all(|w| w[0] == w[1]) {
                continue;
            }
            n += 1;
            differing.push((p0.run, k.clone(), cells.iter().map(|c| cell(*c)).collect::<Vec<_>>()));
        }
        let launches: Vec<String> = plans
            .iter()
            .map(|rp| rp.plans[i].plan.launches().to_string())
            .collect();
        let _ = writeln!(
            s,
            "| {} n={} | {} | {n} |",
            p0.run.mode.name(),
            p0.run.rows,
            launches.join(" / ")
        );
    }
    let _ = writeln!(s);
    if differing.is_empty() {
        let _ = writeln!(s, "Every rank runs the same kernels at every run.\n");
        return;
    }
    let _ = writeln!(s, "Sites whose kernels differ between ranks (`rule: kernels` x groups):\n");
    let _ = writeln!(s, "| run | site | {} |", head.join(" | "));
    let _ = writeln!(s, "|---|---|{}", "---|".repeat(ranks.len()));
    for (run, site, cells) in differing {
        let _ = writeln!(
            s,
            "| {} n={} | `{site}` | {} |",
            run.mode.name(),
            run.rows,
            cells.join(" | ")
        );
    }
    let _ = writeln!(s);
}

fn cell(c: Option<&BTreeMap<String, u64>>) -> String {
    match c {
        None => "(none)".into(),
        Some(m) => m
            .iter()
            .map(|(what, n)| format!("`{what}` x{n}"))
            .collect::<Vec<_>>()
            .join("<br>"),
    }
}
