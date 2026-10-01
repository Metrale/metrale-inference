// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: The closing sections of a rendered Venn report, aggregated over every table: the
//! parameterization plan (each family change, ordered by the largest step share it touches), the
//! novel kernels, and the reused kernels still waiting for a microbench at the target point.
//!
//! Owner: metrale-circuit (venn).
//! Invariants: aggregation keys are ordered maps, so the sections are deterministic.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use super::families::ParamKind;
use super::report::VennReport;
use super::{Class, Finding, Run};

#[derive(Default)]
struct Agg {
    kind: BTreeSet<&'static str>,
    sites: BTreeSet<String>,
    best: (f64, Option<Run>),
}

impl Agg {
    fn add(&mut self, site: &str, share: f64, run: Run) {
        self.sites.insert(site.to_string());
        if self.best.1.is_none() || share > self.best.0 {
            self.best = (share, Some(run));
        }
    }
}

fn change(f: &Finding) -> String {
    f.diffs
        .iter()
        .filter(|d| d.kind != ParamKind::Runtime)
        .map(|d| format!("{} {} (have {})", d.param, d.target, d.other))
        .collect::<Vec<_>>()
        .join(", ")
}

fn ranked(map: BTreeMap<(String, String), Agg>) -> Vec<((String, String), Agg)> {
    let mut v: Vec<_> = map.into_iter().collect();
    v.sort_by(|a, b| {
        b.1.best
            .0
            .total_cmp(&a.1.best.0)
            .then_with(|| a.0.cmp(&b.0))
    });
    v
}

fn run_name(r: Option<Run>) -> String {
    r.map_or_else(String::new, |r| format!("{} n={}", r.mode.name(), r.rows))
}

/// 2026-09-29: Append the summary sections of `r` to `s`.
pub(super) fn summary(r: &VennReport, s: &mut String) {
    let mut plan: BTreeMap<(String, String), Agg> = BTreeMap::new();
    let mut novel: BTreeMap<(String, String), Agg> = BTreeMap::new();
    let mut unmeasured: BTreeMap<(String, String), Agg> = BTreeMap::new();
    for t in &r.tables {
        for row in &t.rows {
            let Some(p) = &row.primary else {
                novel
                    .entry((row.site.clone(), row.op.clone()))
                    .or_default()
                    .add(&row.site, row.share, t.run);
                continue;
            };
            if p.class == Class::SharedUnmeasured {
                let point: Vec<String> = p.point.iter().map(|(k, v)| format!("{k}={v}")).collect();
                unmeasured
                    .entry((p.family.clone(), point.join(" ")))
                    .or_default()
                    .add(&row.site, row.share, t.run);
            }
            for f in std::iter::once(p).chain(&row.also) {
                if matches!(
                    f.class,
                    Class::PolicyVariant | Class::ParameterizationOpportunity
                ) {
                    let a = plan.entry((f.family.clone(), change(f))).or_default();
                    a.kind.insert(f.class.name());
                    a.add(&row.site, row.share, t.run);
                }
            }
        }
    }
    s.push_str(
        "\n## Parameterization plan (ordered by the largest step share each change touches)\n\n",
    );
    s.push_str(
        "| # | Family | Change | Class | Sites | Max est. share |\n|---|---|---|---|---|---|\n",
    );
    for (i, ((family, what), a)) in ranked(plan).into_iter().enumerate() {
        let kinds: Vec<&str> = a.kind.into_iter().collect();
        let sites: Vec<String> = a.sites.into_iter().map(|x| format!("`{x}`")).collect();
        let _ = writeln!(
            s,
            "| {} | {family} | {what} | {} | {} | {:.1}% ({}) |",
            i + 1,
            kinds.join(", "),
            sites.join(", "),
            100.0 * a.best.0,
            run_name(a.best.1)
        );
    }
    s.push_str("\n## Novel kernels (build, prove, then microbench against the roofline)\n\n");
    if novel.is_empty() {
        s.push_str("None.\n");
    } else {
        s.push_str("| Site | Op | Max est. share |\n|---|---|---|\n");
        for ((site, op), a) in ranked(novel) {
            let _ = writeln!(
                s,
                "| `{site}` | `{op}` | {:.1}% ({}) |",
                100.0 * a.best.0,
                run_name(a.best.1)
            );
        }
    }
    s.push_str("\n## Reused, unmeasured at the target point (microbench in this order)\n\n");
    s.push_str("| # | Family | Point | Sites | Max est. share |\n|---|---|---|---|---|\n");
    for (i, ((family, point), a)) in ranked(unmeasured).into_iter().enumerate() {
        let sites: Vec<String> = a.sites.into_iter().map(|x| format!("`{x}`")).collect();
        let point = if point.is_empty() {
            "-".to_string()
        } else {
            point
        };
        let _ = writeln!(
            s,
            "| {} | {family} | {point} | {} | {:.1}% ({}) |",
            i + 1,
            sites.join(", "),
            100.0 * a.best.0,
            run_name(a.best.1)
        );
    }
}
