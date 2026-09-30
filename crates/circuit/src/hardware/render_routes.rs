// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The runtime routes of a hardware plan, rendered. A report lists each route that
//! applies to a report run, with the route's estimate and launches beside the primary arm's and
//! the rules it runs instead. A plan (`--format plan`) is followed by each route's plan.
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - A report or plan with no applicable route renders exactly as it did before routes existed.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use super::gaps::{GapTable, gap_table};
use super::plan::{Planned, Resolved};
use super::runtime::RuntimeRoute;
use super::{HwError, HwReport, OnePlan};
use crate::fuser::{FusionPlan, Policy};
use crate::ir::Circuit;

/// 2026-09-30: One route at one report run, estimated.
#[derive(Debug, Clone)]
pub struct RouteRow {
    /// 2026-09-30: The route.
    pub route: RuntimeRoute,
    /// 2026-09-30: The primary arm's table at this run.
    pub primary_us: f64,
    /// 2026-09-30: The primary arm's launches.
    pub primary_launches: u64,
    /// 2026-09-30: The route's table at this run.
    pub table: GapTable,
}

/// 2026-09-30: Every route of every table in `tables`, estimated under the route's policy.
pub fn route_rows(
    r: &Resolved,
    c: &Circuit,
    policy: &Policy,
    label: &str,
    tables: &[GapTable],
) -> Result<Vec<RouteRow>, HwError> {
    let mut out = Vec::new();
    for t in tables {
        for rp in &t.planned.routes {
            let settings = rp.route.policy(policy).settings;
            out.push(RouteRow {
                route: rp.route.clone(),
                primary_us: t.total_us,
                primary_launches: t.planned.plan.launches(),
                table: gap_table(r, c, &settings, label, rp.planned.clone())?,
            });
        }
    }
    Ok(out)
}

fn rules_of(p: &FusionPlan) -> BTreeSet<&str> {
    p.groups.iter().map(|g| g.rule.as_str()).collect()
}

/// 2026-09-30: The rules `route` runs that `primary` does not, and those it replaces.
fn swapped(primary: &FusionPlan, route: &FusionPlan) -> (Vec<String>, Vec<String>) {
    let (a, b) = (rules_of(primary), rules_of(route));
    (
        b.difference(&a).map(|s| format!("`{s}`")).collect(),
        a.difference(&b).map(|s| format!("`{s}`")).collect(),
    )
}

/// 2026-09-30: The "Runtime routes" section; nothing when no route applies.
pub(super) fn routes_section(s: &mut String, r: &HwReport) {
    if r.routes.is_empty() {
        return;
    }
    let _ = writeln!(
        s,
        "## Runtime routes\n\nConditions the engine checks at run time, under which a step runs another arm than the plan's (FUSIONS.toml `[[runtime]]`). The estimates above are the primary arm's; each route is planned and estimated beside it.\n"
    );
    let mut described = BTreeSet::new();
    for row in &r.routes {
        let x = &row.route;
        if described.insert(x.id.as_str()) {
            let _ = writeln!(
                s,
                "- `{}`: when {}; planned as `{}` ({}).",
                x.id,
                x.why,
                x.plans_as_text(),
                x.cite
            );
        }
    }
    let _ = writeln!(
        s,
        "\n| case | route | time (ms) | route time (ms) | launches | route launches | route runs | instead of |\n|---|---|---:|---:|---:|---:|---|---|"
    );
    for row in &r.routes {
        let p = &row.table.planned.plan;
        let primary = r
            .tables
            .iter()
            .find(|t| t.planned.run == row.table.planned.run)
            .map(|t| &t.planned.plan);
        let (runs, instead) = primary.map(|q| swapped(q, p)).unwrap_or_default();
        let _ = writeln!(
            s,
            "| decode C={} ({} n={}) | `{}` | {:.3} | {:.3} | {} | {} | {} | {} |",
            p.rows,
            p.mode.name(),
            p.rows,
            row.route.id,
            row.primary_us / 1e3,
            row.table.total_us / 1e3,
            row.primary_launches,
            p.launches(),
            runs.join(", "),
            instead.join(", ")
        );
    }
    let _ = writeln!(s);
}

/// 2026-09-30: `one`'s plan text, followed by each applicable route's plan under a heading that
/// names the route, its condition and the settings it is planned as.
pub fn plan_text(circuit: &Circuit, one: &OnePlan) -> String {
    let mut s = crate::render::render(circuit, &one.planned.plan, &one.header);
    for rp in &one.planned.routes {
        s.push_str(&route_plan_text(circuit, one, &rp.route, &rp.planned));
    }
    s
}

fn route_plan_text(circuit: &Circuit, one: &OnePlan, route: &RuntimeRoute, p: &Planned) -> String {
    let header = super::model::with_settings(&one.header, &route.policy(&one.policy));
    format!(
        "\n# runtime route `{}`: when {}; planned as `{}` ({})\n\n{}",
        route.id,
        route.why,
        route.plans_as_text(),
        route.cite,
        crate::render::render(circuit, &p.plan, &header)
    )
}
