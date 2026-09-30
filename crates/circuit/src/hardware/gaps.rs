// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The Venn/gap table of one plan on one device: every node of the mode's section,
//! grouped by site (a template node across its layers), classified against the families the
//! device can run, and ranked by its share of the device's estimated step time.
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - One classifier: `venn::classify::classify_node`, over the class's families (points
//!   re-resolved on the class, evidence only where measured on it). A planned group is
//!   classified through its kernels; a placeholder group through the families that implement
//!   the op, and is Novel when none does.
//! - A site's execution line states how the device runs the node's DECLARED formats
//!   ([`super::exec`]); nothing is re-planned at a wider format.
//! - Flags are plan groups that loop per row at more than one row, costed as the time the loop
//!   adds over one multi-row launch.

use std::collections::{BTreeMap, BTreeSet};

use super::HwError;
use super::estimate::activation_of;
use super::exec::{Exec, exec_of};
use super::plan::{NOVEL_EMITTER, Planned, Resolved};
use crate::fuser::section_of;
use crate::ir::{Circuit, LayerKind, NodeIdx};
use crate::rules::Repeat;
use crate::venn::classify::{candidates, classify_node};
use crate::venn::roofline::node_cost;
use crate::venn::{Class, Subject, VennError};

/// 2026-09-30: One site.
#[derive(Debug, Clone)]
pub struct GapRow {
    /// 2026-09-30: `block.node`.
    pub site: String,
    /// 2026-09-30: Op name.
    pub op: String,
    /// 2026-09-30: Declared weight and activation, for weight-reading ops.
    pub formats: Option<(String, String)>,
    /// 2026-09-30: How the device executes the declared formats (weight-reading ops).
    pub exec: Option<Exec>,
    /// 2026-09-30: Nodes (layers).
    pub count: usize,
    /// 2026-09-30: Estimated time, microseconds.
    pub time_us: f64,
    /// 2026-09-30: Share of the step.
    pub share: f64,
    /// 2026-09-30: Class.
    pub class: Class,
    /// 2026-09-30: Family of the primary finding.
    pub family: Option<String>,
    /// 2026-09-30: What runs it (kernels or rule), or why nothing does.
    pub detail: String,
    /// 2026-09-30: Differing parameters of the primary finding.
    pub diffs: Vec<String>,
}

/// 2026-09-30: A group that loops per row.
#[derive(Debug, Clone)]
pub struct Flag {
    /// 2026-09-30: Layer kind (`None`: prologue / epilogue).
    pub layer_kind: Option<LayerKind>,
    /// 2026-09-30: Sites.
    pub sites: BTreeSet<String>,
    /// 2026-09-30: Rules.
    pub rules: BTreeSet<String>,
    /// 2026-09-30: Time the loop adds, microseconds.
    pub added_us: f64,
    /// 2026-09-30: Over the step.
    pub share: f64,
}

/// 2026-09-30: One plan's table.
#[derive(Debug, Clone)]
pub struct GapTable {
    /// 2026-09-30: The plan.
    pub planned: Planned,
    /// 2026-09-30: Estimated step, microseconds (per-row loops included).
    pub total_us: f64,
    /// 2026-09-30: Sites, by descending share.
    pub rows: Vec<GapRow>,
    /// 2026-09-30: Per-row fallbacks, by descending cost.
    pub flags: Vec<Flag>,
}

impl GapTable {
    /// 2026-09-30: Share of the step whose class is one of `classes`.
    pub fn share_of(&self, classes: &[Class]) -> f64 {
        self.rows
            .iter()
            .filter(|r| classes.contains(&r.class))
            .map(|r| r.share)
            .sum()
    }
}

fn site_of(c: &Circuit, n: NodeIdx) -> String {
    format!("{}.{}", c.nodes[n].block, c.nodes[n].local)
}

/// 2026-09-30: Build the table of `planned` for the circuit `c` under `settings` on `r`.
pub fn gap_table(
    r: &Resolved,
    c: &Circuit,
    settings: &BTreeMap<String, String>,
    recipe: &str,
    planned: Planned,
) -> Result<GapTable, HwError> {
    let plan = &planned.plan;
    let rf = &r.roofline.roofline;
    let (mode, rows) = (plan.mode, plan.rows);
    let with_plan = Subject {
        recipe,
        circuit: c,
        settings,
        plan: Some(plan),
    };
    let bare = Subject {
        plan: None,
        ..with_plan
    };
    let group_of: BTreeMap<NodeIdx, usize> = plan
        .groups
        .iter()
        .enumerate()
        .flat_map(|(g, grp)| grp.nodes.iter().map(move |&n| (n, g)))
        .collect();
    let section = section_of(mode);
    let scope: Vec<NodeIdx> = c
        .blocks
        .iter()
        .filter(|b| b.section == section)
        .flat_map(|b| b.first..b.end)
        .collect();
    let mut by_site: BTreeMap<(String, String), GapRow> = BTreeMap::new();
    let mut flags: BTreeMap<Option<LayerKind>, Flag> = BTreeMap::new();
    let mut total = 0.0;
    for &n in &scope {
        let node = &c.nodes[n];
        let g = &plan.groups[*group_of
            .get(&n)
            .ok_or_else(|| HwError::Plan(format!("node `{}` is in no group", node.id)))?];
        let one = node_cost(c, node, mode, 1, settings, rf)?.time_us;
        let all = node_cost(c, node, mode, rows, settings, rf)?.time_us;
        let per_row = matches!(g.repeat, Repeat::PerRow | Repeat::PerRowButLast) && rows > 1;
        let time = if per_row { rows as f64 * one } else { all };
        total += time;
        if per_row {
            let kind = node.layer.map(|l| c.layer_kinds[l]);
            let f = flags.entry(kind).or_insert_with(|| Flag {
                layer_kind: kind,
                sites: BTreeSet::new(),
                rules: BTreeSet::new(),
                added_us: 0.0,
                share: 0.0,
            });
            f.sites.insert(site_of(c, n));
            f.rules.insert(g.rule.clone());
            f.added_us += time - all;
        }
        let act = activation_of(c, node);
        let formats = node.weight.zip(act).map(|(w, a)| (w.name(), a.name()));
        let key = (site_of(c, n), format!("{formats:?}"));
        if let Some(row) = by_site.get_mut(&key) {
            row.count += 1;
            row.time_us += time;
            continue;
        }
        let novel = g.emitter == NOVEL_EMITTER;
        let classified = if novel {
            classify_node(&bare, n, rows, &[], &r.families)
        } else {
            classify_node(&with_plan, n, rows, &[], &r.families)
        };
        let (class, family, diffs, detail) = match classified {
            Ok((Some(f), _)) => {
                let detail = if novel {
                    format!(
                        "no rule of this class covers it; family `{}` implements the op",
                        f.family
                    )
                } else {
                    kernels_of(g)
                };
                let diffs = f
                    .diffs
                    .iter()
                    .map(|d| format!("{} {}->{}", d.param, d.other, d.target))
                    .collect();
                // 2026-09-30: A family that no rule of the class selects has no evidence in
                // this plan, whatever it measured elsewhere.
                let class = if novel && f.class == Class::Shared {
                    Class::SharedUnmeasured
                } else {
                    f.class
                };
                (class, Some(f.family), diffs, detail)
            }
            Ok((None, _)) => {
                let one_row: Vec<&str> = candidates(&r.families, c, node, 1)
                    .into_iter()
                    .map(|f| f.id.as_str())
                    .collect();
                let detail = if one_row.is_empty() {
                    "no family available on this device implements it".to_string()
                } else {
                    format!(
                        "no rule of this class covers it; {} run it one row per launch only",
                        one_row.join(", ")
                    )
                };
                (Class::Novel, None, Vec::new(), detail)
            }
            Err(VennError::UnmappedKernel { kernels, .. }) => (
                Class::SharedUnmeasured,
                None,
                Vec::new(),
                format!("{kernels} (in no kernel family)"),
            ),
            Err(VennError::Load(msg)) if !novel => (
                Class::SharedUnmeasured,
                None,
                Vec::new(),
                format!("{} (unclassified: {msg})", kernels_of(g)),
            ),
            Err(e) => return Err(HwError::Plan(e.to_string())),
        };
        by_site.insert(
            key.clone(),
            GapRow {
                site: key.0,
                op: node.op.name(),
                exec: node.weight.zip(act).map(|(w, a)| exec_of(&r.device, w, a)),
                formats,
                count: 1,
                time_us: time,
                share: 0.0,
                class,
                family,
                detail,
                diffs,
            },
        );
    }
    let mut rows_out: Vec<GapRow> = by_site.into_values().collect();
    for row in &mut rows_out {
        row.share = if total > 0.0 {
            row.time_us / total
        } else {
            0.0
        };
    }
    rows_out.sort_by(|a, b| {
        b.time_us
            .total_cmp(&a.time_us)
            .then_with(|| a.site.cmp(&b.site))
    });
    let mut flags: Vec<Flag> = flags.into_values().collect();
    for f in &mut flags {
        f.share = if total > 0.0 { f.added_us / total } else { 0.0 };
    }
    flags.sort_by(|a, b| b.added_us.total_cmp(&a.added_us));
    Ok(GapTable {
        planned,
        total_us: total,
        rows: rows_out,
        flags,
    })
}

fn kernels_of(g: &crate::fuser::Group) -> String {
    if g.kernels.is_empty() {
        format!("({} emitter) rule={}", g.emitter, g.rule)
    } else {
        let k: Vec<String> = g.kernels.iter().map(|k| k.to_string()).collect();
        format!("{} rule={}", k.join(" + "), g.rule)
    }
}
