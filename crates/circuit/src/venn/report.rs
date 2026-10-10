// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Build a whole Venn report: fuse the compared models at every requested mode and
//! row count, classify and estimate every target node, group the nodes into sites (one template
//! node across its layers), rank them by estimated step share, and flag the layer kinds that
//! cannot batch rows.
//!
//! Owner: metrale-circuit (venn).
//! Invariants:
//! - The compared models are fused with every kernel their rules name counted as built (the
//!   offline view `met circuit show` takes). The target is fused only when it is a golden
//!   instance, whose rules cover it; otherwise its nodes are matched to families by op.
//! - 2026-10-10: A compared model that is not golden is planned with a placeholder for each
//!   node no rule covers (`crate::fuser_cover`, as `met circuit plan` does); those nodes run no
//!   family, so they are no usage, and the report lists their sites ([`Uncovered`]). A golden
//!   compared model is still fused strictly: a gap in its rules is an error.
//! - A flag's cost is the estimate of running the flagged nodes once per row minus running them
//!   once for all rows, so flags rank by the time the missing multi-row path costs.

use std::collections::BTreeMap;

use super::classify::{candidates, classify_node, usages};
use super::families::{EvidenceSource, Families};
use super::measurements::Measurements;
use super::roofline::node_cost;
use super::{Finding, Run, Subject, VennError};
use crate::Loaded;
use crate::fuser::{AvailableKernels, FusionPlan, fuse, section_of};
use crate::fuser_cover::{NOVEL_EMITTER, fuse_covering};
use crate::instances::Instance;
use crate::ir::{Circuit, LayerKind, NodeIdx};
use crate::rules::Mode;

/// 2026-09-29: One side of the diagram as loaded.
#[derive(Debug, Clone, Copy)]
pub struct Side<'a> {
    /// 2026-09-29: The instance.
    pub instance: &'a Instance,
    /// 2026-09-29: Its circuit and rules.
    pub loaded: &'a Loaded,
}

/// 2026-09-29: Everything a report is built from.
#[derive(Debug, Clone)]
pub struct VennInputs<'a> {
    /// 2026-09-29: The model being added.
    pub target: Side<'a>,
    /// 2026-09-29: The supported models it is compared with.
    pub against: Vec<Side<'a>>,
    /// 2026-09-29: The kernel-family manifest.
    pub families: &'a Families,
    /// 2026-09-29: measurements.toml rows.
    pub measurements: &'a Measurements,
    /// 2026-09-29: Modes and row counts, in report order.
    pub runs: Vec<Run>,
    /// 2026-09-29: The command that regenerates the report, printed in its header.
    pub command: String,
}

/// 2026-09-29: One site: a template node across the layers that instantiate it.
#[derive(Debug, Clone)]
pub struct Row {
    /// 2026-09-29: `block.node`.
    pub site: String,
    /// 2026-09-29: Op name.
    pub op: String,
    /// 2026-09-29: Weight format, for weight-reading ops.
    pub weight: Option<String>,
    /// 2026-09-29: First-input format.
    pub activation: Option<String>,
    /// 2026-09-29: Nodes (layers) it covers.
    pub count: usize,
    /// 2026-09-29: Estimated time over those nodes, microseconds.
    pub time_us: f64,
    /// 2026-09-29: `time_us` over the table's total.
    pub share: f64,
    /// 2026-09-29: The classification; `None` is novel.
    pub primary: Option<Finding>,
    /// 2026-09-29: Other families' opportunities.
    pub also: Vec<Finding>,
}

/// 2026-09-29: One mode and row count.
#[derive(Debug, Clone)]
pub struct Table {
    /// 2026-09-29: Mode and rows.
    pub run: Run,
    /// 2026-09-29: Estimated step time of the target, microseconds.
    pub total_us: f64,
    /// 2026-09-29: Sites, by descending share.
    pub rows: Vec<Row>,
}

/// 2026-09-29: A layer kind that cannot batch rows in a mode.
#[derive(Debug, Clone)]
pub struct Flag {
    /// 2026-09-29: Layer kind, or `None` for the prologue, epilogue and draft head.
    pub layer_kind: Option<LayerKind>,
    /// 2026-09-29: Mode and rows.
    pub run: Run,
    /// 2026-09-29: Where the fact comes from: a legacy citation, or the circuit's sites.
    pub source: String,
    /// 2026-09-29: One line.
    pub note: String,
    /// 2026-09-29: Estimated time the per-row fallback adds, microseconds.
    pub added_us: f64,
    /// 2026-09-29: `added_us` over the multi-row step estimate.
    pub share: f64,
}

/// 2026-09-29: A built report.
#[derive(Debug, Clone)]
pub struct VennReport {
    /// 2026-09-29: The target instance.
    pub target: Instance,
    /// 2026-09-29: Its circuit's arch and layer kinds.
    pub arch: String,
    /// 2026-09-29: Its layer kinds.
    pub layer_kinds: Vec<LayerKind>,
    /// 2026-09-29: Weight-reading sites: `site` to (weight, activation).
    pub precision: BTreeMap<String, (String, String)>,
    /// 2026-09-29: The compared instances.
    pub against: Vec<Instance>,
    /// 2026-09-29: Estimate constants.
    pub roofline: super::families::Roofline,
    /// 2026-09-29: Flags, by descending cost.
    pub flags: Vec<Flag>,
    /// 2026-09-29: Tables, in run order.
    pub tables: Vec<Table>,
    /// 2026-09-29: Measured rows cited by the tables' evidence (`key` to % of floor).
    pub cited: BTreeMap<String, f64>,
    /// 2026-10-10: Per compared model that is not golden and run, the sites no rule covers
    /// (planned by placeholders, so they count as no usage); empty when every compared model is
    /// golden.
    pub uncovered: Vec<Uncovered>,
    /// 2026-09-29: The regenerating command.
    pub command: String,
}

/// 2026-10-10: The sites of one compared model that no rule covers in one run.
#[derive(Debug, Clone)]
pub struct Uncovered {
    /// 2026-10-10: The compared recipe.
    pub recipe: String,
    /// 2026-10-10: Mode and rows.
    pub run: Run,
    /// 2026-10-10: `block.node` sites, sorted.
    pub sites: Vec<String>,
}

fn fused(side: &Side<'_>, run: Run) -> Result<FusionPlan, VennError> {
    let l = side.loaded;
    fuse(
        &l.circuit,
        &l.rules,
        &AvailableKernels::all_named_by(&l.rules),
        &side.instance.policy,
        run.mode,
        run.rows,
    )
    .map_err(|e| run_error(side, run, e.to_string()))
}

/// 2026-10-10: A compared model's plan. A golden instance is fused strictly (its rules cover
/// every node); any other is planned the way `met circuit plan` plans a checkpoint, with a
/// placeholder for each node no rule covers, and those nodes' sites are returned.
fn fused_compared(side: &Side<'_>, run: Run) -> Result<(FusionPlan, Option<Uncovered>), VennError> {
    if side.instance.golden {
        return Ok((fused(side, run)?, None));
    }
    let l = side.loaded;
    let mut rules = l.rules.clone();
    let mut refused = Vec::new();
    let plan = fuse_covering(
        &l.circuit,
        &mut rules,
        &AvailableKernels::all_named_by(&l.rules),
        &side.instance.policy,
        run.mode,
        run.rows,
        &mut refused,
    )
    .map_err(|e| run_error(side, run, e.to_string()))?;
    let sites: std::collections::BTreeSet<String> = plan
        .groups
        .iter()
        .filter(|g| g.emitter == NOVEL_EMITTER)
        .flat_map(|g| g.nodes.iter().map(|&n| site_of(&l.circuit, n)))
        .collect();
    let uncovered = (!sites.is_empty()).then(|| Uncovered {
        recipe: side.instance.recipe.clone(),
        run,
        sites: sites.into_iter().collect(),
    });
    Ok((plan, uncovered))
}

fn run_error(side: &Side<'_>, run: Run, e: String) -> VennError {
    VennError::Load(format!(
        "{} {} n={}: {e}",
        side.instance.recipe,
        run.mode.name(),
        run.rows
    ))
}

fn in_section(c: &Circuit, mode: Mode) -> Vec<NodeIdx> {
    let section = section_of(mode);
    c.blocks
        .iter()
        .filter(|b| b.section == section)
        .flat_map(|b| b.first..b.end)
        .collect()
}

fn site_of(c: &Circuit, n: NodeIdx) -> String {
    site(&c.nodes[n])
}

/// 2026-09-30: `block.local`, prefixed `draft.` in the draft head, whose blocks may reuse a
/// main-stack template (the Nemotron-H draft MoE is the `moe` block at `mtp.layers.1`).
fn site(node: &crate::ir::Node) -> String {
    let draft = if node.id.starts_with("draft.") {
        "draft."
    } else {
        ""
    };
    format!("{draft}{}.{}", node.block, node.local)
}

/// 2026-09-29: Build the report.
pub fn build(inp: &VennInputs<'_>) -> Result<VennReport, VennError> {
    let mut cited = BTreeMap::new();
    for f in &inp.families.families {
        for e in &f.evidence {
            if let EvidenceSource::Measurement(key) = &e.source {
                let m = inp
                    .measurements
                    .get(key)
                    .ok_or_else(|| VennError::UnknownMeasurement {
                        family: f.id.clone(),
                        key: key.clone(),
                    })?;
                cited.insert(key.clone(), m.pct_of_floor);
            }
        }
    }
    let tc = &inp.target.loaded.circuit;
    let settings = &inp.target.instance.policy.settings;
    let mut tables = Vec::with_capacity(inp.runs.len());
    let mut flags = Vec::new();
    let mut uncovered = Vec::new();
    for &run in &inp.runs {
        let scope = in_section(tc, run.mode);
        if scope.is_empty() {
            return Err(VennError::Run(format!(
                "the target circuit has no {} section",
                run.mode.name()
            )));
        }
        let mut plans: Vec<FusionPlan> = Vec::with_capacity(inp.against.len());
        for s in &inp.against {
            let (plan, gaps) = fused_compared(s, run)?;
            plans.push(plan);
            uncovered.extend(gaps);
        }
        let target_plan = if inp.target.instance.golden {
            Some(fused(&inp.target, run)?)
        } else {
            None
        };
        let against: Vec<Subject<'_>> = inp
            .against
            .iter()
            .zip(&plans)
            .map(|(s, p)| Subject {
                recipe: &s.instance.recipe,
                circuit: &s.loaded.circuit,
                settings: &s.instance.policy.settings,
                plan: Some(p),
            })
            .collect();
        let used = usages(&against, inp.families)?;
        let target = Subject {
            recipe: &inp.target.instance.recipe,
            circuit: tc,
            settings,
            plan: target_plan.as_ref(),
        };
        let mut by_site: BTreeMap<(String, String, Option<String>, Option<String>), Row> =
            BTreeMap::new();
        let mut total = 0.0;
        for &n in &scope {
            let node = &tc.nodes[n];
            let cost = node_cost(
                tc,
                node,
                run.mode,
                run.rows,
                settings,
                &inp.families.roofline,
            )?;
            total += cost.time_us;
            let weight = node.weight.map(|w| w.name());
            let activation = node.inputs.first().map(|&e| tc.edges[e].format.name());
            let key = (
                site_of(tc, n),
                node.op.name(),
                weight.clone(),
                activation.clone(),
            );
            if let Some(row) = by_site.get_mut(&key) {
                row.count += 1;
                row.time_us += cost.time_us;
                continue;
            }
            let (primary, also) = classify_node(&target, n, run.rows, &used, inp.families)?;
            by_site.insert(
                key.clone(),
                Row {
                    site: key.0,
                    op: key.1,
                    weight,
                    activation,
                    count: 1,
                    time_us: cost.time_us,
                    share: 0.0,
                    primary,
                    also,
                },
            );
        }
        let mut rows: Vec<Row> = by_site.into_values().collect();
        for r in &mut rows {
            r.share = if total > 0.0 { r.time_us / total } else { 0.0 };
        }
        rows.sort_by(|a, b| {
            b.time_us
                .total_cmp(&a.time_us)
                .then_with(|| a.site.cmp(&b.site))
        });
        if run.rows > 1 {
            flags.extend(flags_for(inp, tc, &scope, run, total)?);
        }
        tables.push(Table {
            run,
            total_us: total,
            rows,
        });
    }
    flags.sort_by(|a, b| {
        b.added_us
            .total_cmp(&a.added_us)
            .then_with(|| a.run.cmp(&b.run))
            .then_with(|| a.source.cmp(&b.source))
    });
    let mut precision = BTreeMap::new();
    for n in &tc.nodes {
        if let Some(w) = n.weight {
            let act = n
                .inputs
                .first()
                .map(|&e| tc.edges[e].format.name())
                .unwrap_or_default();
            precision.insert(site(n), (w.name(), act));
        }
    }
    Ok(VennReport {
        target: inp.target.instance.clone(),
        arch: tc.arch.clone(),
        layer_kinds: tc.layer_kinds.clone(),
        precision,
        against: inp.against.iter().map(|s| s.instance.clone()).collect(),
        roofline: inp.families.roofline,
        flags,
        tables,
        cited,
        uncovered,
        command: inp.command.clone(),
    })
}

fn penalty(
    inp: &VennInputs<'_>,
    tc: &Circuit,
    nodes: &[NodeIdx],
    run: Run,
) -> Result<f64, VennError> {
    let settings = &inp.target.instance.policy.settings;
    let r = &inp.families.roofline;
    let mut added = 0.0;
    for &n in nodes {
        let one = node_cost(tc, &tc.nodes[n], run.mode, 1, settings, r)?;
        let all = node_cost(tc, &tc.nodes[n], run.mode, run.rows, settings, r)?;
        added += run.rows as f64 * one.time_us - all.time_us;
    }
    Ok(added)
}

fn flags_for(
    inp: &VennInputs<'_>,
    tc: &Circuit,
    scope: &[NodeIdx],
    run: Run,
    total: f64,
) -> Result<Vec<Flag>, VennError> {
    let kind_of = |n: NodeIdx| tc.nodes[n].layer.map(|l| tc.layer_kinds[l]);
    let mut out = Vec::new();
    for l in &inp.families.legacy {
        if l.arch != tc.arch || !l.per_sequence.contains(&run.mode) || run.rows <= l.rows_above {
            continue;
        }
        let nodes: Vec<NodeIdx> = scope
            .iter()
            .copied()
            .filter(|&n| kind_of(n) == Some(l.layer_kind))
            .filter(|&n| l.sites.is_empty() || l.sites.contains(&site_of(tc, n)))
            .collect();
        if nodes.is_empty() {
            continue;
        }
        let added = penalty(inp, tc, &nodes, run)?;
        out.push(Flag {
            layer_kind: Some(l.layer_kind),
            run,
            source: format!("legacy: {}", l.cite),
            note: l.note.clone(),
            added_us: added,
            share: if total > 0.0 { added / total } else { 0.0 },
        });
    }
    let mut by_kind: BTreeMap<Option<LayerKind>, Vec<NodeIdx>> = BTreeMap::new();
    for &n in scope {
        let node = &tc.nodes[n];
        let fams = inp.families;
        if candidates(fams, tc, node, run.rows).is_empty()
            && !candidates(fams, tc, node, 1).is_empty()
        {
            by_kind.entry(kind_of(n)).or_default().push(n);
        }
    }
    for (kind, nodes) in by_kind {
        let added = penalty(inp, tc, &nodes, run)?;
        let sites: std::collections::BTreeSet<String> =
            nodes.iter().map(|&n| site_of(tc, n)).collect();
        out.push(Flag {
            layer_kind: kind,
            run,
            source: format!(
                "circuit: {}",
                sites.into_iter().collect::<Vec<_>>().join(", ")
            ),
            note: "every family that runs these ops covers one row per launch".into(),
            added_us: added,
            share: if total > 0.0 { added / total } else { 0.0 },
        });
    }
    Ok(out)
}
