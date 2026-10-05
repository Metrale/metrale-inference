// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The Latent Kernel Blueprint on one hardware class, for one model: which
//! generators (kernel families) and relations (`bit_identical` rules) the class's plans use,
//! the LKB coverage of the step and its measured part, and the LKB residual (the class's own
//! kernels that are no family's point). Definitions: book/src/architecture/lkb.md.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - Nothing here recomputes a plan or a share: coverage reads the hardware report's gap
//!   tables ([`crate::hardware::HwReport`]), so `met circuit lkb` and `met circuit plan`
//!   cannot disagree.
//! - Pure: the only files read are `KERNEL.toml` shadow tables, through [`Repo`].
//! - "LKB residual" is never shortened to "residual" in names or output: the residual stream
//!   and the `residual_add` op already own that word.

mod laxity;
mod render;

#[cfg(test)]
mod lkb_tests;

use std::collections::{BTreeMap, BTreeSet};

use crate::hardware::plan::NOVEL_EMITTER;
use crate::hardware::{HwError, HwReport};
use crate::rules::Numerics;
use crate::venn::families::How;
use crate::venn::repo::Repo;
use crate::venn::{Class, Run};

pub use laxity::{GroupLaxity, PlanLaxity, plan_laxity};
pub use render::{render_markdown, render_toml, report_section};

/// 2026-10-05: Coverage of one report run.
#[derive(Debug, Clone, PartialEq)]
pub struct Coverage {
    /// 2026-10-05: Mode and rows.
    pub run: Run,
    /// 2026-10-05: Share of the step run by kernels of families (`1 - uncovered`).
    pub lkb: f64,
    /// 2026-10-05: Share run by family points measured on this class (inside the evidence
    /// envelope).
    pub measured: f64,
    /// 2026-10-05: Share no kernel of the class runs: the report's "novel" rows, and the rows a
    /// placeholder group plans because no rule of the class lowers them (the report classifies
    /// those by the family that implements the op, so its "novel" share alone overstates
    /// coverage).
    pub uncovered: f64,
    /// 2026-10-05: The estimated step, microseconds (the report's).
    pub step_us: f64,
    /// 2026-10-05: Groups by the numerics tag of the rule that formed them, placeholder groups
    /// under `uncovered`.
    pub groups: BTreeMap<&'static str, usize>,
}

/// 2026-10-05: Which tier of the class's tree a residual source sits in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// 2026-10-05: `kernels/<class>/common/`.
    Class,
    /// 2026-10-05: `kernels/<class>/<model>/<quant>/`.
    Model,
}

impl Tier {
    /// 2026-10-05: The report spelling.
    pub fn name(self) -> &'static str {
        match self {
            Tier::Class => "class",
            Tier::Model => "model",
        }
    }
}

/// 2026-10-05: One compiled module of the class's own tree that no family names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidualSource {
    /// 2026-10-05: Module name.
    pub module: String,
    /// 2026-10-05: Repo-relative source.
    pub path: String,
    /// 2026-10-05: Source lines.
    pub lines: usize,
    /// 2026-10-05: Tier.
    pub tier: Tier,
    /// 2026-10-05: The `KERNEL.toml [shadow]` reason when it replaces an inherited file.
    pub shadow: Option<String>,
}

/// 2026-10-05: One `how = "copy"` point of a family on this class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CopyPoint {
    /// 2026-10-05: Family id.
    pub family: String,
    /// 2026-10-05: The point's parameter values, `name=value` joined by commas.
    pub values: String,
    /// 2026-10-05: Its sources.
    pub files: Vec<String>,
}

/// 2026-10-05: A place where the LKB could grow by promotion (book/src/architecture/lkb.md,
/// "Promotion"): a second user of a parameter has appeared.
#[derive(Debug, Clone, PartialEq)]
pub enum Candidate {
    /// 2026-10-05: A family realizes several points as per-point file copies: one template
    /// instantiated at each would delete the copies.
    CopyPoints {
        /// 2026-10-05: Family id.
        family: String,
        /// 2026-10-05: Its copy points.
        count: usize,
    },
    /// 2026-10-05: This model runs a family at a point it lacks (a parameterization
    /// opportunity or a policy variant in the report's gap table).
    Point {
        /// 2026-10-05: `block.node`.
        site: String,
        /// 2026-10-05: Family id.
        family: String,
        /// 2026-10-05: The report class name.
        class: &'static str,
        /// 2026-10-05: Differing parameters, `param other->target`.
        diffs: Vec<String>,
        /// 2026-10-05: Largest share of the step over the report runs.
        share: f64,
    },
}

/// 2026-10-05: The LKB of one model on one class.
#[derive(Debug, Clone, PartialEq)]
pub struct Lkb {
    /// 2026-10-05: Model label.
    pub model: String,
    /// 2026-10-05: Device id.
    pub device: String,
    /// 2026-10-05: The class and its ancestors, own first.
    pub chain: Vec<String>,
    /// 2026-10-05: Families the class's manifest resolves to.
    pub families: usize,
    /// 2026-10-05: Their points by `how`.
    pub points: BTreeMap<&'static str, usize>,
    /// 2026-10-05: Families the report's plans run.
    pub generators_used: BTreeSet<String>,
    /// 2026-10-05: `bit_identical` rules the report's plans apply.
    pub relations_used: BTreeSet<String>,
    /// 2026-10-05: One per report run, in the report's order.
    pub coverage: Vec<Coverage>,
    /// 2026-10-05: The class's own compiled modules no family names.
    pub residual: Vec<ResidualSource>,
    /// 2026-10-05: The class's copy points.
    pub copy_points: Vec<CopyPoint>,
    /// 2026-10-05: Promotion candidates: copy-point families by count, then missing points by
    /// share of the step, largest first.
    pub promotion: Vec<Candidate>,
    /// 2026-10-05: The modelled fusion gain of each report run's plan, in `coverage` order
    /// (filled by [`lkb`]; [`from_report`] leaves it empty).
    pub laxity: Vec<PlanLaxity>,
    /// 2026-10-05: The regenerating command.
    pub command: String,
}

impl Lkb {
    /// 2026-10-05: Lines of the residual sources.
    pub fn residual_lines(&self) -> usize {
        self.residual.iter().map(|s| s.lines).sum()
    }
}

/// 2026-10-05: The LKB of `report`'s model on its device's class, with the shadow reasons the
/// class's `KERNEL.toml` files give.
pub fn lkb(report: &HwReport, repo: &dyn Repo, command: String) -> Result<Lkb, HwError> {
    let resolved = &report.resolved;
    let shadows = shadow_reasons(
        repo,
        &resolved.device.class,
        resolved.sources.target.as_deref(),
    )?;
    let mut out = from_report(report, &shadows, command);
    let gbps = resolved.roofline.roofline.dram_gbps;
    for t in &report.tables {
        out.laxity
            .push(plan_laxity(&report.model.circuit, &t.planned.plan, gbps)?);
    }
    Ok(out)
}

/// 2026-10-05: The LKB of `report`'s model on its device's class; `shadows` maps a residual
/// source's stem to its `[shadow]` reason (empty: no reasons, counts unchanged).
pub fn from_report(report: &HwReport, shadows: &BTreeMap<String, String>, command: String) -> Lkb {
    let resolved = &report.resolved;
    let class = resolved.device.class.clone();
    let families = &resolved.families.families;
    let named: BTreeSet<&str> = families
        .iter()
        .flat_map(|f| f.kernels.iter().map(|k| k.module.as_str()))
        .collect();
    let own = format!("kernels/{class}/");
    let common = format!("{own}common/");
    let residual = resolved
        .sources
        .modules
        .iter()
        .filter(|(name, m)| m.path.starts_with(&own) && !named.contains(name.as_str()))
        .map(|(name, m)| ResidualSource {
            module: name.clone(),
            path: m.path.clone(),
            lines: m.text.lines().count(),
            tier: if m.path.starts_with(&common) {
                Tier::Class
            } else {
                Tier::Model
            },
            shadow: shadows.get(stem(&m.path)).cloned(),
        })
        .collect();
    let mut points = BTreeMap::new();
    let mut copy_points = Vec::new();
    for f in families {
        for p in &f.points {
            *points.entry(p.how.name()).or_insert(0) += 1;
            if p.how == How::Copy {
                copy_points.push(CopyPoint {
                    family: f.id.clone(),
                    values: p
                        .values
                        .iter()
                        .map(|(k, v)| format!("{k}={v}"))
                        .collect::<Vec<_>>()
                        .join(","),
                    files: p.files.clone(),
                });
            }
        }
    }
    let mut generators_used = BTreeSet::new();
    let mut relations_used = BTreeSet::new();
    let mut coverage = Vec::new();
    for t in &report.tables {
        generators_used.extend(
            t.rows
                .iter()
                .filter(|r| !r.placeholder)
                .filter_map(|r| r.family.clone()),
        );
        let mut groups = BTreeMap::new();
        for g in &t.planned.plan.groups {
            let tag = if g.emitter == NOVEL_EMITTER {
                "uncovered"
            } else {
                g.numerics.class()
            };
            *groups.entry(tag).or_insert(0) += 1;
            if matches!(g.numerics, Numerics::BitIdentical { .. }) && g.emitter != NOVEL_EMITTER {
                relations_used.insert(g.rule.clone());
            }
        }
        let uncovered: f64 = t
            .rows
            .iter()
            .filter(|r| r.placeholder || r.class == Class::Novel)
            .map(|r| r.share)
            .sum();
        coverage.push(Coverage {
            run: t.planned.run,
            lkb: 1.0 - uncovered,
            measured: t.share_of(&[Class::Shared]),
            uncovered,
            step_us: t.total_us,
            groups,
        });
    }
    Lkb {
        model: report.model.label.clone(),
        device: resolved.device.id.clone(),
        chain: resolved.chain.iter().map(|c| c.name.clone()).collect(),
        families: families.len(),
        points,
        generators_used,
        relations_used,
        coverage,
        residual,
        promotion: promotion(report, &copy_points),
        copy_points,
        laxity: Vec::new(),
        command,
    }
}

fn promotion(report: &HwReport, copies: &[CopyPoint]) -> Vec<Candidate> {
    let mut by_family: BTreeMap<&str, usize> = BTreeMap::new();
    for c in copies {
        *by_family.entry(c.family.as_str()).or_insert(0) += 1;
    }
    let mut copy: Vec<Candidate> = by_family
        .into_iter()
        .filter(|(_, n)| *n >= 2)
        .map(|(f, n)| Candidate::CopyPoints {
            family: f.to_string(),
            count: n,
        })
        .collect();
    copy.sort_by_key(|c| match c {
        Candidate::CopyPoints { count, .. } => std::cmp::Reverse(*count),
        Candidate::Point { .. } => std::cmp::Reverse(0),
    });
    let mut points: BTreeMap<(String, String), Candidate> = BTreeMap::new();
    for t in &report.tables {
        for r in &t.rows {
            let (Some(family), false) = (&r.family, r.placeholder) else {
                continue;
            };
            if !matches!(
                r.class,
                Class::ParameterizationOpportunity | Class::PolicyVariant
            ) {
                continue;
            }
            let e = points
                .entry((r.site.clone(), family.clone()))
                .or_insert(Candidate::Point {
                    site: r.site.clone(),
                    family: family.clone(),
                    class: r.class.name(),
                    diffs: r.diffs.clone(),
                    share: 0.0,
                });
            if let Candidate::Point { share, .. } = e {
                *share = share.max(r.share);
            }
        }
    }
    let mut points: Vec<Candidate> = points.into_values().collect();
    points.sort_by(|a, b| {
        let s = |c: &Candidate| match c {
            Candidate::Point { share, .. } => *share,
            Candidate::CopyPoints { .. } => 0.0,
        };
        s(b).total_cmp(&s(a))
    });
    copy.extend(points);
    copy
}

fn stem(path: &str) -> &str {
    let file = path.rsplit('/').next().unwrap_or(path);
    file.split('.').next().unwrap_or(file)
}

/// 2026-10-05: `[shadow]` stem to reason, from the class's common `KERNEL.toml` and its model
/// target's; a missing file declares none.
fn shadow_reasons(
    repo: &dyn Repo,
    class: &str,
    target: Option<&str>,
) -> Result<BTreeMap<String, String>, HwError> {
    let mut rels = vec![format!("kernels/{class}/common/KERNEL.toml")];
    rels.extend(target.map(|t| format!("kernels/{t}/KERNEL.toml")));
    let mut out = BTreeMap::new();
    for rel in rels {
        let Ok(text) = repo.read(&rel) else {
            continue;
        };
        let doc: toml::Table =
            toml::from_str(&text).map_err(|e| HwError::Class(format!("{rel}: {e}")))?;
        if let Some(toml::Value::Table(t)) = doc.get("shadow") {
            for (k, v) in t {
                let reason = v.as_str().ok_or_else(|| {
                    HwError::Class(format!("{rel} [shadow] `{k}`: the reason must be a string"))
                })?;
                out.insert(k.clone(), reason.to_string());
            }
        }
    }
    Ok(out)
}
