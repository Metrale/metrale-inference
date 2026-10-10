// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: FUSIONS row ranges derived from SCHEDULES: for every FUSIONS.toml rule whose
//! kernels a schedule names, (a) the swept rows inside the rule's `rows` where a default build
//! would launch something else (a bit-identical faster sibling won, or the routed default there
//! is another kernel), and (b) the `rows` range the sweep proves for the rule, when every swept
//! shape the rule routes agrees.
//!
//! Owner: metrale-accuracy (envelope).
//! Invariants:
//! - Pure: FUSIONS text and parsed schedules in, a report out; nothing is rewritten here.
//! - A rule's shapes are the schedules whose `default` is one of its kernels (and whose weight
//!   format its pattern names, when it names one): the sites it routes today.
//! - A proposal only covers rows where every shape's default build launches the rule's kernel
//!   (`same`, or a `bit_identical` winner, or today's default under an opt-in winner), so it
//!   never changes an output byte; shapes that disagree at a common row get no proposal.
//! - `differs` rules are opt-in levers, outside the default build: not checked.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use metrale_circuit::{Numerics as RuleNumerics, Rule, parse_rule_set};

use super::schedules::{Schedule, Schedules, Shape};

/// 2026-10-10: Why a swept range inside a rule's rows disagrees with it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Why {
    /// 2026-10-10: The default is the rule's kernel, but a bit-identical faster sibling won and
    /// is enabled by default.
    FasterSibling(String),
    /// 2026-10-10: A default build launches this other entry point here (`""` for a `new`
    /// cell, which no served plan routes).
    NotRouted(String),
}

/// 2026-10-10: One disagreement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Disagreement {
    /// 2026-10-10: The shape.
    pub shape: Shape,
    /// 2026-10-10: The schedule's rows inside the rule's rows.
    pub rows: [u64; 2],
    /// 2026-10-10: Why.
    pub why: Why,
}

/// 2026-10-10: The range the sweep proves for a rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Proposal {
    /// 2026-10-10: Every shape agrees on this one contiguous range; `changes` when it is not
    /// the rule's `rows` today.
    Range {
        /// 2026-10-10: `[lo, hi]`.
        rows: [u64; 2],
        /// 2026-10-10: Differs from the rule's rows.
        changes: bool,
    },
    /// 2026-10-10: No single range is proven, and why.
    None(String),
}

/// 2026-10-10: One rule's check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleCheck {
    /// 2026-10-10: Rule id.
    pub rule: String,
    /// 2026-10-10: Its kernels (`module::function`).
    pub kernels: Vec<String>,
    /// 2026-10-10: Its rows today.
    pub rows: [u64; 2],
    /// 2026-10-10: The swept shapes it routes.
    pub shapes: Vec<Shape>,
    /// 2026-10-10: Disagreements inside its rows.
    pub disagreements: Vec<Disagreement>,
    /// 2026-10-10: The proven range.
    pub proposal: Proposal,
}

/// 2026-10-10: The report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionsReport {
    /// 2026-10-10: Hardware class.
    pub hardware: String,
    /// 2026-10-10: Checked rules, in FUSIONS file order.
    pub rules: Vec<RuleCheck>,
}

/// 2026-10-10: FUSIONS.toml did not load.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("FUSIONS.toml: {0}")]
pub struct FusionsError(pub String);

/// 2026-10-10: Inclusive intervals, sorted, disjoint and not touching.
type Spans = Vec<(u64, u64)>;

fn normalize(mut v: Spans) -> Spans {
    v.sort();
    let mut out: Spans = Vec::with_capacity(v.len());
    for (lo, hi) in v {
        match out.last_mut() {
            Some(last) if lo <= last.1.saturating_add(1) => last.1 = last.1.max(hi),
            _ => out.push((lo, hi)),
        }
    }
    out
}

fn intersect(a: &Spans, b: &Spans) -> Spans {
    let mut out = Vec::new();
    for &(alo, ahi) in a {
        for &(blo, bhi) in b {
            let (lo, hi) = (alo.max(blo), ahi.min(bhi));
            if lo <= hi {
                out.push((lo, hi));
            }
        }
    }
    normalize(out)
}

fn check_rule(rule: &Rule, s: &Schedules) -> Option<RuleCheck> {
    let kernels: BTreeSet<String> = rule.kernels.iter().map(|k| k.to_string()).collect();
    if kernels.is_empty() || matches!(rule.numerics, RuleNumerics::Differs { .. }) {
        return None;
    }
    let names = |e: &Schedule| kernels.contains(&e.kernel) || kernels.contains(&e.default);
    if !s.schedule.iter().any(names) {
        return None;
    }
    let weights: BTreeSet<String> = rule
        .pattern
        .iter()
        .filter_map(|p| p.weight.map(|w| w.name()))
        .collect();
    let routes = |e: &Schedule| {
        kernels.contains(&e.default) && (weights.is_empty() || weights.contains(&e.weight))
    };
    let shapes: Vec<Shape> = s
        .schedule
        .iter()
        .filter(|e| routes(e))
        .map(Schedule::shape)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let (rlo, rhi) = rule.rows;
    let mut disagreements = Vec::new();
    let mut agree: Vec<Spans> = Vec::new();
    let mut other: Vec<Spans> = Vec::new();
    for shape in &shapes {
        let mut entries: Vec<&Schedule> =
            s.schedule.iter().filter(|e| e.shape() == *shape).collect();
        entries.sort_by_key(|e| e.rows);
        let (mut yes, mut no) = (Vec::new(), Vec::new());
        for e in entries {
            let launches = kernels.contains(e.routed());
            let [lo, hi] = e.rows;
            if launches { &mut yes } else { &mut no }.push((lo, hi));
            if launches || hi < rlo || lo > rhi {
                continue;
            }
            let why = if kernels.contains(&e.default) {
                Why::FasterSibling(e.kernel.clone())
            } else {
                Why::NotRouted(e.routed().to_string())
            };
            disagreements.push(Disagreement {
                shape: shape.clone(),
                rows: [lo.max(rlo), hi.min(rhi)],
                why,
            });
        }
        agree.push(normalize(yes));
        other.push(normalize(no));
    }
    let proposal = propose(&agree, &other, rule.rows);
    Some(RuleCheck {
        rule: rule.id.clone(),
        kernels: kernels.into_iter().collect(),
        rows: [rlo, rhi],
        shapes,
        disagreements,
        proposal,
    })
}

fn propose(agree: &[Spans], other: &[Spans], rows: (u64, u64)) -> Proposal {
    if agree.is_empty() {
        return Proposal::None("no swept shape routes the rule's kernels today".into());
    }
    for (i, a) in agree.iter().enumerate() {
        for (j, o) in other.iter().enumerate() {
            if let Some(&(lo, hi)) = intersect(a, o).first().filter(|_| i != j) {
                return Proposal::None(format!(
                    "the shapes disagree at rows [{lo}, {hi}] (one launches the rule's kernel, another does not)"
                ));
            }
        }
    }
    let common = agree[1..]
        .iter()
        .fold(agree[0].clone(), |g, a| intersect(&g, a));
    match common.as_slice() {
        [] => Proposal::None("no row where every shape launches the rule's kernel".into()),
        [(lo, hi)] => Proposal::Range {
            rows: [*lo, *hi],
            changes: (*lo, *hi) != rows,
        },
        _ => Proposal::None(format!(
            "the agreeing rows are not one range: {}",
            common
                .iter()
                .map(|(lo, hi)| format!("[{lo}, {hi}]"))
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

/// 2026-10-10: Check every rule of `fusions` against `s`.
pub fn check_fusions(fusions: &str, s: &Schedules) -> Result<FusionsReport, FusionsError> {
    let set = parse_rule_set(fusions).map_err(|e| FusionsError(e.to_string()))?;
    Ok(FusionsReport {
        hardware: s.hardware.clone(),
        rules: set.rules.iter().filter_map(|r| check_rule(r, s)).collect(),
    })
}

impl FusionsReport {
    /// 2026-10-10: The report as Markdown: a summary table, then each rule's disagreements.
    pub fn render(&self) -> String {
        let mut o = String::new();
        let _ = writeln!(
            o,
            "# FUSIONS row ranges against SCHEDULES ({})\n",
            self.hardware
        );
        let dis: usize = self.rules.iter().map(|r| r.disagreements.len()).sum();
        let changes = self
            .rules
            .iter()
            .filter(|r| matches!(r.proposal, Proposal::Range { changes: true, .. }))
            .count();
        let _ = writeln!(
            o,
            "Rules checked: {}. Disagreements: {dis}. Proposed row changes: {changes}.\n",
            self.rules.len()
        );
        o.push_str("| rule | kernels | rows | shapes | disagreements | proposal |\n");
        o.push_str("|---|---|---|---|---|---|\n");
        for r in &self.rules {
            let proposal = match &r.proposal {
                Proposal::Range {
                    rows,
                    changes: true,
                } => format!("[{}, {}] (change)", rows[0], rows[1]),
                Proposal::Range {
                    rows,
                    changes: false,
                } => format!("[{}, {}] (unchanged)", rows[0], rows[1]),
                Proposal::None(why) => format!("none: {why}"),
            };
            let _ = writeln!(
                o,
                "| {} | {} | [{}, {}] | {} | {} | {} |",
                r.rule,
                r.kernels.join(", "),
                r.rows[0],
                r.rows[1],
                r.shapes.len(),
                r.disagreements.len(),
                proposal
            );
        }
        for r in self.rules.iter().filter(|r| !r.disagreements.is_empty()) {
            let _ = writeln!(o, "\n## {}\n", r.rule);
            for d in &r.disagreements {
                let why = match &d.why {
                    Why::FasterSibling(k) => {
                        format!("bit-identical faster sibling `{k}` is the default choice")
                    }
                    Why::NotRouted(k) if k.is_empty() => {
                        "no served plan routes it (`new`)".to_string()
                    }
                    Why::NotRouted(k) => format!("a default build launches `{k}`"),
                };
                let _ = writeln!(
                    o,
                    "- {} rows [{}, {}]: {why}",
                    d.shape, d.rows[0], d.rows[1]
                );
            }
        }
        o
    }
}

#[cfg(test)]
#[path = "fusions_tests.rs"]
mod tests;
