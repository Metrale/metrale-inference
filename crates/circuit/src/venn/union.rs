// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The union Venn: every instance's Venn report (each model as the target, against
//! the golden instances) folded into one view of the whole model set: which kernel families
//! each model uses and how (shared, policy variant, parameterization opportunity, novel), the
//! op vocabulary and how fast it saturates as models are added, the shape union per family with
//! what is measured, each model's in-envelope step share, and the deduplicated parameterization
//! plan and novel-kernel list. The input to the per-family envelope sweeps.
//!
//! Owner: metrale-circuit (venn).
//! Invariants:
//! - Pure: built from the reports, the circuits and the instances it is given; no I/O.
//! - One classifier: every class and share is the Venn report's own; this module only folds.
//! - "In envelope" at a run means the Venn class is Shared (a microbench record at the family
//!   point and row count) AND the record covers the node's shape: its regime names the node's
//!   `N` and `K` (` · ... N=<out> K=<k>`), or it names no shape and was taken on this model's
//!   own kernel target (the record then measured this model's shapes). A Shared node whose
//!   records are at other shapes is "point measured" only.

use std::collections::{BTreeMap, BTreeSet};

use super::families::{EvidenceSource, ParamKind, Values};
use super::report::{VennReport, site};
use super::{Class, Diff, Run};
use crate::instances::Instance;
use crate::ir::Circuit;

/// 2026-10-10: One model of the union: its instance, circuit and Venn report.
#[derive(Debug, Clone, Copy)]
pub struct UnionInput<'a> {
    /// 2026-10-10: The instance.
    pub instance: &'a Instance,
    /// 2026-10-10: Its instantiated circuit.
    pub circuit: &'a Circuit,
    /// 2026-10-10: Its Venn report against the golden instances.
    pub report: &'a VennReport,
}

/// 2026-10-10: How far a site is inside the measured envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Envelope {
    /// 2026-10-10: Shared, with a record at the node's shape (or on this model's own target).
    Shape,
    /// 2026-10-10: Shared at the family point and row count, records at other shapes only.
    Point,
    /// 2026-10-10: Not Shared.
    Outside,
}

/// 2026-10-10: One site of one model at one run.
#[derive(Debug, Clone)]
pub struct SiteUse {
    /// 2026-10-10: Index into [`UnionReport::models`].
    pub model: usize,
    /// 2026-10-10: Mode and rows.
    pub run: Run,
    /// 2026-10-10: `block.node`.
    pub site: String,
    /// 2026-10-10: Op name (`linear:q`, `rms_norm`, ...).
    pub op: String,
    /// 2026-10-10: Weight and first-input formats, for weight-reading ops.
    pub weight: Option<String>,
    /// 2026-10-10: First-input format.
    pub activation: Option<String>,
    /// 2026-10-10: Estimated time over the site's nodes, microseconds.
    pub time_us: f64,
    /// 2026-10-10: Share of the model's estimated step at this run.
    pub share: f64,
    /// 2026-10-10: The primary class; Novel when nothing implements it.
    pub class: Class,
    /// 2026-10-10: The primary family.
    pub family: Option<String>,
    /// 2026-10-10: The target's point in that family.
    pub point: Values,
    /// 2026-10-10: Compile-time and policy parameters that differ from the comparison.
    pub diffs: Vec<Diff>,
    /// 2026-10-10: `(out, k)` of the weight a weight-reading site reads.
    pub shape: Option<(u64, u64)>,
    /// 2026-10-10: Envelope status.
    pub envelope: Envelope,
}

/// 2026-10-10: One model's line.
#[derive(Debug, Clone)]
pub struct ModelSummary {
    /// 2026-10-10: Recipe.
    pub recipe: String,
    /// 2026-10-10: Checkpoint.
    pub checkpoint: String,
    /// 2026-10-10: Circuit arch.
    pub arch: String,
    /// 2026-10-10: Golden (rules cover it).
    pub golden: bool,
    /// 2026-10-10: Layer count per layer kind.
    pub layers: BTreeMap<String, usize>,
    /// 2026-10-10: Estimated step per run, microseconds.
    pub step_us: BTreeMap<Run, f64>,
}

/// 2026-10-10: The folded union.
#[derive(Debug, Clone)]
pub struct UnionReport {
    /// 2026-10-10: Models, in instance order (the order of addition).
    pub models: Vec<ModelSummary>,
    /// 2026-10-10: Every site of every model at every run.
    pub uses: Vec<SiteUse>,
    /// 2026-10-10: The runs, in report order.
    pub runs: Vec<Run>,
    /// 2026-10-10: The compared (golden) recipes.
    pub against: Vec<String>,
}

/// 2026-10-10: Fold the reports. Every report must cover the same runs.
pub fn build_union(inputs: &[UnionInput<'_>]) -> Result<UnionReport, String> {
    let first = inputs.first().ok_or("the union needs at least one model")?;
    let runs: Vec<Run> = first.report.tables.iter().map(|t| t.run).collect();
    let against: BTreeSet<String> = inputs
        .iter()
        .flat_map(|i| i.report.against.iter().map(|a| a.recipe.clone()))
        .collect();
    let mut models = Vec::with_capacity(inputs.len());
    let mut uses = Vec::new();
    for (m, inp) in inputs.iter().enumerate() {
        let got: Vec<Run> = inp.report.tables.iter().map(|t| t.run).collect();
        if got != runs {
            return Err(format!(
                "{}: report runs {got:?}, the union's are {runs:?}",
                inp.instance.recipe
            ));
        }
        let mut layers = BTreeMap::new();
        for k in &inp.circuit.layer_kinds {
            *layers.entry(k.name().to_string()).or_insert(0) += 1;
        }
        let target_dir = inp.instance.target.split('/').nth(1).unwrap_or_default();
        let mut step_us = BTreeMap::new();
        for t in &inp.report.tables {
            step_us.insert(t.run, t.total_us);
            for r in &t.rows {
                let shape = site_shape(inp.circuit, &r.site, &r.op);
                let (class, family, point, diffs, envelope) = match &r.primary {
                    None => (
                        Class::Novel,
                        None,
                        Values::new(),
                        Vec::new(),
                        Envelope::Outside,
                    ),
                    Some(f) => {
                        let envelope = if f.class != Class::Shared {
                            Envelope::Outside
                        } else if f.evidence.iter().any(|e| covers(e, shape, target_dir)) {
                            Envelope::Shape
                        } else {
                            Envelope::Point
                        };
                        let diffs = f
                            .diffs
                            .iter()
                            .filter(|d| d.kind != ParamKind::Runtime)
                            .cloned()
                            .collect();
                        (
                            f.class,
                            Some(f.family.clone()),
                            f.point.clone(),
                            diffs,
                            envelope,
                        )
                    }
                };
                uses.push(SiteUse {
                    model: m,
                    run: t.run,
                    site: r.site.clone(),
                    op: r.op.clone(),
                    weight: r.weight.clone(),
                    activation: r.activation.clone(),
                    time_us: r.time_us,
                    share: r.share,
                    class,
                    family,
                    point,
                    diffs,
                    shape,
                    envelope,
                });
            }
        }
        models.push(ModelSummary {
            recipe: inp.instance.recipe.clone(),
            checkpoint: inp.instance.checkpoint.clone(),
            arch: inp.instance.arch.clone(),
            golden: inp.instance.golden,
            layers,
            step_us,
        });
    }
    Ok(UnionReport {
        models,
        uses,
        runs,
        against: against.into_iter().collect(),
    })
}

/// 2026-10-10: The weight shape of the first node of `site` running `op`.
fn site_shape(c: &Circuit, site_id: &str, op: &str) -> Option<(u64, u64)> {
    let n = c
        .nodes
        .iter()
        .find(|n| site(n) == site_id && n.op.name() == op)?;
    if !n.op.reads_linear_weight() {
        return None;
    }
    c.weight_shape(n)
}

/// 2026-10-10: Whether evidence record `e` covers a node of `shape` on the model whose kernel
/// target directory is `target_dir`.
fn covers(e: &EvidenceSource, shape: Option<(u64, u64)>, target_dir: &str) -> bool {
    let EvidenceSource::Measurement(key) = e else {
        return false;
    };
    let (kernel, regime) = key.split_once(" @ ").unwrap_or((key.as_str(), ""));
    match regime.split_once(" · ") {
        Some((_, suffix)) => match shape {
            Some((out, k)) => dim_of(suffix, "N") == Some(out) && dim_of(suffix, "K") == Some(k),
            None => false,
        },
        None => kernel.split("::").next() == Some(target_dir),
    }
}

/// 2026-10-10: `<name>=<value>` in a measurement's shape suffix.
fn dim_of(suffix: &str, name: &str) -> Option<u64> {
    suffix.split_whitespace().find_map(|w| {
        let (k, v) = w.split_once('=')?;
        (k == name).then(|| v.trim_end_matches(',').parse().ok())?
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shape_suffix_must_name_the_nodes_n_and_k() {
        let e = |k: &str| EvidenceSource::Measurement(k.into());
        let at = e("qwen3.8-27b::w4a16_gemv_sw @ decode C=1 (R=4) · gate/up M=1 N=17408 K=5120");
        assert!(covers(&at, Some((17408, 5120)), "gemma-4-31b"));
        assert!(!covers(&at, Some((17408, 4096)), "gemma-4-31b"));
        assert!(!covers(&at, None, "qwen3.8-27b"));
        // 2026-10-10: A record without a shape measured its own model's shapes only.
        let own = e("qwen3.6-35b-a3b::moe_expert_down_act_fp8_grouped @ decode C=1 (R=2, MTP k=1)");
        assert!(covers(&own, Some((2048, 512)), "qwen3.6-35b-a3b"));
        assert!(!covers(&own, Some((2048, 512)), "gemma-4-26b-a4b"));
        assert!(!covers(
            &EvidenceSource::Microbench("x".into()),
            None,
            "qwen3.6-35b-a3b"
        ));
    }
}
