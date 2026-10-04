// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The check a plan must pass: for every node of every group, the pipeline the
//! group's kernel family declares for it ([`super::declare`]) equals the pipeline the plan
//! requires of it ([`super::require`]), inputs, every step and outputs alike. A fused group
//! composes its nodes' pipelines: each member's outputs are handed to the next member in the
//! format the kernel declares, which must be the format the plan stores or the rule states it
//! holds, so a fused edge cannot change format silently.
//!
//! Owner: metrale-circuit (pipeline).
//! Invariants:
//! - Every node of the plan's section gets a required pipeline or the check fails.
//! - A group whose kernels no family lists, or whose family's declaration does not resolve for a
//!   node, fails like a differing one ([`PipelineError::Undeclared`], [`PipelineError::Declared`]
//!   inside a [`PipelineError::Mismatch`]); only a hardware planner's placeholder group (no
//!   kernel of the class covers the op) is exempt, since nothing runs it.
//! - The family of a node is the one, among those listing a kernel of its group and
//!   implementing its op (with its role and weight format), that names the role explicitly,
//!   else the first in manifest order.

use std::collections::BTreeMap;

use super::require::{Need, required};
use super::{Mismatch, NodePipeline, PipelineError, Stated, differences};
use crate::fuser::{FusionPlan, Group};
use crate::hardware::plan::NOVEL_EMITTER;
use crate::ir::{Circuit, Node, NodeIdx, OpKind};
use crate::rules::Rule;
use crate::venn::Subject;
use crate::venn::classify::point_of;
use crate::venn::families::{Families, Family};

/// 2026-10-02: The required pipeline of every node a plan runs, and the steps its rules state.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlanPipelines {
    /// 2026-10-02: By node index; `None` outside the plan's section.
    pub nodes: Vec<Option<NodePipeline>>,
    /// 2026-10-02: What a rule states, by node.
    pub stated: BTreeMap<NodeIdx, Stated>,
}

impl PlanPipelines {
    /// 2026-10-02: Node `n`'s pipeline as a plan line shows it: `in -> [...] -> out`, a step a
    /// rule states marked ` (rule)`.
    pub fn line(&self, n: NodeIdx) -> Option<String> {
        let p = self.nodes.get(n)?.as_ref()?;
        Some(p.render(self.stated.get(&n).unwrap_or(&Stated::default())))
    }

    /// 2026-10-02: Node `n`'s pipeline as facts ([`NodePipeline::facts`]).
    pub fn facts(&self, n: NodeIdx) -> Option<Vec<String>> {
        let p = self.nodes.get(n)?.as_ref()?;
        Some(p.facts(self.stated.get(&n).unwrap_or(&Stated::default())))
    }
}

/// 2026-10-02: The family that declares node `node`'s pipeline in group `g`.
pub fn family_of<'f>(families: &'f Families, g: &Group, node: &Node) -> Option<&'f Family> {
    let listed = |f: &Family| {
        if g.kernels.is_empty() {
            f.emitters.contains(&g.emitter)
        } else {
            g.kernels.iter().any(|k| f.kernels.contains(k))
        }
    };
    let rank = |f: &Family| -> Option<u8> {
        f.ops
            .iter()
            .filter(|s| s.names(&node.op))
            .filter(|s| s.weight.is_empty() || node.weight.is_some_and(|w| s.weight.contains(&w)))
            .filter_map(|s| match node.op {
                OpKind::Linear(r) if s.roles.contains(&r) => Some(0),
                OpKind::Linear(_) if s.roles.is_empty() => Some(1),
                OpKind::Linear(_) => None,
                _ => Some(1),
            })
            .min()
    };
    families
        .families
        .iter()
        .filter(|f| listed(f))
        .filter_map(|f| rank(f).map(|r| (r, f)))
        .min_by_key(|(r, _)| *r)
        .map(|(_, f)| f)
}

/// 2026-10-02: Check `plan` of `c`: every node's declared pipeline against its requirement
/// under `settings`, the plan's groups' rules being among `rules`.
pub fn check_plan(
    c: &Circuit,
    plan: &FusionPlan,
    rules: &[Rule],
    families: &Families,
    settings: &BTreeMap<String, String>,
) -> Result<PlanPipelines, PipelineError> {
    let need = Need::of_plan(c, plan, rules, settings);
    let subject = Subject {
        recipe: "",
        circuit: c,
        settings,
        plan: None,
    };
    let mut out = PlanPipelines {
        nodes: vec![None; c.nodes.len()],
        stated: BTreeMap::new(),
    };
    let mut mismatches = Vec::new();
    for g in &plan.groups {
        for &n in &g.nodes {
            let node = &c.nodes[n];
            let req = required(&need, n)?;
            let stated = Stated {
                steps: need.stated_steps(n).collect(),
                inputs: node
                    .inputs
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| need.held_input(n, **e))
                    .map(|(i, _)| i)
                    .collect(),
                outputs: node
                    .outputs
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| need.holds_in_group(**e))
                    .map(|(i, _)| i)
                    .collect(),
            };
            if stated != Stated::default() {
                out.stated.insert(n, stated);
            }
            if g.emitter != NOVEL_EMITTER {
                // 2026-10-02: A kernel whose pipeline is undeclared or unresolvable cannot be
                // shown to run the requirement: it is a mismatch of its rule, like a differing one.
                let found = match declared(families, g, n, &subject) {
                    Ok((_, decl)) if decl == req => None,
                    Ok((fam, decl)) => Some((fam.to_string(), differences(&req, &decl))),
                    Err(PipelineError::Declared { family, detail, .. }) => {
                        Some((family, vec![detail]))
                    }
                    Err(PipelineError::Undeclared { kernels, .. }) => Some((
                        "-".to_string(),
                        vec![format!(
                            "no kernel family declares a pipeline for {kernels}"
                        )],
                    )),
                    Err(e) => Some(("-".to_string(), vec![e.to_string()])),
                };
                if let Some((family, diffs)) = found {
                    mismatches.push(Mismatch {
                        node: node.id.clone(),
                        rule: g.rule.clone(),
                        family,
                        diffs,
                    });
                }
            }
            out.nodes[n] = Some(req);
        }
    }
    if mismatches.is_empty() {
        Ok(out)
    } else {
        Err(PipelineError::Mismatch(mismatches))
    }
}

fn declared<'f>(
    families: &'f Families,
    g: &Group,
    n: NodeIdx,
    subject: &Subject<'_>,
) -> Result<(&'f str, NodePipeline), PipelineError> {
    let node = &subject.circuit.nodes[n];
    let kernels = || {
        if g.kernels.is_empty() {
            format!("({} emitter)", g.emitter)
        } else {
            g.kernels
                .iter()
                .map(|k| k.to_string())
                .collect::<Vec<_>>()
                .join(" + ")
        }
    };
    let fam = family_of(families, g, node).ok_or_else(|| PipelineError::Undeclared {
        node: node.id.clone(),
        op: node.op.name(),
        kernels: kernels(),
    })?;
    let fail = |detail: String| PipelineError::Declared {
        node: node.id.clone(),
        op: node.op.name(),
        family: fam.id.clone(),
        detail,
    };
    let (point, _) = point_of(fam, subject, node).map_err(|e| fail(e.to_string()))?;
    let p = super::declare::declared_for(
        &fam.pipeline,
        &fam.point_pipelines(),
        &g.kernels,
        &super::declare::Site::of(subject.circuit, n),
        &point,
    )
    .map_err(fail)?;
    Ok((fam.id.as_str(), p))
}

#[cfg(test)]
#[path = "check_tests.rs"]
mod check_tests;
