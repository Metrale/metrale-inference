// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Which registered models a set of changed files reaches through the kernels they
//! run (`met circuit affected`): file → kernel family (KERNEL_FAMILIES.toml `files`, `discover`
//! globs and macro files) → instance (INSTANCES.toml, through its fused plans) → model.
//!
//! Owner: metrale-circuit (venn).
//! Invariants:
//! - Pure: no I/O. The caller passes the changed paths, the parsed manifests and each
//!   instance's family usage ([`plan_usage`], [`op_usage`]).
//! - Derived, never hand-kept: the only inputs are the family manifest and the instances' plans,
//!   so a family's users change exactly when its plans or its `files` do.
//! - Fail-closed: a path under `kernels/` that no family lists is `unmapped` (every model);
//!   an instance whose usage is unknown (a plan group no family lists, a node no family
//!   implements) uses every family on its hardware. A path outside `kernels/` is `non_kernel`:
//!   this map says nothing about it, and its caller treats it as reaching every model.
//! - Deterministic: models, reasons and path lists are sorted.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use super::classify::implements;
use super::families::{Discover, Families, Family};
use crate::fuser::FusionPlan;
use crate::ir::Circuit;
use crate::precision::glob;

/// 2026-10-15: The families one instance runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Usage {
    /// 2026-10-15: Exactly these family ids.
    Known(BTreeSet<String>),
    /// 2026-10-15: Not derivable; the text says why. It uses every family on its hardware.
    Unknown(String),
}

impl Usage {
    fn runs(&self, family: &str) -> bool {
        match self {
            Usage::Known(set) => set.contains(family),
            Usage::Unknown(_) => true,
        }
    }

    /// 2026-10-15: The union of two usages; unknown absorbs.
    pub fn union(self, other: Usage) -> Usage {
        match (self, other) {
            (Usage::Unknown(why), _) | (_, Usage::Unknown(why)) => Usage::Unknown(why),
            (Usage::Known(mut a), Usage::Known(b)) => {
                a.extend(b);
                Usage::Known(a)
            }
        }
    }
}

/// 2026-10-15: One instance and the families it runs.
#[derive(Debug, Clone)]
pub struct InstanceUse {
    /// 2026-10-15: Recipe id.
    pub recipe: String,
    /// 2026-10-15: Checkpoint id.
    pub checkpoint: String,
    /// 2026-10-15: Kernel target, `hw/model/quant`.
    pub target: String,
    /// 2026-10-15: Its families.
    pub usage: Usage,
}

impl InstanceUse {
    fn hw(&self) -> &str {
        self.target.split('/').next().unwrap_or_default()
    }

    fn model(&self) -> &str {
        self.target.split('/').nth(1).unwrap_or_default()
    }
}

/// 2026-10-15: One link of the chain that puts a model in the affected set.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct Hop {
    /// 2026-10-15: The changed file.
    pub file: String,
    /// 2026-10-15: The family that lists it.
    pub family: String,
    /// 2026-10-15: The instance (recipe) that runs the family.
    pub instance: String,
    /// 2026-10-15: True when the instance's usage is unknown and it is counted fail-closed.
    pub assumed: bool,
}

/// 2026-10-15: One affected model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AffectedModel {
    /// 2026-10-15: Model directory (`kernels/<hw>/<model>/`).
    pub model: String,
    /// 2026-10-15: Hardware directory.
    pub hw: String,
    /// 2026-10-15: Checkpoint id.
    pub checkpoint: String,
    /// 2026-10-15: Why, one hop per (file, family, instance).
    pub reason: Vec<Hop>,
}

/// 2026-10-15: What a diff reaches.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Affected {
    /// 2026-10-15: Models reached through a family, by (hw, model, checkpoint).
    pub models: Vec<AffectedModel>,
    /// 2026-10-15: Paths under `kernels/` no family lists: they reach every model.
    pub unmapped: Vec<String>,
    /// 2026-10-15: Paths a family lists that no instance runs.
    pub unused: Vec<String>,
    /// 2026-10-15: Paths outside `kernels/`; this map does not scope them.
    pub non_kernel: Vec<String>,
    /// 2026-10-15: Instances whose usage is unknown (recipe, why): each counts as running
    /// every family on its hardware.
    pub assumed: BTreeMap<String, String>,
}

/// 2026-10-15: Whether `family` names `path`: a point's `files`, a `file` discovery glob, or
/// a `macro` discovery file.
pub fn lists(family: &Family, path: &str) -> bool {
    family
        .points
        .iter()
        .any(|p| p.files.iter().any(|f| f == path))
        || family.discover.iter().any(|d| match d {
            Discover::File { glob: g, .. } => glob(g, path),
            Discover::Macro { file, .. } => file == path,
        })
}

/// 2026-10-15: The models `changed` reaches. `manifests` holds one parsed
/// KERNEL_FAMILIES.toml per hardware that has one; a path on a hardware without one is
/// unmapped.
pub fn affected(changed: &[String], manifests: &[Families], uses: &[InstanceUse]) -> Affected {
    let mut out = Affected::default();
    for u in uses {
        if let Usage::Unknown(why) = &u.usage {
            out.assumed.insert(u.recipe.clone(), why.clone());
        }
    }
    let mut hops: BTreeMap<(String, String, String), BTreeSet<Hop>> = BTreeMap::new();
    for path in changed {
        let Some(rest) = path.strip_prefix("kernels/") else {
            out.non_kernel.push(path.clone());
            continue;
        };
        let hw = rest.split('/').next().unwrap_or_default();
        let families: Vec<&Family> = manifests
            .iter()
            .filter(|m| m.hardware == hw)
            .flat_map(|m| m.families.iter())
            .filter(|f| lists(f, path))
            .collect();
        if families.is_empty() {
            out.unmapped.push(path.clone());
            continue;
        }
        let mut used = false;
        for f in &families {
            for u in uses.iter().filter(|u| u.hw() == hw && u.usage.runs(&f.id)) {
                used = true;
                hops.entry((u.hw().into(), u.model().into(), u.checkpoint.clone()))
                    .or_default()
                    .insert(Hop {
                        file: path.clone(),
                        family: f.id.clone(),
                        instance: u.recipe.clone(),
                        assumed: matches!(u.usage, Usage::Unknown(_)),
                    });
            }
        }
        if !used {
            out.unused.push(path.clone());
        }
    }
    out.models = hops
        .into_iter()
        .map(|((hw, model, checkpoint), reason)| AffectedModel {
            model,
            hw,
            checkpoint,
            reason: reason.into_iter().collect(),
        })
        .collect();
    for list in [&mut out.unmapped, &mut out.unused, &mut out.non_kernel] {
        list.sort();
        list.dedup();
    }
    out
}

/// 2026-10-15: The families a fused plan launches: each group's family by its kernels or, for a
/// kernel-less group, its emitter. A group no family lists makes the usage unknown.
pub fn plan_usage(circuit: &Circuit, plan: &FusionPlan, fams: &Families) -> Usage {
    let mut set = BTreeSet::new();
    for g in &plan.groups {
        for &n in &g.nodes {
            let node = &circuit.nodes[n];
            match fams.of_group(&g.kernels, &g.emitter, &node.op) {
                Some(f) => {
                    set.insert(f.id.clone());
                }
                None => {
                    return Usage::Unknown(format!(
                        "node `{}` ({}) runs {}, which no family lists",
                        node.id,
                        node.op.name(),
                        if g.kernels.is_empty() {
                            format!("the `{}` emitter", g.emitter)
                        } else {
                            g.kernels
                                .iter()
                                .map(|k| k.to_string())
                                .collect::<Vec<_>>()
                                .join(" + ")
                        }
                    ));
                }
            }
        }
    }
    Usage::Known(set)
}

/// 2026-10-15: For an instance without checked-in plans: every family that implements one of
/// its nodes at any row count (an over-approximation). A node no family implements makes the
/// usage unknown.
pub fn op_usage(circuit: &Circuit, fams: &Families) -> Usage {
    let mut set = BTreeSet::new();
    for node in &circuit.nodes {
        let found: Vec<&Family> = fams
            .families
            .iter()
            .filter(|f| implements(f, circuit, node))
            .collect();
        if found.is_empty() {
            return Usage::Unknown(format!(
                "node `{}` ({}) is implemented by no family",
                node.id,
                node.op.name()
            ));
        }
        set.extend(found.into_iter().map(|f| f.id.clone()));
    }
    Usage::Known(set)
}

#[cfg(test)]
#[path = "affected_tests.rs"]
mod affected_tests;
