// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The sweep: every (family, point, shape) a described model runs. Each instance in
//! `kernels/circuits/INSTANCES.toml` on the hardware is loaded, fused at every planned mode and
//! row count, and each fused node is placed in its family by the Venn's own
//! [`metrale_circuit::venn::classify::usages`], so the accuracy sweep and `met circuit venn`
//! agree on which kernel runs where. A newly described model is swept with no new test code.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - Business logic only: every byte arrives through [`Repo`] (SBIO).
//! - Deterministic order: points sort by family, then point, then shape.
//! - A node whose kernel belongs to no family is an error (the Venn's `UnmappedKernel`), never
//!   a silently smaller sweep.

use std::collections::{BTreeMap, BTreeSet};

use metrale_circuit::fuser::{AvailableKernels, fuse, section_of};
use metrale_circuit::instances::{Instance, parse_instances};
use metrale_circuit::rules::Mode;
use metrale_circuit::venn::classify::{Usage, candidates, point_of, usages};
use metrale_circuit::venn::families::{Families, Values};
use metrale_circuit::venn::{Repo, Subject, load_instance, parse_families};

/// 2026-10-09: What sizes one launch: the op, its formats, its widths and its rows.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Shape {
    /// 2026-10-09: The node's op (`linear:q`, `rms_norm`, ...).
    pub op: String,
    /// 2026-10-09: Weight format, when the node has a weight.
    pub weight: Option<String>,
    /// 2026-10-09: First input's format.
    pub activation: Option<String>,
    /// 2026-10-09: First output's format.
    pub output: Option<String>,
    /// 2026-10-09: First input's feature width (K of a projection).
    pub in_dim: u64,
    /// 2026-10-09: First output's feature width (N of a projection).
    pub out_dim: u64,
    /// 2026-10-09: Rows of the launch (the plan's row count).
    pub rows: u64,
    /// 2026-10-09: The family's runtime parameters at the node (top_k, channels, ...).
    pub runtime: Values,
}

/// 2026-10-09: One point of the sweep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccuracyPoint {
    /// 2026-10-09: Family id.
    pub family: String,
    /// 2026-10-09: The kernels the plan group launches (`module::function` joined by ` + `).
    pub kernels: String,
    /// 2026-10-09: The compile-time and policy values.
    pub point: Values,
    /// 2026-10-09: The launch's shape.
    pub shape: Shape,
    /// 2026-10-09: Recipes that run it.
    pub users: BTreeSet<String>,
    /// 2026-10-09: Checkpoints that run it (a point two checkpoints share is a shared kernel).
    pub checkpoints: BTreeSet<String>,
    /// 2026-10-09: Kernel targets (`<hw>/<model>/<quant>`) of the instances that run it.
    pub targets: BTreeSet<String>,
    /// 2026-10-09: Node sites (`block.local`) that run it.
    pub sites: BTreeSet<String>,
}

/// 2026-10-09: The sweep of one hardware class.
#[derive(Debug, Clone)]
pub struct Sweep {
    /// 2026-10-09: Hardware class.
    pub hardware: String,
    /// 2026-10-09: Recipes swept.
    pub recipes: Vec<String>,
    /// 2026-10-09: The distinct points.
    pub points: Vec<AccuracyPoint>,
}

/// 2026-10-09: A sweep that cannot be built.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SweepError {
    /// 2026-10-09: A repository file or a circuit failed to load.
    #[error("{0}")]
    Load(String),
    /// 2026-10-09: No instance runs on the hardware.
    #[error("no instance in kernels/circuits/INSTANCES.toml targets `{0}`")]
    NoInstance(String),
}

fn hardware_of(inst: &Instance) -> &str {
    inst.target.split('/').next().unwrap_or("")
}

type Key = (String, Values, Shape, String);

/// 2026-10-09: Every point the instances on `hardware` run.
pub fn sweep(repo: &dyn Repo, hardware: &str) -> Result<Sweep, SweepError> {
    let load = |e: String| SweepError::Load(e);
    let all = parse_instances(&repo.read("kernels/circuits/INSTANCES.toml").map_err(load)?)
        .map_err(|e| load(e.to_string()))?;
    let mine: Vec<&Instance> = all.iter().filter(|i| hardware_of(i) == hardware).collect();
    if mine.is_empty() {
        return Err(SweepError::NoInstance(hardware.to_string()));
    }
    let fams = parse_families(
        &repo
            .read(&format!("kernels/{hardware}/common/KERNEL_FAMILIES.toml"))
            .map_err(load)?,
    )
    .map_err(|e| load(e.to_string()))?;
    let mut acc: BTreeMap<Key, AccuracyPoint> = BTreeMap::new();
    for inst in &mine {
        let loaded = load_instance(repo, inst).map_err(|e| load(e.to_string()))?;
        for (mode, rows) in &inst.plans {
            for &n in rows {
                add_run(&mut acc, inst, &loaded, &fams, *mode, n)?;
            }
        }
    }
    Ok(Sweep {
        hardware: hardware.to_string(),
        recipes: mine.iter().map(|i| i.recipe.clone()).collect(),
        points: acc.into_values().collect(),
    })
}

fn add_run(
    acc: &mut BTreeMap<Key, AccuracyPoint>,
    inst: &Instance,
    loaded: &metrale_circuit::Loaded,
    fams: &Families,
    mode: Mode,
    rows: u64,
) -> Result<(), SweepError> {
    let c = &loaded.circuit;
    let err =
        |e: String| SweepError::Load(format!("{} {} n={rows}: {e}", inst.recipe, mode.name()));
    // 2026-10-09: A golden instance is fused, so each node sits in the family its plan group
    // launches. A plan-only instance's rules do not cover it yet (the Venn's own rule): its
    // nodes are placed in the first family that implements them at this row count, and the
    // point says so (`kernels = "(by op)"`).
    let plan = if inst.golden {
        Some(
            fuse(
                c,
                &loaded.rules,
                &AvailableKernels::all_named_by(&loaded.rules),
                &inst.policy,
                mode,
                rows,
            )
            .map_err(|e| err(e.to_string()))?,
        )
    } else {
        None
    };
    let subject = Subject {
        recipe: &inst.recipe,
        circuit: c,
        settings: &inst.policy.settings,
        plan: plan.as_ref(),
    };
    let used: Vec<Usage<'_>> = match &plan {
        Some(_) => usages(&[subject], fams).map_err(|e| err(e.to_string()))?,
        None => by_op(&subject, fams, mode, rows).map_err(err)?,
    };
    for u in used {
        let n = u.node;
        let fmt = |e: Option<&usize>| e.map(|&e| c.edges[e].format.name());
        let dim = |e: Option<&usize>| e.map_or(0, |&e| c.edges[e].dim_value);
        let shape = Shape {
            op: n.op.name(),
            weight: n.weight.map(|w| w.name()),
            activation: fmt(n.inputs.first()),
            output: fmt(n.outputs.first()),
            in_dim: dim(n.inputs.first()),
            out_dim: dim(n.outputs.first()),
            rows,
            runtime: u.runtime.clone(),
        };
        let key = (
            u.family.id.clone(),
            u.point.clone(),
            shape.clone(),
            u.kernels.clone(),
        );
        let p = acc.entry(key).or_insert_with(|| AccuracyPoint {
            family: u.family.id.clone(),
            kernels: u.kernels.clone(),
            point: u.point.clone(),
            shape,
            users: BTreeSet::new(),
            checkpoints: BTreeSet::new(),
            targets: BTreeSet::new(),
            sites: BTreeSet::new(),
        });
        p.users.insert(inst.recipe.clone());
        p.checkpoints.insert(inst.checkpoint.clone());
        p.targets.insert(inst.target.clone());
        p.sites.insert(format!("{}.{}", n.block, n.local));
    }
    Ok(())
}

/// 2026-10-09: The nodes of `mode`'s section placed by op in their first candidate family.
fn by_op<'a>(
    s: &Subject<'a>,
    fams: &'a Families,
    mode: Mode,
    rows: u64,
) -> Result<Vec<Usage<'a>>, String> {
    let c = s.circuit;
    let section = section_of(mode);
    let mut out = Vec::new();
    for b in c.blocks.iter().filter(|b| b.section == section) {
        for idx in b.first..b.end {
            let node = &c.nodes[idx];
            let Some(family) = candidates(fams, c, node, rows).into_iter().next() else {
                continue;
            };
            let (point, runtime) = point_of(family, s, node).map_err(|e| e.to_string())?;
            out.push(Usage {
                recipe: s.recipe,
                node,
                family,
                kernels: "(by op)".to_string(),
                point,
                runtime,
            });
        }
    }
    Ok(out)
}

impl AccuracyPoint {
    /// 2026-10-09: The point's key in records and reports: family, values, kernels and shape.
    pub fn key(&self) -> String {
        let vals = self
            .point
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",");
        let s = &self.shape;
        // 2026-10-10: Two models can run one point and shape at different runtime values (the
        // MoE expert count and top-k of DeepSeek-V4 and GLM-5.3): the key names them.
        let rt = if s.runtime.is_empty() {
            String::new()
        } else {
            let r: Vec<String> = s.runtime.iter().map(|(k, v)| format!("{k}={v}")).collect();
            format!(" rt[{}]", r.join(","))
        };
        format!(
            "{}[{vals}] {} {} w={} a={} k={} n={} rows={}{rt}",
            self.family,
            self.kernels,
            s.op,
            s.weight.as_deref().unwrap_or("-"),
            s.activation.as_deref().unwrap_or("-"),
            s.in_dim,
            s.out_dim,
            s.rows
        )
    }

    /// 2026-10-09: Two or more checkpoints run this exact point (the Venn's shared kernel).
    pub fn shared(&self) -> bool {
        self.checkpoints.len() > 1
    }
}
