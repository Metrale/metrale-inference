// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the kernel tree from a circuit's fusion plan (pure).
//!
//! The plan comes from the engine's own planner (`metrale-circuit`): the
//! instance's circuit is instantiated and fused for one mode and row count,
//! and each launch group becomes a node at
//! `<arch>/<mode>/<layer>/<rule>[<member locals>]`, with one leaf per kernel it
//! launches (`module::func`). Group nodes carry the join key a circuit
//! renderer needs: the group's index in the plan, its rule and the plan digest.
//!
//! Attribution is an ordered alignment, never a name lookup: each pinned decode
//! step (one launch of the segment's largest CUDA graph, the one step
//! definition) is aligned against the plan's launch sequence (groups in order,
//! each group's kernels once per repetition) by a longest common subsequence
//! over profiler short names. A launch the alignment does not match (a kernel
//! the plan does not launch, or one out of the plan's order) is unattributed
//! under its full name, as is every launch outside the pinned steps; a planned
//! launch with no match leaves its leaf with fewer launches. Nothing is
//! guessed: equal short names are matched only in order.

use std::collections::{BTreeMap, HashMap};

use indexmap::IndexMap;
use metrale_circuit::{Instance, Mode, PrecisionSpec, Sources, fuse, parse_instances};
use serde_json::{Map, Value, json};

use crate::core::error::{Error, Result};
use crate::core::model::Kernel;
use crate::core::repr::quote;
use crate::core::segment::pinned_steps;
use crate::core::tree::{Join, Loose, Node, Source, Tree, child};
use crate::fail;

/// The file an instance's precision is read from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrecisionFile {
    Table(String),
    CheckpointPlan(String),
}

/// What must be read to plan one instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Needs {
    pub arch: String,
    pub precision: PrecisionFile,
    /// The hardware class whose FUSIONS.toml applies (the target's first part).
    pub hw: String,
}

/// The texts one plan is built from.
#[derive(Debug, Clone)]
pub struct PlanTexts {
    pub instances: String,
    pub circuit: String,
    /// (name, text) of every block library the circuit includes.
    pub blocks: Vec<(String, String)>,
    pub precision: String,
    pub fusions: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanGroup {
    /// The group's index in the fusion plan.
    pub index: u64,
    pub rule: String,
    /// `layer.N`, or the prologue/epilogue block (`embed`, `head`).
    pub layer: String,
    /// `rule[member locals]`, `#n`-suffixed on a repeat within the layer.
    pub key: String,
    /// Member node locals, in pattern order.
    pub members: Vec<String>,
    /// Member ops, in pattern order.
    pub ops: Vec<String>,
    /// (module, func) of each kernel, in launch order.
    pub kernels: Vec<(String, String)>,
    /// Launches of each kernel per step.
    pub reps: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanView {
    pub recipe: String,
    pub arch: String,
    pub mode: String,
    pub rows: u64,
    pub digest: String,
    pub rules_digest: String,
    /// Groups that launch kernels, in execution order.
    pub groups: Vec<PlanGroup>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Alignment {
    pub steps: usize,
    pub launches_per_step: usize,
    /// Captured launches matched to a planned launch, over all steps.
    pub matched: u64,
    /// Planned launches no captured launch matched, over all steps.
    pub missed: u64,
    /// Captured launches inside the steps that matched nothing.
    pub unmatched: u64,
    /// Launches of the segment outside the pinned steps.
    pub outside_steps: u64,
}

/// A refusal to build a DP table larger than this many cells (u32 each); a
/// pinned step is a few thousand launches, so only a malformed input reaches it.
const MAX_CELLS: usize = 1 << 28;

fn instance(instances: &str, recipe: &str, arch: &str) -> Result<Instance> {
    let all = parse_instances(instances).map_err(|e| Error(e.to_string()))?;
    let Some(inst) = all.into_iter().find(|i| i.recipe == recipe) else {
        fail!(
            "circuit: no instance serves recipe {} (--policy)",
            quote(recipe)
        );
    };
    if inst.arch != arch {
        fail!(
            "circuit: recipe {} is an instance of arch {}, not {} (--circuit)",
            quote(recipe),
            quote(&inst.arch),
            quote(arch)
        );
    }
    Ok(inst)
}

/// What the CLI must read for `recipe`, which must be an instance of `arch`.
pub fn needs(instances: &str, recipe: &str, arch: &str) -> Result<Needs> {
    let inst = instance(instances, recipe, arch)?;
    let precision = match &inst.precision {
        PrecisionSpec::Table(n) => PrecisionFile::Table(n.clone()),
        PrecisionSpec::Policy {
            checkpoint_plan, ..
        } => PrecisionFile::CheckpointPlan(checkpoint_plan.clone()),
    };
    let Some(hw) = inst.target.split('/').next().filter(|h| !h.is_empty()) else {
        fail!(
            "circuit: target {} names no hardware class",
            quote(&inst.target)
        );
    };
    Ok(Needs {
        arch: inst.arch.clone(),
        precision,
        hw: hw.to_string(),
    })
}

/// The block libraries a circuit text includes.
pub fn includes(circuit: &str) -> Result<Vec<String>> {
    metrale_circuit::includes_of(circuit).map_err(|e| Error(e.to_string()))
}

/// Fuse the plan of `recipe` for `mode` at `rows`, with every kernel any rule
/// names available (the offline view `met circuit show` prints).
pub fn plan_view(
    t: &PlanTexts,
    recipe: &str,
    arch: &str,
    mode: &str,
    rows: u64,
) -> Result<PlanView> {
    let inst = instance(&t.instances, recipe, arch)?;
    let m = match Mode::parse(mode) {
        Some(Mode::VerifyBatch) => {
            fail!("circuit: mode verify_batch is planned per row table; not supported")
        }
        Some(m) => m,
        None => fail!("circuit: unknown mode {}", quote(mode)),
    };
    let blocks: Vec<(&str, &str)> = t
        .blocks
        .iter()
        .map(|(n, s)| (n.as_str(), s.as_str()))
        .collect();
    let src = Sources {
        circuit: &t.circuit,
        precision: &t.precision,
        rules: &t.fusions,
        blocks: &blocks,
    };
    let loaded = metrale_circuit::load(&inst, src).map_err(|e| Error(format!("circuit: {e}")))?;
    let available = metrale_circuit::AvailableKernels::all_named_by(&loaded.rules);
    let plan = fuse(
        &loaded.circuit,
        &loaded.rules,
        &available,
        &inst.policy,
        m,
        rows,
    )
    .map_err(|e| Error(format!("circuit: {e}")))?;
    let c = &loaded.circuit;
    let mut groups = Vec::new();
    let mut seen: HashMap<(String, String), u64> = HashMap::new();
    for (gi, g) in plan.groups.iter().enumerate() {
        let Some(reps) = g.repeat.count(plan.rows) else {
            fail!(
                "circuit: group {gi} ({}) repeats per run; a per-run plan is not supported",
                g.rule
            );
        };
        if g.kernels.is_empty() || reps == 0 {
            continue;
        }
        let nodes: Vec<&metrale_circuit::ir::Node> = g.nodes.iter().map(|&n| &c.nodes[n]).collect();
        let Some(first) = nodes.first() else {
            fail!("circuit: group {gi} ({}) has no member nodes", g.rule);
        };
        let layer = match first.layer {
            Some(l) => format!("layer.{l}"),
            None => first
                .id
                .rsplit_once('.')
                .map_or(first.block.clone(), |(p, _)| p.to_string()),
        };
        let members: Vec<String> = nodes.iter().map(|n| n.local.clone()).collect();
        let base = format!("{}[{}]", g.rule, members.join(","));
        let n = seen.entry((layer.clone(), base.clone())).or_insert(0);
        *n += 1;
        let key = if *n == 1 { base } else { format!("{base}#{n}") };
        groups.push(PlanGroup {
            index: gi as u64,
            rule: g.rule.clone(),
            layer,
            key,
            members,
            ops: nodes.iter().map(|n| n.op.name()).collect(),
            kernels: g
                .kernels
                .iter()
                .map(|k| (k.module.clone(), k.func.clone()))
                .collect(),
            reps,
        });
    }
    Ok(PlanView {
        recipe: inst.recipe.clone(),
        arch: plan.arch.clone(),
        mode: m.name().to_string(),
        rows: plan.rows,
        digest: plan.digest.clone(),
        rules_digest: loaded.rules_digest.clone(),
        groups,
    })
}

/// The planned launch sequence of one step: (group, kernel) positions and the
/// short name each is expected under.
fn expected(plan: &PlanView, names: &BTreeMap<String, String>) -> Vec<(usize, usize, String)> {
    let mut out = Vec::new();
    for (gi, g) in plan.groups.iter().enumerate() {
        for _ in 0..g.reps {
            for (ki, (module, func)) in g.kernels.iter().enumerate() {
                let short = names.get(&format!("{module}::{func}")).unwrap_or(func);
                out.push((gi, ki, short.clone()));
            }
        }
    }
    out
}

fn intern<'a>(ids: &mut HashMap<&'a str, u32>, s: &'a str) -> u32 {
    let next = ids.len() as u32;
    *ids.entry(s).or_insert(next)
}

/// Longest common subsequence of `a` and `b`: the matched (i, j) pairs, in order.
/// The walk back from the end takes an equal pair at once, so among equally
/// long alignments each captured launch takes the latest planned slot it can.
fn lcs(a: &[u32], b: &[u32]) -> Result<Vec<(usize, usize)>> {
    let w = b.len() + 1;
    let cells = (a.len() + 1).saturating_mul(w);
    if cells > MAX_CELLS {
        fail!(
            "circuit: a step of {} launches against a plan of {} is too large to align",
            a.len(),
            b.len()
        );
    }
    let mut dp = vec![0u32; cells];
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            dp[i * w + j] = if a[i - 1] == b[j - 1] {
                dp[(i - 1) * w + j - 1] + 1
            } else {
                dp[(i - 1) * w + j].max(dp[i * w + j - 1])
            };
        }
    }
    let (mut i, mut j, mut out) = (a.len(), b.len(), Vec::new());
    while i > 0 && j > 0 {
        if a[i - 1] == b[j - 1] {
            out.push((i - 1, j - 1));
            i -= 1;
            j -= 1;
        } else if dp[i * w + j - 1] >= dp[(i - 1) * w + j] {
            j -= 1;
        } else {
            i -= 1;
        }
    }
    out.reverse();
    Ok(out)
}

/// The tree of `segment` (one device's kernels, ordered by start) under `plan`.
pub fn circuit_tree(
    plan: &PlanView,
    names: &BTreeMap<String, String>,
    segment: &[Kernel],
    provenance: Map<String, Value>,
) -> Result<(Tree, Alignment)> {
    let exp = expected(plan, names);
    if exp.is_empty() {
        fail!("circuit: the plan launches no kernels");
    }
    let steps = pinned_steps(segment)?;
    let mut ids: HashMap<&str, u32> = HashMap::new();
    let planned: Vec<u32> = exp.iter().map(|(_, _, s)| intern(&mut ids, s)).collect();
    let mut slots: Vec<Vec<Node>> = plan
        .groups
        .iter()
        .map(|g| vec![Node::default(); g.kernels.len()])
        .collect();
    let mut matched_at = vec![false; segment.len()];
    let mut in_step = vec![false; segment.len()];
    let mut al = Alignment {
        steps: steps.steps.len(),
        launches_per_step: exp.len(),
        matched: 0,
        missed: 0,
        unmatched: 0,
        outside_steps: 0,
    };
    for step in &steps.steps {
        let captured: Vec<u32> = step
            .iter()
            .map(|&i| intern(&mut ids, &segment[i].name))
            .collect();
        let pairs = lcs(&captured, &planned)?;
        for &(ci, pj) in &pairs {
            let k = &segment[step[ci]];
            let (g, slot, _) = exp[pj];
            slots[g][slot].record(k);
            matched_at[step[ci]] = true;
        }
        for &i in step {
            in_step[i] = true;
        }
        al.matched += pairs.len() as u64;
        al.missed += (exp.len() - pairs.len()) as u64;
        al.unmatched += (step.len() - pairs.len()) as u64;
    }
    let mut unattributed: IndexMap<String, Loose> = IndexMap::new();
    for (i, k) in segment.iter().enumerate() {
        if !matched_at[i] {
            al.outside_steps += u64::from(!in_step[i]);
            unattributed
                .entry(k.demangled.clone())
                .or_default()
                .record(k);
        }
    }
    let mut roots: IndexMap<String, Node> = IndexMap::new();
    let mode = child(&mut child(&mut roots, &plan.arch)?.kernels, &plan.mode)?;
    for (g, leaves) in plan.groups.iter().zip(slots) {
        let layer = child(&mut mode.kernels, &g.layer)?;
        let node = child(&mut layer.kernels, &g.key)?;
        node.circuit = Some(Join {
            group: g.index,
            rule: g.rule.clone(),
            plan_digest: plan.digest.clone(),
        });
        node.members = g.members.clone();
        if g.members.len() >= 2 {
            node.fuses = g.ops.clone();
        }
        for ((module, func), leaf) in g.kernels.iter().zip(leaves) {
            child(&mut node.kernels, &format!("{module}::{func}"))?.absorb(leaf);
        }
    }
    let mut tree = Tree {
        provenance,
        source: Source::Circuit,
        kernels: roots,
        unattributed,
        baseline: None,
        slope: None,
    };
    tree.seal();
    Ok((tree, al))
}

/// The circuit block of a circuit tree's provenance.
pub fn plan_provenance(plan: &PlanView, al: &Alignment) -> Value {
    json!({
        "arch": plan.arch,
        "mode": plan.mode,
        "rows": plan.rows,
        "recipe": plan.recipe,
        "plan_digest": plan.digest,
        "rules_digest": plan.rules_digest,
        "available_kernels": "all named by the rules (offline view)",
        "launches_per_step": al.launches_per_step,
        "aligned_steps": al.steps,
        "matched": al.matched,
        "missed": al.missed,
        "unmatched": al.unmatched,
        "outside_steps": al.outside_steps,
    })
}
