// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The fuser: cover a circuit's nodes with rule groups for one mode and row count.
//!
//! Greedy and deterministic. Rules are tried by priority (highest first, then id), and each
//! rule scans the nodes in execution order; a node joins at most one group. A rule is skipped
//! when its mode or rows do not apply, a kernel is absent, a capability is missing, a policy
//! setting it names differs, or its `differs` lever is not opted in.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - Every node of the mode's section ends in exactly one group, or `fuse` returns
//!   [`FuseError::Uncovered`]; no node is left to an implicit kernel.
//! - A policy must state every setting any rule reads ([`FuseError::PolicyMissing`]).
//! - The plan is a pure function of its inputs; [`FusionPlan::digest`] changes exactly when
//!   what the plan runs changes (see [`crate::digest`]).

use std::collections::{BTreeMap, BTreeSet};

use crate::format::Format;
use crate::ir::{Circuit, EdgeIdx, NodeIdx, Section};
use crate::rules::{KernelId, Mode, Numerics, Repeat, Rule};

#[path = "fuser_match.rs"]
mod fuser_match;

pub(crate) use fuser_match::fits as pattern_fits;

/// 2026-09-28: The kernels and capabilities a target provides (the boot probe's answer).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AvailableKernels {
    /// 2026-09-28: Present entry points.
    pub kernels: BTreeSet<KernelId>,
    /// 2026-09-28: Present capability bits (`KernelCaps`).
    pub caps: BTreeSet<String>,
}

impl AvailableKernels {
    /// 2026-09-28: Every kernel and capability any of `rules` names: the offline view used
    /// when no target is probed (`met circuit show`).
    pub fn all_named_by(rules: &[Rule]) -> Self {
        AvailableKernels {
            kernels: rules
                .iter()
                .flat_map(|r| r.kernels.iter().cloned())
                .collect(),
            caps: rules
                .iter()
                .flat_map(|r| r.requires.iter().cloned())
                .collect(),
        }
    }
}

/// 2026-09-28: What the recipe decides: opt-in levers for `differs` rules, and the settings
/// (`kv_cache_dtype`, `row_tiers`, ...) that `when` clauses read.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Policy {
    /// 2026-09-28: Levers whose `differs` rules may be selected.
    pub opt_in_levers: BTreeSet<String>,
    /// 2026-09-28: Settings, all stated explicitly.
    pub settings: BTreeMap<String, String>,
}

/// 2026-09-28: Whether an edge is written to memory or stays inside one kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EdgeState {
    /// 2026-09-28: Written to a buffer between launches.
    Materialized,
    /// 2026-09-28: Produced and consumed inside this group only.
    Fused(usize),
}

/// 2026-09-28: One launch group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Group {
    /// 2026-09-28: The rule that formed it.
    pub rule: String,
    /// 2026-09-28: The kernels its emitter launches.
    pub kernels: Vec<KernelId>,
    /// 2026-09-28: Launch repetition.
    pub repeat: Repeat,
    /// 2026-09-29: Copy-engine transfers besides the kernels, and how often.
    pub copies: Option<Repeat>,
    /// 2026-09-28: Emitter id.
    pub emitter: String,
    /// 2026-09-28: Numerics class of the rule.
    pub numerics: Numerics,
    /// 2026-09-28: Member nodes, in pattern order.
    pub nodes: Vec<NodeIdx>,
    /// 2026-09-30: A `per_run` group's launches for each run of the plan's row table, in batch
    /// order; empty for any other repeat.
    pub runs: Vec<crate::runs::RunLaunches>,
}

impl Group {
    /// 2026-09-30: Kernel launches per step at `rows` rows (per run for a `per_run` group).
    pub fn launch_count(&self, rows: u64) -> u64 {
        match self.repeat.count(rows) {
            Some(c) => self.kernels.len() as u64 * c,
            None => self.runs.iter().map(|r| r.launch_count()).sum(),
        }
    }

    /// 2026-09-30: Copy-engine transfers per step at `rows` rows.
    pub fn copy_count(&self, rows: u64) -> u64 {
        match self.repeat {
            crate::rules::Repeat::PerRun => self.runs.iter().map(|r| r.copy_count()).sum(),
            _ => self.copies.and_then(|c| c.count(rows)).unwrap_or(0),
        }
    }
}

/// 2026-09-28: The fused plan of one circuit for one mode and row count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionPlan {
    /// 2026-09-28: The circuit's arch.
    pub arch: String,
    /// 2026-09-28: Mode.
    pub mode: Mode,
    /// 2026-09-28: Padded rows the plan was chosen for.
    pub rows: u64,
    /// 2026-09-30: The row table of a [`Mode::VerifyBatch`] plan; `None` in every other mode.
    pub table: Option<crate::runs::RowTable>,
    /// 2026-09-28: Groups in execution order.
    pub groups: Vec<Group>,
    /// 2026-09-28: Per edge: `None` outside this mode's section, else its state.
    pub edge_states: Vec<Option<EdgeState>>,
    /// 2026-09-28: Per edge: the format it is stored in under this plan.
    pub edge_formats: Vec<Format>,
    /// 2026-09-28: Lower-case hex SHA-256 (see [`crate::digest`]).
    pub digest: String,
}

impl FusionPlan {
    /// 2026-09-28: Kernel launches per step: kernels times repetitions, over every group.
    pub fn launches(&self) -> u64 {
        self.groups.iter().map(|g| g.launch_count(self.rows)).sum()
    }

    /// 2026-09-29: Copy-engine transfers per step, over every group.
    pub fn copies(&self) -> u64 {
        self.groups.iter().map(|g| g.copy_count(self.rows)).sum()
    }
}

/// 2026-09-28: Why no plan was produced.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FuseError {
    /// 2026-09-28: No applicable rule covers a node.
    #[error("no applicable rule covers node `{node}` ({op}) in {mode} at {rows} rows")]
    Uncovered {
        /// 2026-09-28: Node id.
        node: String,
        /// 2026-09-28: Its op.
        op: String,
        /// 2026-09-28: Mode.
        mode: &'static str,
        /// 2026-09-28: Rows.
        rows: u64,
    },
    /// 2026-09-28: A rule reads a setting the policy does not state.
    #[error("rule `{rule}` reads setting `{key}`, which the policy does not state")]
    PolicyMissing {
        /// 2026-09-28: Rule id.
        rule: String,
        /// 2026-09-28: Setting.
        key: String,
    },
    /// 2026-09-28: A kernel reads an edge in a format its producer does not store.
    #[error("edge `{edge}` is stored as {stored} but rule `{rule}` reads it as {expected}")]
    FormatConflict {
        /// 2026-09-28: Edge id.
        edge: String,
        /// 2026-09-28: Stored format.
        stored: String,
        /// 2026-09-28: Reading rule.
        rule: String,
        /// 2026-09-28: Format it reads.
        expected: String,
    },
    /// 2026-09-28: The mode's section has no nodes.
    #[error("the circuit has no {0} nodes")]
    EmptySection(&'static str),
    /// 2026-09-28: The groups cannot be ordered (a rule set that fuses across a dependency).
    #[error("the {0} groups formed do not have an execution order")]
    Cycle(usize),
    /// 2026-09-28: `rows` is zero.
    #[error("a plan needs at least one row")]
    ZeroRows,
    /// 2026-09-30: A `verify_batch` plan without a row table, or a table in another mode.
    #[error("{0}")]
    RowTable(String),
}

/// 2026-09-28: The section a mode runs.
pub fn section_of(mode: Mode) -> Section {
    match mode {
        Mode::Draft => Section::Draft,
        Mode::Decode | Mode::MultiSeq | Mode::Verify | Mode::VerifyBatch => Section::Main,
    }
}

/// 2026-09-28: Fuse `circuit` for `mode` at `rows` padded rows. A `verify_batch` plan needs a
/// row table: [`fuse_table`].
pub fn fuse(
    circuit: &Circuit,
    rules: &[Rule],
    available: &AvailableKernels,
    policy: &Policy,
    mode: Mode,
    rows: u64,
) -> Result<FusionPlan, FuseError> {
    if mode == Mode::VerifyBatch {
        return Err(FuseError::RowTable(
            "a verify_batch plan is fused for a row table (`fuse_table`)".into(),
        ));
    }
    fuse_inner(circuit, rules, available, policy, (mode, rows, None))
}

/// 2026-09-30: Fuse `circuit` for a batched verify of `table`.
pub fn fuse_table(
    circuit: &Circuit,
    rules: &[Rule],
    available: &AvailableKernels,
    policy: &Policy,
    table: &crate::runs::RowTable,
) -> Result<FusionPlan, FuseError> {
    let rows = table.rows();
    fuse_inner(
        circuit,
        rules,
        available,
        policy,
        (Mode::VerifyBatch, rows, Some(table)),
    )
}

fn fuse_inner(
    circuit: &Circuit,
    rules: &[Rule],
    available: &AvailableKernels,
    policy: &Policy,
    (mode, rows, table): (Mode, u64, Option<&crate::runs::RowTable>),
) -> Result<FusionPlan, FuseError> {
    if rows == 0 {
        return Err(FuseError::ZeroRows);
    }
    for r in rules {
        if let Some(key) = r.when.keys().find(|k| !policy.settings.contains_key(*k)) {
            return Err(FuseError::PolicyMissing {
                rule: r.id.clone(),
                key: key.clone(),
            });
        }
    }
    let section = section_of(mode);
    let block_of = block_index(circuit);
    let in_scope: Vec<bool> = block_of
        .iter()
        .map(|&b| circuit.blocks[b].section == section)
        .collect();
    if !in_scope.iter().any(|&s| s) {
        return Err(FuseError::EmptySection(mode.name()));
    }
    let mut order: Vec<&Rule> = rules
        .iter()
        .filter(|r| applies(r, available, policy, mode, rows))
        .filter(|r| {
            r.runs.is_empty()
                || table.is_some_and(|t| crate::runs::resolve_runs(t, &r.runs).is_some())
        })
        .collect();
    order.sort_by(|a, b| b.priority.cmp(&a.priority).then_with(|| a.id.cmp(&b.id)));

    let mut owner: Vec<Option<usize>> = vec![None; circuit.nodes.len()];
    let mut found: Vec<(&Rule, Vec<NodeIdx>)> = Vec::new();
    let m = fuser_match::Matcher {
        circuit,
        in_scope: &in_scope,
        block_of: &block_of,
    };
    for rule in order {
        for s in 0..circuit.nodes.len() {
            if !in_scope[s] || owner[s].is_some() {
                continue;
            }
            if let Some(chain) = m.chain_at(rule, s, &owner) {
                for &n in &chain {
                    owner[n] = Some(found.len());
                }
                found.push((rule, chain));
            }
        }
    }
    if let Some(n) = (0..circuit.nodes.len()).find(|&n| in_scope[n] && owner[n].is_none()) {
        return Err(FuseError::Uncovered {
            node: circuit.nodes[n].id.clone(),
            op: circuit.nodes[n].op.name(),
            mode: mode.name(),
            rows,
        });
    }
    let exec = fuser_match::execution_order(circuit, &in_scope, &owner, found.len());
    if exec.len() != found.len() {
        return Err(FuseError::Cycle(found.len()));
    }
    let mut remap = vec![0usize; found.len()];
    for (new, &old) in exec.iter().enumerate() {
        remap[old] = new;
    }
    let groups: Vec<Group> = exec
        .iter()
        .map(|&g| {
            let (rule, nodes) = &found[g];
            Group {
                rule: rule.id.clone(),
                kernels: rule.kernels.clone(),
                repeat: rule.repeat,
                copies: rule.copies,
                emitter: rule.emitter.clone(),
                numerics: rule.numerics.clone(),
                nodes: nodes.clone(),
                runs: table
                    .and_then(|t| crate::runs::resolve_runs(t, &rule.runs))
                    .filter(|_| !rule.runs.is_empty())
                    .unwrap_or_default(),
            }
        })
        .collect();
    let owner: Vec<Option<usize>> = owner.iter().map(|o| o.map(|g| remap[g])).collect();
    let edge_formats = stored_formats(circuit, &groups, rules);
    check_reads(circuit, &groups, rules, &edge_formats)?;
    // 2026-09-30: Outputs a pattern element marks `stored` stay in memory.
    let stored: BTreeSet<EdgeIdx> = found
        .iter()
        .flat_map(|(rule, nodes)| {
            rule.pattern
                .iter()
                .zip(nodes)
                .filter(|(p, _)| p.stored)
                .flat_map(|(_, &n)| circuit.nodes[n].outputs.iter().copied())
        })
        .collect();
    let edge_states = edge_states(circuit, &in_scope, &owner, &stored);
    let mut plan = FusionPlan {
        arch: circuit.arch.clone(),
        mode,
        rows,
        table: table.cloned(),
        groups,
        edge_states,
        edge_formats,
        digest: String::new(),
    };
    plan.digest = crate::digest::plan_digest(circuit, &plan);
    Ok(plan)
}

fn applies(r: &Rule, avail: &AvailableKernels, policy: &Policy, mode: Mode, rows: u64) -> bool {
    r.modes.contains(&mode)
        && (r.rows.0..=r.rows.1).contains(&rows)
        && r.kernels.iter().all(|k| avail.kernels.contains(k))
        && r.requires.iter().all(|c| avail.caps.contains(c))
        && r.when
            .iter()
            .all(|(k, v)| policy.settings.get(k) == Some(v))
        && match &r.numerics {
            Numerics::Differs { lever } => policy.opt_in_levers.contains(lever),
            Numerics::BitIdentical { .. } | Numerics::Reference => true,
        }
}

fn block_index(circuit: &Circuit) -> Vec<usize> {
    let mut out = vec![0; circuit.nodes.len()];
    for (b, blk) in circuit.blocks.iter().enumerate() {
        for slot in &mut out[blk.first..blk.end] {
            *slot = b;
        }
    }
    out
}

fn rule_of<'a>(rules: &'a [Rule], id: &str) -> Option<&'a Rule> {
    rules.iter().find(|r| r.id == id)
}

fn stored_formats(circuit: &Circuit, groups: &[Group], rules: &[Rule]) -> Vec<Format> {
    let mut out: Vec<Format> = circuit.edges.iter().map(|e| e.format).collect();
    for g in groups {
        let Some(rule) = rule_of(rules, &g.rule) else {
            continue;
        };
        for (p, &n) in rule.pattern.iter().zip(&g.nodes) {
            if let Some(f) = p.writes {
                for &e in &circuit.nodes[n].outputs {
                    out[e] = f;
                }
            }
        }
    }
    out
}

fn check_reads(
    circuit: &Circuit,
    groups: &[Group],
    rules: &[Rule],
    stored: &[Format],
) -> Result<(), FuseError> {
    for g in groups {
        let Some(rule) = rule_of(rules, &g.rule) else {
            continue;
        };
        for (p, &n) in rule.pattern.iter().zip(&g.nodes) {
            let (Some(want), Some(&e)) = (p.input, circuit.nodes[n].inputs.first()) else {
                continue;
            };
            if stored[e] != want {
                return Err(FuseError::FormatConflict {
                    edge: circuit.edges[e].id.clone(),
                    stored: stored[e].name(),
                    rule: rule.id.clone(),
                    expected: want.name(),
                });
            }
        }
    }
    Ok(())
}

fn edge_states(
    circuit: &Circuit,
    in_scope: &[bool],
    owner: &[Option<usize>],
    stored: &BTreeSet<EdgeIdx>,
) -> Vec<Option<EdgeState>> {
    circuit
        .edges
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let producer_in = e.producer.is_some_and(|p| in_scope[p]);
            let read_in = e.consumers.iter().any(|&c| in_scope[c]);
            if !producer_in && !read_in {
                return None;
            }
            let g = e.producer.filter(|&p| in_scope[p]).and_then(|p| owner[p]);
            let internal = g.is_some()
                && !e.is_output
                && !stored.contains(&i)
                && !e.consumers.is_empty()
                && e.consumers.iter().all(|&c| owner[c] == g);
            Some(match (internal, g) {
                (true, Some(g)) => EdgeState::Fused(g),
                _ => EdgeState::Materialized,
            })
        })
        .collect()
}

/// 2026-09-28: The edges a group reads from outside itself and the ones it writes, in edge
/// order. Used by the renderers and the buffer planner.
pub fn group_io(circuit: &Circuit, plan: &FusionPlan, g: usize) -> (Vec<EdgeIdx>, Vec<EdgeIdx>) {
    let members: BTreeSet<NodeIdx> = plan.groups[g].nodes.iter().copied().collect();
    let mut ins = BTreeSet::new();
    let mut outs = BTreeSet::new();
    for &n in &members {
        let node = &circuit.nodes[n];
        for &e in &node.inputs {
            if circuit.edges[e]
                .producer
                .is_none_or(|p| !members.contains(&p))
            {
                ins.insert(e);
            }
        }
        for &e in &node.outputs {
            if plan.edge_states[e] == Some(EdgeState::Materialized) {
                outs.insert(e);
            }
        }
    }
    (ins.into_iter().collect(), outs.into_iter().collect())
}

#[cfg(test)]
#[path = "fuser_tests.rs"]
mod fuser_tests;
