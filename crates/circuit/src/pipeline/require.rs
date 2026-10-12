// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The REQUIRED pipeline of a node: what the circuit's formats, under the serving
//! policy and the plan's stated routing, ask each step to compute at. This is the "what we
//! need" view (`met circuit precision`) and the side every kernel declaration is checked
//! against ([`super::check`]).
//!
//! Where each requirement comes from:
//! - inputs and outputs: the formats the plan stores each edge in (the circuit's, or a rule's
//!   `writes`); inside a group, the format a rule states the edge is handed on in (`holds`);
//! - `act`: the input format, unless `activation_quantization` routes the projection's family
//!   to a fixed format at this row count, or, under `adaptive` (today's routing, which the rules
//!   encode), the rule states the format its kernels run the activation at;
//! - `weight`: the node's weight format, its values widened to the activation's precision for a
//!   16/32-bit activation, consumed natively (E4M3, E2M1) under an 8- or 4-bit one (an NVFP4
//!   weight under FP8 activations by the exact E2M1 to E4M3 conversion);
//! - `cache`: `kv_cache_dtype`; `state`: the state's declared format, a keyed one from its
//!   setting (`ssm_h_storage` from `ssm_h_dtype`);
//! - every accumulation, scale application, element-wise compute, softmax, routing score and
//!   blend sum: FP32, the circuit's reference definition of those ops;
//! - any step a plan's rule states (`steps` on a pattern element): the stated value. A rule
//!   states where today's routing departs from the reference; the plan prints it, and the
//!   kernel must declare the same.
//!
//! Owner: metrale-circuit (pipeline).
//! Invariants:
//! - Pure; a setting the requirement reads and the policy does not state is an error.
//! - The base requirement (no plan) and a plan's requirement differ only where the plan states
//!   something: a stored, held or stated format.

use std::collections::BTreeMap;

use super::vocab::steps_of;
use super::{NodePipeline, Num, PipelineError, Step, StepKind, Value};
use crate::format::Format;
use crate::fuser::{EdgeState, FusionPlan};
use crate::ir::{Circuit, EdgeIdx, Node, NodeIdx, OpKind};
use crate::rules::Rule;
use crate::state::{StateAccess, StateDtype, StateFormat};

/// 2026-10-02: Everything a requirement reads besides the node.
#[derive(Debug, Clone)]
pub struct Need<'a> {
    /// 2026-10-02: The circuit.
    pub circuit: &'a Circuit,
    /// 2026-10-02: The policy's settings.
    pub settings: &'a BTreeMap<String, String>,
    /// 2026-10-02: The row count the plan runs (the activation-quantization ladder reads it).
    pub rows: u64,
    /// 2026-10-02: Each edge's stored format.
    pub stored: Vec<Format>,
    /// 2026-10-02: Each node's group under the plan; all `None` without one.
    pub group_of: Vec<Option<usize>>,
    /// 2026-10-02: Each edge's state under the plan; all `None` without one.
    pub edge_states: Vec<Option<EdgeState>>,
    /// 2026-10-02: In-group hand-off formats the plan's rules state (`holds`), by edge.
    pub held: BTreeMap<EdgeIdx, Format>,
    /// 2026-10-02: Step values the plan's rules state (`steps`), by node.
    pub stated: BTreeMap<NodeIdx, BTreeMap<StepKind, Value>>,
}

impl<'a> Need<'a> {
    /// 2026-10-02: The requirement before any plan: the circuit's own formats.
    pub fn base(circuit: &'a Circuit, settings: &'a BTreeMap<String, String>, rows: u64) -> Self {
        Need {
            circuit,
            settings,
            rows,
            stored: circuit.edges.iter().map(|e| e.format).collect(),
            group_of: vec![None; circuit.nodes.len()],
            edge_states: vec![None; circuit.edges.len()],
            held: BTreeMap::new(),
            stated: BTreeMap::new(),
        }
    }

    /// 2026-10-02: The requirement under `plan`, whose groups' rules are among `rules` (a
    /// group whose rule is not there, a hardware planner's placeholder, states nothing).
    pub fn of_plan(
        circuit: &'a Circuit,
        plan: &FusionPlan,
        rules: &[Rule],
        settings: &'a BTreeMap<String, String>,
    ) -> Self {
        let mut need = Need::base(circuit, settings, plan.rows);
        need.stored.clone_from(&plan.edge_formats);
        need.edge_states.clone_from(&plan.edge_states);
        for (g, grp) in plan.groups.iter().enumerate() {
            for &n in &grp.nodes {
                need.group_of[n] = Some(g);
            }
            let Some(rule) = rules.iter().find(|r| r.id == grp.rule) else {
                continue;
            };
            for (p, &n) in rule.pattern.iter().zip(&grp.nodes) {
                if let Some(f) = p.holds {
                    for &e in &circuit.nodes[n].outputs {
                        need.held.insert(e, f);
                    }
                }
                if !p.steps.is_empty() {
                    need.stated.insert(n, p.steps.clone());
                }
            }
        }
        need
    }

    fn input(&self, node: NodeIdx, e: EdgeIdx) -> Format {
        match self.held.get(&e) {
            Some(f) if self.held_input(node, e) => *f,
            _ => self.stored[e],
        }
    }

    /// 2026-10-02: Node `node` reads edge `e` from a producer in its own group, in a format a
    /// rule states that producer holds it in.
    pub fn held_input(&self, node: NodeIdx, e: EdgeIdx) -> bool {
        self.held.contains_key(&e)
            && self.circuit.edges[e].producer.is_some_and(|p| {
                self.group_of[p].is_some() && self.group_of[p] == self.group_of[node]
            })
    }

    fn output(&self, e: EdgeIdx) -> Format {
        match (self.edge_states[e], self.held.get(&e)) {
            (Some(EdgeState::Fused(_)), Some(f)) => *f,
            _ => self.stored[e],
        }
    }

    /// 2026-10-02: Edge `e` is handed on inside its group in a format a rule states.
    pub fn holds_in_group(&self, e: EdgeIdx) -> bool {
        self.held.contains_key(&e) && matches!(self.edge_states[e], Some(EdgeState::Fused(_)))
    }

    /// 2026-10-02: The steps a rule states for node `n`.
    pub fn stated_steps(&self, n: NodeIdx) -> impl Iterator<Item = StepKind> + '_ {
        self.stated
            .get(&n)
            .into_iter()
            .flat_map(|m| m.keys().copied())
    }
}

/// 2026-10-02: The pipeline `need` requires of node `n`.
pub fn required(need: &Need<'_>, n: NodeIdx) -> Result<NodePipeline, PipelineError> {
    let node = &need.circuit.nodes[n];
    let fail = |detail: String| PipelineError::Required {
        node: node.id.clone(),
        op: node.op.name(),
        detail,
    };
    let inputs: Vec<Format> = node.inputs.iter().map(|&e| need.input(n, e)).collect();
    let outputs: Vec<Format> = node.outputs.iter().map(|&e| need.output(e)).collect();
    let stated = need.stated.get(&n);
    let first = |what: &str| {
        inputs
            .first()
            .copied()
            .ok_or_else(|| fail(format!("{what} needs an input")))
    };
    let mut projection: Option<(Format, Format)> = None;
    let mut steps = Vec::new();
    for &kind in steps_of(&node.op) {
        let said = stated.and_then(|m| m.get(&kind)).cloned();
        let value = match (kind, said) {
            (StepKind::Act, said) => {
                let weight = node
                    .weight
                    .ok_or_else(|| fail("a projection without a weight format".into()))?;
                let said = match said {
                    Some(Value::Format(f)) => Some(f),
                    Some(other) => return Err(fail(format!("a stated act `{}`", other.name()))),
                    None => None,
                };
                let act = super::act_policy::required_act(
                    need.settings,
                    &node.op,
                    need.rows,
                    (first("a projection")?, said, weight),
                )
                .map_err(fail)?;
                projection = Some((act, weight));
                Value::Format(act)
            }
            (_, Some(v)) => v,
            (StepKind::Gather, None) => match node.op {
                OpKind::Embed => Value::Format(
                    outputs
                        .first()
                        .copied()
                        .ok_or_else(|| fail("an embedding writes its rows".into()))?,
                ),
                _ => Value::Format(first("a gather")?),
            },
            (StepKind::Weight | StepKind::Mma | StepKind::Scale, None) => {
                let (act, weight) =
                    projection.ok_or_else(|| fail("`act` precedes the weight steps".into()))?;
                derived(kind, act, weight).map_err(fail)?
            }
            (
                StepKind::Accumulate
                | StepKind::Compute
                | StepKind::Scores
                | StepKind::Softmax
                | StepKind::Score
                | StepKind::Scatter
                | StepKind::Reduce,
                None,
            ) => Value::Num(Num::F32),
            (StepKind::Combine, None) => Value::OptNum((inputs.len() > 2).then_some(Num::F32)),
            (StepKind::Move | StepKind::Compare, None) => Value::Format(first("a move")?),
            (StepKind::Cache, None) => Value::Kv(
                need.settings
                    .get("kv_cache_dtype")
                    .cloned()
                    .ok_or_else(|| fail("the policy states no `kv_cache_dtype`".into()))?,
            ),
            (StepKind::State, None) => Value::State(state_of(need, node).map_err(fail)?),
        };
        steps.push(Step { kind, value });
    }
    Ok(NodePipeline {
        inputs,
        steps,
        outputs,
    })
}

/// 2026-10-02: The weight, multiply and scale requirement of an `act` x `weight` projection.
fn derived(kind: StepKind, act: Format, weight: Format) -> Result<Value, String> {
    if matches!(act, Format::Mxfp4) || matches!(weight, Format::Mxfp4) {
        return Err("MXFP4 E8M0/group32 pipeline is not implemented".into());
    }
    let operand = match (act, weight) {
        (Format::Bf16, _) => Num::Bf16,
        (Format::F32, _) => Num::F32,
        (Format::Fp8E4m3 { .. }, Format::Fp8E4m3 { .. } | Format::Nvfp4 { .. }) => Num::E4m3,
        (Format::Nvfp4 { .. }, Format::Nvfp4 { .. }) => Num::E2m1,
        (a, w) => {
            return Err(format!(
                "no multiply takes a {w} weight under a {a} activation"
            ));
        }
    };
    Ok(match kind {
        StepKind::Weight => Value::Weight {
            stored: weight,
            operand,
        },
        StepKind::Mma => Value::Mma {
            a: Num::of_format(act).ok_or_else(|| format!("an activation in {act}"))?,
            b: operand,
        },
        _ => Value::OptNum((!(act.is_plain() && weight.is_plain())).then_some(Num::F32)),
    })
}

/// 2026-10-02: The precision of the state `node` updates (or snapshots).
fn state_of(need: &Need<'_>, node: &Node) -> Result<StateDtype, String> {
    let want = match node.op {
        OpKind::StateSnapshot => StateAccess::Snapshot,
        _ => StateAccess::Update,
    };
    let (idx, _) = node
        .state
        .iter()
        .find(|(_, a)| *a == want)
        .ok_or_else(|| format!("the node touches no state as `{want:?}`"))?;
    let decl = &need.circuit.states[*idx];
    match &decl.format {
        StateFormat::Fixed(d) => Ok(*d),
        StateFormat::Keyed(key) => keyed_state(key, need.settings),
    }
}

/// 2026-10-02: The precision a keyed state format holds under `settings`. The recurrent h-state
/// (`ssm_h_storage`) holds the precision `--ssm-h-dtype` names: FP16 under `f16` and
/// `f16-pool` (which differ only in the pool's size), FP32 under `f32`.
pub fn keyed_state(key: &str, settings: &BTreeMap<String, String>) -> Result<StateDtype, String> {
    let setting = match key {
        "ssm_h_storage" => "ssm_h_dtype",
        other => other,
    };
    let value = settings
        .get(setting)
        .ok_or_else(|| format!("the policy states no `{setting}`"))?;
    match (key, value.as_str()) {
        ("ssm_h_storage", "f32") => Ok(StateDtype::F32),
        ("ssm_h_storage", "f16" | "f16-pool") => Ok(StateDtype::F16),
        ("ssm_h_storage", other) => Err(format!(
            "`ssm_h_dtype = {other}` names no h-state precision"
        )),
        (_, other) => StateDtype::parse(other)
            .ok_or_else(|| format!("`{setting} = {other}` is no state dtype")),
    }
}

#[cfg(test)]
#[path = "require_tests.rs"]
mod require_tests;
