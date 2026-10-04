// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Split from `fuser.rs` (an exact copy): the formats a plan stores its edges in,
//! the check that every pattern element reads the format it names, each edge's state, and the
//! edges a group reads and writes.
//!
//! Owner: metrale-circuit.
//! Invariants: as `fuser.rs`.

use std::collections::BTreeSet;

use super::{EdgeState, FuseError, FusionPlan, Group};
use crate::format::Format;
use crate::ir::{Circuit, EdgeIdx, NodeIdx};
use crate::rules::Rule;

fn rule_of<'a>(rules: &'a [Rule], id: &str) -> Option<&'a Rule> {
    rules.iter().find(|r| r.id == id)
}

pub(super) fn stored_formats(circuit: &Circuit, groups: &[Group], rules: &[Rule]) -> Vec<Format> {
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

pub(super) fn check_reads(
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

pub(super) fn edge_states(
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
