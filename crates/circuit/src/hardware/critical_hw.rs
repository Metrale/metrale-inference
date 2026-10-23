// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The critical pairs of one plan of a model on a device ([`crate::critical`]): the
//! matches the class's rules form, resolved each way through the planner's own primary arm
//! ([`super::plan::fuse_primary_seeded`]) and costed at the headroom floor.
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - The rules are the ones today's plan was chosen from: the class's, less the ones that arm
//!   refused. Placeholders cover only what no rule covers, so they are never in a pair.
//! - The cost is the headroom floor of the step: Σ over launch sites of
//!   `max(bytes / bandwidth, FLOPs / peak)` at the class's context assumption
//!   ([`super::headroom_sites::launch_sites`]), with no launch floor stated.
//! - A match's signature names it independently of its layer: rule, and each node's
//!   `block.local`, layer kind and weight format. Layers with the same signatures share one
//!   resolved representative.

use crate::prelude::*;

use super::HwError;
use super::headroom_sites::launch_sites;
use super::plan::{Planned, Resolved, fuse_primary_seeded};
use crate::critical::{CriticalPair, Outcome, critical_pairs};
use crate::fuser::{Match, Policy, match_instances};
use crate::ir::Circuit;
use crate::venn::Run;

/// 2026-10-23: The headroom floor of `p`, microseconds per step.
pub fn floor_us(r: &Resolved, c: &Circuit, policy: &Policy, p: &Planned) -> Result<f64, HwError> {
    Ok(launch_sites(r, c, &policy.settings, p, None)?
        .iter()
        .map(|s| s.mem_us.max(s.compute_us))
        .sum())
}

/// 2026-10-23: `block.local:kind:weight` per node, after the rule id.
pub fn signature(c: &Circuit, m: &Match) -> String {
    let nodes: Vec<String> = m
        .nodes
        .iter()
        .map(|&n| {
            let node = &c.nodes[n];
            format!(
                "{}.{}:{}:{}",
                node.block,
                node.local,
                node.layer.map_or("-", |l| c.layer_kinds[l].name()),
                node.weight.map_or_else(|| "-".into(), |w| w.name())
            )
        })
        .collect();
    format!("{} {}", m.rule, nodes.join(","))
}

/// 2026-10-23: A match's site as the plan tables name a group: `block [locals]`.
pub fn site(c: &Circuit, m: &Match) -> String {
    let locals: Vec<&str> = m.nodes.iter().map(|&n| c.nodes[n].local.as_str()).collect();
    format!("{} [{}]", c.nodes[m.nodes[0]].block, locals.join(","))
}

/// 2026-10-23: Today's plan of `c` at `run` on `r`, the number of matches over the free
/// circuit, and the critical pairs among them.
pub fn critical_pairs_on(
    r: &Resolved,
    c: &Circuit,
    policy: &Policy,
    run: Run,
) -> Result<(Planned, usize, Vec<CriticalPair>), HwError> {
    let today = fuse_primary_seeded(r, c, policy, run, &[])?;
    let rules: Vec<_> = r
        .rule_list()
        .iter()
        .filter(|x| !today.refused.iter().any(|(id, _)| *id == x.id))
        .cloned()
        .collect();
    let matches = match_instances(
        c,
        &rules,
        &r.availability.kernels,
        policy,
        (run.mode, run.rows),
    );
    let current: Vec<Match> = today
        .plan
        .groups
        .iter()
        .map(|g| Match {
            rule: g.rule.clone(),
            nodes: g.nodes.clone(),
        })
        .collect();
    let current_cost = floor_us(r, c, policy, &today)?;
    let pairs = critical_pairs(
        &matches,
        (&current, current_cost),
        |m| signature(c, m),
        |m| match fuse_primary_seeded(r, c, policy, run, core::slice::from_ref(m)) {
            Ok(p) => Ok(Ok(Outcome {
                digest: p.plan.digest.clone(),
                cost: floor_us(r, c, policy, &p)?,
                placeholders: p.novel.len(),
                launches: p.plan.launches(),
            })),
            Err(HwError::Plan(why)) => Ok(Err(why)),
            Err(e) => Err(e),
        },
    )?;
    Ok((today, matches.len(), pairs))
}
