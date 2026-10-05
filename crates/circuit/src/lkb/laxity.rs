// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The modelled gain of a plan's fusions, its laxity (book/src/appendix/lkb-math.md,
//! section 5): per group, the bytes its fused edges never write to or read back from DRAM, the
//! repeated reads of an input several members share (a horizontal fusion reads it once), that
//! traffic's time at the device's bandwidth, and the launches the group saves over one launch
//! per member op.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - Only the round trip and the launch count are modelled. A launch's time is not: no device
//!   constant for it is measured, so launches saved are reported as a count.
//! - Energy laxity is never derived from time (`J = ∫ P dt`): it is reported unmeasured until a
//!   measurement exists.
//! - The per-node roofline ([`crate::venn::roofline::node_cost`]) is unchanged; laxity is added
//!   beside it, so no existing estimate moves.

#[cfg(test)]
#[path = "laxity_tests.rs"]
mod laxity_tests;

use crate::fuser::{EdgeState, FusionPlan};
use crate::ir::Circuit;
use crate::venn::roofline::{CostError, edge_bytes};

/// 2026-10-05: The laxity of one group.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupLaxity {
    /// 2026-10-05: The rule that formed it.
    pub rule: String,
    /// 2026-10-05: Its numerics tag.
    pub numerics: &'static str,
    /// 2026-10-05: Member ops.
    pub ops: usize,
    /// 2026-10-05: Edges produced and consumed inside it.
    pub fused_edges: usize,
    /// 2026-10-05: Bytes those edges would write and read back if materialized, plus the
    /// repeated reads of shared inputs, per step.
    pub bytes: f64,
    /// 2026-10-05: `bytes` at the device's DRAM bandwidth, microseconds per step.
    pub time_us: f64,
    /// 2026-10-05: Launches saved per step against one launch per member op (`None`: a
    /// `per_run` group, whose count depends on the run table).
    pub launches_saved: Option<u64>,
}

/// 2026-10-05: The laxity of one plan: every group with more than one member op.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PlanLaxity {
    /// 2026-10-05: Multi-op groups, in plan order.
    pub groups: Vec<GroupLaxity>,
}

impl PlanLaxity {
    /// 2026-10-05: Total modelled time, microseconds per step.
    pub fn time_us(&self) -> f64 {
        self.groups.iter().map(|g| g.time_us).sum()
    }

    /// 2026-10-05: Total bytes kept out of DRAM per step.
    pub fn bytes(&self) -> f64 {
        self.groups.iter().map(|g| g.bytes).sum()
    }

    /// 2026-10-05: Total launches saved per step (groups with a fixed count).
    pub fn launches_saved(&self) -> u64 {
        self.groups.iter().filter_map(|g| g.launches_saved).sum()
    }
}

/// 2026-10-05: The laxity of `plan` over circuit `c` at `dram_gbps` GB/s.
pub fn plan_laxity(
    c: &Circuit,
    plan: &FusionPlan,
    dram_gbps: f64,
) -> Result<PlanLaxity, CostError> {
    let mut groups = Vec::new();
    for (g, group) in plan.groups.iter().enumerate() {
        if group.nodes.len() < 2 {
            continue;
        }
        let mut bytes = 0.0;
        let mut fused_edges = 0;
        for (e, state) in plan.edge_states.iter().enumerate() {
            if *state != Some(EdgeState::Fused(g)) {
                continue;
            }
            let edge = &c.edges[e];
            let Some(producer) = edge.producer else {
                continue;
            };
            let size = edge_bytes(c, &c.nodes[producer], e, plan.rows)?;
            // 2026-10-05: One write and one read per consumer, all inside the group.
            bytes += size * (1 + edge.consumers.len()) as f64;
            fused_edges += 1;
        }
        // 2026-10-05: A materialized input that k > 1 members read is read once by the group.
        for (e, edge) in c.edges.iter().enumerate() {
            if plan.edge_states[e] != Some(EdgeState::Materialized) {
                continue;
            }
            let k = edge
                .consumers
                .iter()
                .filter(|n| group.nodes.contains(n))
                .count();
            if k > 1 {
                let reader = &c.nodes[edge.consumers[0]];
                bytes += edge_bytes(c, reader, e, plan.rows)? * (k - 1) as f64;
            }
        }
        // 2026-10-05: A kernel-less group (a host copy) launches nothing either way.
        let launches_saved = if group.kernels.is_empty() {
            None
        } else {
            group
                .repeat
                .count(plan.rows)
                .map(|n| (group.nodes.len() as u64).saturating_sub(group.kernels.len() as u64) * n)
        };
        groups.push(GroupLaxity {
            rule: group.rule.clone(),
            numerics: group.numerics.class(),
            ops: group.nodes.len(),
            fused_edges,
            bytes,
            time_us: bytes / (dram_gbps * 1e3),
            launches_saved,
        });
    }
    Ok(PlanLaxity { groups })
}
