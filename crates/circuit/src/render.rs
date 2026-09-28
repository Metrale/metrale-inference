// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The stable text rendering of a plan: the golden files under
//! `kernels/circuits/plans/` and `met circuit show`.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - The text is a pure function of its inputs, ASCII only, one group per line, with no
//!   widths that depend on terminal size, so a diff of two renderings is a diff of plans.
//! - Every in-plan edge appears exactly once: as a fused edge of the group it lives in, as a
//!   written output of the group that stores it, or as a plan input.

use std::fmt::Write as _;

use crate::fuser::{EdgeState, FusionPlan};
use crate::ir::Circuit;
use crate::rules::Numerics;

/// 2026-09-28: `key: value` lines printed under the title, in order (recipe, checkpoint,
/// policy). The caller decides what they say; the renderer only prints them.
pub type Header = Vec<(String, String)>;

/// 2026-09-28: Render `plan` of `circuit`.
pub fn render(circuit: &Circuit, plan: &FusionPlan, header: &Header) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "# circuit plan: {} {} n={}",
        plan.arch,
        plan.mode.name(),
        plan.rows
    );
    for (k, v) in header {
        let _ = writeln!(s, "# {k}: {v}");
    }
    let fused = plan
        .edge_states
        .iter()
        .filter(|s| matches!(s, Some(EdgeState::Fused(_))))
        .count();
    let materialized = plan
        .edge_states
        .iter()
        .filter(|s| **s == Some(EdgeState::Materialized))
        .count();
    let _ = writeln!(s, "digest: {}", plan.digest);
    let _ = writeln!(
        s,
        "groups: {}  launches: {}  edges: {} fused, {} materialized",
        plan.groups.len(),
        plan.launches(),
        fused,
        materialized
    );
    let inputs: Vec<String> = circuit
        .edges
        .iter()
        .enumerate()
        .filter(|(e, edge)| {
            plan.edge_states[*e].is_some()
                && edge
                    .producer
                    .is_none_or(|p| !plan.groups.iter().any(|g| g.nodes.contains(&p)))
        })
        .map(|(e, edge)| format!("{} {}", edge.id, plan.edge_formats[e]))
        .collect();
    if !inputs.is_empty() {
        let _ = writeln!(s, "inputs: {}", inputs.join(", "));
    }
    let mut last_layer: Option<Option<usize>> = None;
    for (g, grp) in plan.groups.iter().enumerate() {
        let first = &circuit.nodes[grp.nodes[0]];
        if last_layer != Some(first.layer) {
            let title = match first.layer {
                Some(i) => format!("layer {i} ({})", circuit.layer_kinds[i].name()),
                None => first.block.clone(),
            };
            let _ = writeln!(s, "== {title}");
            last_layer = Some(first.layer);
        }
        let _ = writeln!(s, "{}", group_line(circuit, plan, g));
    }
    s
}

fn group_line(circuit: &Circuit, plan: &FusionPlan, g: usize) -> String {
    let grp = &plan.groups[g];
    let kernels = if grp.kernels.is_empty() {
        "(copy)".to_string()
    } else {
        grp.kernels
            .iter()
            .map(|k| k.to_string())
            .collect::<Vec<_>>()
            .join(" + ")
    };
    let repeat = match grp.repeat.count(plan.rows) {
        1 => String::new(),
        n => format!(" x{n}"),
    };
    let numerics = match &grp.numerics {
        Numerics::Reference => "reference".to_string(),
        Numerics::BitIdentical { microtest } => format!("bit_identical({microtest})"),
        Numerics::Differs { lever } => format!("differs({lever})"),
    };
    let block = circuit.nodes[grp.nodes[0]]
        .id
        .rsplit_once('.')
        .map_or("", |(b, _)| b);
    let members: Vec<&str> = grp
        .nodes
        .iter()
        .map(|&n| circuit.nodes[n].local.as_str())
        .collect();
    let mut line = format!(
        "g{g:04} {block} [{}] {kernels}{repeat} {numerics} rule={}",
        members.join(","),
        grp.rule
    );
    let mut fused = Vec::new();
    let mut writes = Vec::new();
    for &n in &grp.nodes {
        for &e in &circuit.nodes[n].outputs {
            let edge = &circuit.edges[e];
            let local = edge
                .id
                .rsplit_once('.')
                .map_or(edge.id.as_str(), |(_, l)| l);
            match plan.edge_states[e] {
                Some(EdgeState::Fused(_)) => fused.push(local.to_string()),
                Some(EdgeState::Materialized) => {
                    writes.push(format!("{local}:{}", plan.edge_formats[e]))
                }
                None => {}
            }
        }
    }
    if !fused.is_empty() {
        let _ = write!(line, " fuses={{{}}}", fused.join(","));
    }
    if !writes.is_empty() {
        let _ = write!(line, " writes={{{}}}", writes.join(","));
    }
    line
}
