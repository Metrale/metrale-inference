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

use crate::fuser::{EdgeState, FusionPlan, Policy};
use crate::ir::Circuit;
use crate::rules::Numerics;
use crate::runtime::RuntimeRoute;

/// 2026-09-28: `key: value` lines printed under the title, in order (recipe, checkpoint,
/// policy). The caller decides what they say; the renderer only prints them.
pub type Header = Vec<(String, String)>;

/// 2026-09-30: `header` with its settings line rebuilt from `policy` (after a device class
/// re-reads its settings, or for a runtime route's arm).
pub fn with_settings(header: &Header, policy: &Policy) -> Header {
    header
        .iter()
        .map(|(k, v)| {
            if k == "settings" {
                let s: Vec<String> = policy
                    .settings
                    .iter()
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect();
                (k.clone(), s.join(" "))
            } else {
                (k.clone(), v.clone())
            }
        })
        .collect()
}

/// 2026-09-30: A runtime route's arm, rendered after the primary plan: a heading that names the
/// route, its condition and the settings it plans as, then the arm's plan under `header`.
pub fn route_section(
    circuit: &Circuit,
    route: &RuntimeRoute,
    plan: &FusionPlan,
    header: &Header,
) -> String {
    format!(
        "\n# runtime route `{}`: when {}; planned as `{}` ({})\n\n{}",
        route.id,
        route.why,
        route.plans_as_text(),
        route.cite,
        render(circuit, plan, header)
    )
}

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
    if let Some(t) = &plan.table {
        let _ = writeln!(s, "# row table: {t}");
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
        "groups: {}  launches: {}{}  edges: {} fused, {} materialized",
        plan.groups.len(),
        plan.launches(),
        match plan.copies() {
            0 => String::new(),
            c => format!("  copies: {c}"),
        },
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
    // 2026-09-29: A group with no kernels is work the host does outside the program: the
    // prologue's embedding copy, or sampling from the logits. 2026-09-30: except a hardware
    // plan's placeholder, which marks an op no kernel of the device's class covers.
    let kernels = if grp.emitter == crate::hardware::plan::NOVEL_EMITTER {
        "(novel: no kernel on this class)".to_string()
    } else if !grp.runs.is_empty() {
        // 2026-09-30: A per-run group: each run's launches, `!` marking a fragmented run.
        grp.runs
            .iter()
            .map(|r| {
                let l: Vec<String> = r
                    .launches
                    .iter()
                    .map(|(k, t)| match t.count(r.run) {
                        1 => k.to_string(),
                        c => format!("{k} x{c}"),
                    })
                    .collect();
                let copies = match r.copy_count() {
                    0 => String::new(),
                    c => format!(" + copy x{c}"),
                };
                format!(
                    "[{}x{}{}: {}{copies}]",
                    r.run.k,
                    r.run.n,
                    if r.run.contiguous { "" } else { "!" },
                    l.join(" + ")
                )
            })
            .collect::<Vec<_>>()
            .join(" ")
    } else if grp.kernels.is_empty() {
        "(host)".to_string()
    } else {
        grp.kernels
            .iter()
            .map(|k| k.to_string())
            .collect::<Vec<_>>()
            .join(" + ")
    };
    let repeat = match grp.repeat.count(plan.rows) {
        None | Some(1) => String::new(),
        Some(n) => format!(" x{n}"),
    };
    let repeat = match grp.copies.and_then(|c| c.count(plan.rows)) {
        Some(c) => format!("{repeat} + copy x{c}"),
        None => repeat,
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
