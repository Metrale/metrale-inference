// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `plan_digest`: SHA-256 over a canonical serialization of a plan, the circuit's
//! edges, the whole rule set and the policy.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - The serialization is a fixed-order text with a tag and a schema number up front, one
//!   record per line, and every field labelled, so the same bytes in another field give
//!   another digest.
//! - Rules are hashed in id order, not file order: moving a rule within FUSIONS.toml does not
//!   change a plan, and does not change its digest.
//! - Every rule field is hashed except `cite`, which is documentation: a moved line number
//!   in a citation changes no plan.

use std::fmt::Write as _;

use sha2::{Digest, Sha256};

use crate::fuser::{EdgeState, FusionPlan, Policy};
use crate::ir::Circuit;
use crate::rules::Rule;

/// 2026-09-28: Bumped when the canonical form changes.
pub const DIGEST_SCHEMA: u32 = 1;

/// 2026-09-28: The canonical text the digest is taken over.
pub fn canonical(circuit: &Circuit, plan: &FusionPlan, rules: &[Rule], policy: &Policy) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "metrale-circuit-plan schema={DIGEST_SCHEMA}");
    let _ = writeln!(
        s,
        "plan arch={} mode={} rows={}",
        plan.arch,
        plan.mode.name(),
        plan.rows
    );
    for (i, g) in plan.groups.iter().enumerate() {
        let kernels: Vec<String> = g.kernels.iter().map(|k| k.to_string()).collect();
        let nodes: Vec<&str> = g
            .nodes
            .iter()
            .map(|&n| circuit.nodes[n].id.as_str())
            .collect();
        let _ = writeln!(
            s,
            "group {i} rule={} kernels=[{}] repeat={} emitter={} numerics={:?} nodes=[{}]",
            g.rule,
            kernels.join(","),
            g.repeat.name(),
            g.emitter,
            g.numerics,
            nodes.join(",")
        );
    }
    for (e, edge) in circuit.edges.iter().enumerate() {
        let Some(state) = plan.edge_states[e] else {
            continue;
        };
        let state = match state {
            EdgeState::Materialized => "M".to_string(),
            EdgeState::Fused(g) => format!("F{g}"),
        };
        let _ = writeln!(
            s,
            "edge {} stored={} rows={} dim={}={} state={state}",
            edge.id,
            plan.edge_formats[e].name(),
            edge.rows.text(),
            edge.dim.text(),
            edge.dim_value
        );
    }
    let mut sorted: Vec<&Rule> = rules.iter().collect();
    sorted.sort_by(|a, b| a.id.cmp(&b.id));
    for r in sorted {
        let _ = writeln!(
            s,
            "rule {} pattern={:?} kernels={:?} repeat={} emitter={} rows={:?} modes={:?} \
             requires={:?} when={:?} numerics={:?} priority={}",
            r.id,
            r.pattern,
            r.kernels,
            r.repeat.name(),
            r.emitter,
            r.rows,
            r.modes,
            r.requires,
            r.when,
            r.numerics,
            r.priority
        );
    }
    let levers: Vec<&str> = policy.opt_in_levers.iter().map(String::as_str).collect();
    let _ = writeln!(s, "levers [{}]", levers.join(","));
    for (k, v) in &policy.settings {
        let _ = writeln!(s, "setting {k}={v}");
    }
    s
}

/// 2026-09-28: Lower-case hex SHA-256 of [`canonical`].
pub fn plan_digest(
    circuit: &Circuit,
    plan: &FusionPlan,
    rules: &[Rule],
    policy: &Policy,
) -> String {
    let digest = Sha256::digest(canonical(circuit, plan, rules, policy).as_bytes());
    digest.iter().fold(String::with_capacity(64), |mut acc, b| {
        let _ = write!(acc, "{b:02x}");
        acc
    })
}
