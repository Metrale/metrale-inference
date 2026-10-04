// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Two digests. `plan_digest` is SHA-256 over a canonical serialization of what a
//! plan runs: its groups (members, kernels, repetition, emitter, numerics; 2026-10-04: a side
//! stream and scratch regions when present), its cross-stream events, and every edge's stored
//! format and state, under the plan's arch, mode and rows. `rules_digest` is SHA-256
//! over the FUSIONS.toml text the rules were parsed from.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - The plan serialization is a fixed-order text with a tag and a schema number up front,
//!   one record per line, every field labelled, so the same bytes in another field give
//!   another digest.
//! - A plan digest moves only when the plan does: editing, adding or reordering a rule that
//!   the plan does not select leaves it unchanged. Rule ids and citations are names, not
//!   content, and are not hashed; the kernels a group runs are.
//! - The rule set is attested by `rules_digest` and by the closure hash, which covers
//!   FUSIONS.toml as a config.

use std::fmt::Write as _;

use sha2::{Digest, Sha256};

use crate::fuser::{EdgeState, FusionPlan};
use crate::ir::Circuit;

/// 2026-09-28: Bumped when the canonical form changes.
pub const DIGEST_SCHEMA: u32 = 2;

/// 2026-09-28: The canonical text the plan digest is taken over.
pub fn canonical(circuit: &Circuit, plan: &FusionPlan) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "metrale-circuit-plan schema={DIGEST_SCHEMA}");
    let _ = writeln!(
        s,
        "plan arch={} mode={} rows={}",
        plan.arch,
        plan.mode.name(),
        plan.rows
    );
    // 2026-09-30: A batched verify's row table, and each per-run group's launches below.
    if let Some(t) = &plan.table {
        let _ = writeln!(s, "table {t}");
    }
    for (i, g) in plan.groups.iter().enumerate() {
        let kernels: Vec<String> = g.kernels.iter().map(|k| k.to_string()).collect();
        let nodes: Vec<&str> = g
            .nodes
            .iter()
            .map(|&n| circuit.nodes[n].id.as_str())
            .collect();
        // 2026-10-04: A side stream and scratch regions are written only when present, so a
        // plan without them serializes as before.
        let _ = writeln!(
            s,
            "group {i} kernels=[{}] repeat={}{} emitter={} numerics={:?} nodes=[{}]{}{}",
            kernels.join(","),
            g.repeat.name(),
            g.copies
                .map(|c| format!(" copies={}", c.name()))
                .unwrap_or_default(),
            g.emitter,
            g.numerics,
            nodes.join(","),
            match g.stream {
                crate::streams::Stream::Main => String::new(),
                other => format!(" stream={}", other.name()),
            },
            if g.scratch.is_empty() {
                String::new()
            } else {
                format!(" scratch=[{}]", g.scratch.join(","))
            }
        );
        for r in &g.runs {
            let launches: Vec<String> = r
                .launches
                .iter()
                .map(|(k, t)| format!("{k}*{}", t.name()))
                .collect();
            let _ = writeln!(
                s,
                "  run k={} n={} contiguous={} launches=[{}] copies={}",
                r.run.k,
                r.run.n,
                r.run.contiguous,
                launches.join(","),
                r.copies.map_or("none", |c| c.name())
            );
        }
    }
    for ev in &plan.events {
        let _ = writeln!(s, "{}", crate::streams::event_text(ev));
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
    s
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::with_capacity(64), |mut acc, b| {
        let _ = write!(acc, "{b:02x}");
        acc
    })
}

/// 2026-09-28: Lower-case hex SHA-256 of [`canonical`].
pub fn plan_digest(circuit: &Circuit, plan: &FusionPlan) -> String {
    hex(&Sha256::digest(canonical(circuit, plan).as_bytes()))
}

/// 2026-09-28: Lower-case hex SHA-256 of the FUSIONS.toml text the rules came from.
pub fn rules_digest(fusions_text: &str) -> String {
    hex(&Sha256::digest(fusions_text.as_bytes()))
}

/// 2026-09-28: Lower-case hex SHA-256 over several plan digests, each followed by a newline, in
/// the given order: one value for a set of plans compiled together.
pub fn plans_digest<'a>(digests: impl IntoIterator<Item = &'a str>) -> String {
    let mut h = Sha256::new();
    for d in digests {
        h.update(d.as_bytes());
        h.update(b"\n");
    }
    hex(&h.finalize())
}
