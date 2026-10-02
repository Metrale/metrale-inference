// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: `met circuit memory`'s output: a per-node table (one row per block-template node,
//! summed over the layers that instantiate it, or every node with `per_node`), a per-state and
//! per-cache table, the totals against the budget and the inverse queries; and the same report
//! as JSON, every node and every state term listed.
//!
//! Owner: metrale-circuit (memory).
//! Invariants: deterministic text; MiB are 2^20 bytes.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde_json::{Value, json};

use super::MemoryReport;
use crate::ir::Circuit;

/// 2026-10-02: The answers to the inverse queries, as the caller computed them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inverse {
    /// 2026-10-02: `(isl, osl, most sequences that fit)`; `None` when one does not fit.
    pub max_concurrency: Option<(u64, u64, Option<u64>)>,
    /// 2026-10-02: `(concurrency, osl, longest prompt that fits)`.
    pub max_isl: Option<(u64, u64, Option<u64>)>,
}

fn mib(b: u64) -> String {
    format!("{:.1}", b as f64 / (1u64 << 20) as f64)
}

/// 2026-10-02: Key of a node's row in the folded table: the block template and the node's id
/// inside it, with its section.
fn fold_key(c: &Circuit, i: usize, per_node: bool) -> String {
    let n = &c.nodes[i];
    if per_node || n.layer.is_none() {
        return n.id.clone();
    }
    format!("{}.{}", n.block, n.local)
}

/// 2026-10-02: Key of a state's row in the folded table: its block template and local id,
/// prefixed `draft.` in the draft head.
fn state_key(d: &crate::state::StateDecl) -> String {
    match d.section {
        crate::ir::Section::Main => format!("{}.{}", d.block, d.local),
        crate::ir::Section::Draft => format!("draft.{}.{}", d.block, d.local),
    }
}

/// 2026-10-02: The report as text. `head` are `(key, value)` header lines.
pub fn render_text(
    c: &Circuit,
    r: &MemoryReport,
    head: &[(String, String)],
    inverse: &Inverse,
    per_node: bool,
) -> String {
    let mut s = String::from("# Circuit memory\n\n");
    for (k, v) in head {
        let _ = writeln!(s, "{k}: {v}");
    }
    let mut rows: BTreeMap<String, (usize, [u64; 5], String)> = BTreeMap::new();
    let mut order = Vec::new();
    for m in &r.nodes {
        let v = [m.stored, m.derived, m.activations, m.workspace, m.state];
        if v.iter().all(|&x| x == 0) {
            continue;
        }
        let key = fold_key(c, m.node, per_node);
        let e = rows.entry(key.clone()).or_insert_with(|| {
            order.push(key.clone());
            (0, [0; 5], c.nodes[m.node].op.name())
        });
        e.0 += 1;
        for (j, b) in v.into_iter().enumerate() {
            // 2026-10-02: A family's workspace (column 3) is one buffer: never summed.
            e.1[j] = if j == 3 { e.1[j].max(b) } else { e.1[j] + b };
        }
    }
    let _ = writeln!(
        s,
        "\n## Per node (MiB; x = nodes folded)\n\n| node | op | x | stored | derived | activations | workspace | state |\n|---|---|---:|---:|---:|---:|---:|---:|"
    );
    for k in &order {
        let (n, v, op) = &rows[k];
        let _ = writeln!(
            s,
            "| {k} | {op} | {n} | {} | {} | {} | {} | {} |",
            mib(v[0]),
            mib(v[1]),
            mib(v[2]),
            mib(v[3]),
            mib(v[4])
        );
    }
    let _ = writeln!(
        s,
        "\n## States and caches (MiB; x = declarations folded)\n\n| state | kind | holding | lifetime | dtype | x | units each | MiB |\n|---|---|---|---|---|---:|---:|---:|"
    );
    let decl = |id: &str| c.states.iter().find(|d| d.id == id);
    let mut srows: BTreeMap<(String, String, String, String, String), (usize, u64, u64)> =
        BTreeMap::new();
    for t in &r.states.terms {
        let Some(d) = decl(&t.state) else { continue };
        let key = (
            state_key(d),
            d.kind.name().to_string(),
            format!("{:?}", t.holding).to_lowercase(),
            t.lifetime.name().to_string(),
            t.dtype.name().to_string(),
        );
        let e = srows.entry(key).or_insert((0, t.units, 0));
        e.0 += 1;
        e.2 += t.bytes;
    }
    for t in &r.caches {
        let Some(d) = decl(&t.state) else { continue };
        let key = (
            state_key(d),
            d.kind.name().to_string(),
            if t.host { "host" } else { "cache" }.to_string(),
            d.lifetime.name().to_string(),
            t.dtype.name().to_string(),
        );
        let e = srows.entry(key).or_insert((0, t.units, 0));
        e.0 += 1;
        e.2 += t.bytes;
    }
    for ((id, kind, holding, life, dt), (n, units, bytes)) in &srows {
        if *bytes == 0 {
            continue;
        }
        let _ = writeln!(
            s,
            "| {id} | {kind} | {holding} | {life} | {dt} | {n} | {units} | {} |",
            mib(*bytes)
        );
    }
    if !r.runs.is_empty() {
        let _ = writeln!(
            s,
            "\n## Activation runs (planner)\n\n| run | rows | arena MiB | materialized MiB |\n|---|---:|---:|---:|"
        );
        for a in &r.runs {
            let _ = writeln!(
                s,
                "| {} | {} | {} | {} |",
                a.label,
                a.rows,
                mib(a.arena),
                mib(a.materialized)
            );
        }
    }
    if !r.workspaces.is_empty() {
        let _ = writeln!(
            s,
            "\n## Workspaces (widest run)\n\n| family | workspace | in legacy arena | nodes | MiB |\n|---|---|---|---:|---:|"
        );
        for w in &r.workspaces {
            let _ = writeln!(
                s,
                "| {} | {} | {} | {} | {} |",
                w.family,
                w.name,
                if w.arena { "yes" } else { "no" },
                w.nodes.len(),
                mib(w.bytes)
            );
        }
    }
    let t = &r.totals;
    let _ = writeln!(s, "\n## Totals (MiB)\n\n| term | MiB |\n|---|---:|");
    for (k, v) in [
        ("weights stored (declared formats)", t.weights_stored),
        ("  of which outside the circuit", t.weights_outside),
        ("weights derived (load-time copies)", t.weights_derived),
        ("  of which leaked", t.weights_leaked),
        ("activations planned (circuit peak)", t.activations_planned),
        ("activations charged", t.activations),
        ("workspace charged", t.workspace),
        ("states (recurrent pool, KV)", t.states),
        ("caches (device)", t.caches),
        ("driver", t.driver),
        ("device total", t.device),
        ("budget", r.budget_bytes),
        ("host caches", t.host),
    ] {
        let _ = writeln!(s, "| {k} | {} |", mib(v));
    }
    let h = r.headroom();
    let _ = writeln!(
        s,
        "\nHeadroom: {:.1} MiB ({}).",
        h as f64 / (1u64 << 20) as f64,
        if h >= 0 { "fits" } else { "over budget" }
    );
    let ans = |v: &Option<u64>| v.map_or_else(|| "none fits".to_string(), |x| x.to_string());
    if let Some((isl, osl, c)) = &inverse.max_concurrency {
        let _ = writeln!(s, "Max concurrency at ISL={isl} OSL={osl}: {}.", ans(c));
    }
    if let Some((conc, osl, isl)) = &inverse.max_isl {
        let _ = writeln!(s, "Max ISL at C={conc} OSL={osl}: {}.", ans(isl));
    }
    s
}

/// 2026-10-02: The report as JSON: every node with memory, every state and cache term.
pub fn render_json(
    c: &Circuit,
    r: &MemoryReport,
    head: &[(String, String)],
    inverse: &Inverse,
) -> Value {
    let nodes: Vec<Value> = r
        .nodes
        .iter()
        .filter(|m| m.stored + m.derived + m.activations + m.workspace + m.state > 0)
        .map(|m| {
            let n = &c.nodes[m.node];
            let derived: Vec<Value> = r
                .weights
                .iter()
                .find(|w| w.node == m.node)
                .map(|w| {
                    w.derived
                        .iter()
                        .map(|d| json!({"rule": d.rule, "bytes": d.bytes, "leaked": d.leaked}))
                        .collect()
                })
                .unwrap_or_default();
            json!({
                "node": n.id, "op": n.op.name(), "layer": n.layer,
                "stored": m.stored, "derived": derived, "activations": m.activations,
                "workspace": m.workspace, "state": m.state,
            })
        })
        .collect();
    let states: Vec<Value> = r
        .states
        .terms
        .iter()
        .map(|t| {
            json!({"state": t.state, "holding": format!("{:?}", t.holding).to_lowercase(),
                   "lifetime": t.lifetime.name(), "dtype": t.dtype.name(),
                   "units": t.units, "bytes": t.bytes})
        })
        .chain(r.caches.iter().map(|t| {
            json!({"state": t.state, "kind": t.kind.name(), "host": t.host,
                   "dtype": t.dtype.name(), "units": t.units, "bytes": t.bytes})
        }))
        .collect();
    let t = &r.totals;
    json!({
        "schema": 1,
        "header": head.iter().map(|(k, v)| (k.clone(), Value::String(v.clone())))
            .collect::<serde_json::Map<_, _>>(),
        "nodes": nodes,
        "states": states,
        "runs": r.runs.iter().map(|a| json!({"label": a.label, "rows": a.rows,
            "arena": a.arena, "materialized": a.materialized})).collect::<Vec<_>>(),
        "workspaces": r.workspaces.iter().map(|w| json!({"family": w.family, "name": w.name,
            "bytes": w.bytes, "arena": w.arena, "nodes": w.nodes.len()})).collect::<Vec<_>>(),
        "totals": {
            "weights_stored": t.weights_stored, "weights_outside": t.weights_outside,
            "weights_derived": t.weights_derived,
            "weights_leaked": t.weights_leaked, "activations_planned": t.activations_planned,
            "activations": t.activations, "workspace": t.workspace, "states": t.states,
            "caches": t.caches, "host": t.host, "driver": t.driver, "device": t.device,
            "budget": r.budget_bytes, "headroom": r.headroom().to_string(),
        },
        "inverse": {
            "max_concurrency": inverse.max_concurrency.map(|(i, o, c)|
                json!({"isl": i, "osl": o, "concurrency": c})),
            "max_isl": inverse.max_isl.map(|(c, o, i)|
                json!({"concurrency": c, "osl": o, "isl": i})),
        },
    })
}
