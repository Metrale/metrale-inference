// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: `met circuit memory --json` as typed records. A record's fields serialize in
//! declaration order, so the document's key order is the same whether or not serde_json keeps
//! insertion order (`preserve_order` needs `std`; this crate is `no_std`).
//!
//! Owner: metrale-circuit (memory).
//! Invariants:
//! - Field order is the published order: `schema`, `header`, `nodes`, `states`, `runs`,
//!   `workspaces`, `totals`, `inverse`, each record as written below.
//! - `header` keeps the caller's order; a repeated key keeps its first place and last value.

use crate::prelude::*;

use serde::ser::{Serialize, SerializeMap, Serializer};

use super::MemoryReport;
use super::render::Inverse;
use crate::ir::Circuit;

/// 2026-10-22: The whole document.
#[derive(Debug, serde::Serialize)]
pub struct MemoryJson<'a> {
    schema: u32,
    header: Header<'a>,
    nodes: Vec<NodeJson<'a>>,
    states: Vec<StateJson<'a>>,
    runs: Vec<RunJson<'a>>,
    workspaces: Vec<WorkspaceJson<'a>>,
    totals: TotalsJson,
    inverse: InverseJson,
}

/// 2026-10-22: The header lines as one object, in the caller's order.
#[derive(Debug)]
struct Header<'a>(Vec<(&'a str, &'a str)>);

impl Serialize for Header<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(Some(self.0.len()))?;
        for (k, v) in &self.0 {
            m.serialize_entry(k, v)?;
        }
        m.end()
    }
}

fn header<'a>(head: &'a [(String, String)]) -> Header<'a> {
    let mut out: Vec<(&str, &str)> = Vec::with_capacity(head.len());
    for (k, v) in head {
        match out.iter_mut().find(|(have, _)| *have == k.as_str()) {
            Some(slot) => slot.1 = v.as_str(),
            None => out.push((k.as_str(), v.as_str())),
        }
    }
    Header(out)
}

#[derive(Debug, serde::Serialize)]
struct DerivedJson<'a> {
    rule: &'a str,
    bytes: u64,
    leaked: bool,
}

#[derive(Debug, serde::Serialize)]
struct NodeJson<'a> {
    node: &'a str,
    op: String,
    layer: Option<usize>,
    stored: u64,
    derived: Vec<DerivedJson<'a>>,
    activations: u64,
    workspace: u64,
    state: u64,
}

#[derive(Debug, serde::Serialize)]
#[serde(untagged)]
enum StateJson<'a> {
    Term {
        state: &'a str,
        holding: String,
        lifetime: &'static str,
        dtype: &'static str,
        units: u64,
        bytes: u64,
    },
    Cache {
        state: &'a str,
        kind: &'static str,
        host: bool,
        dtype: &'static str,
        units: u64,
        bytes: u64,
    },
}

#[derive(Debug, serde::Serialize)]
struct RunJson<'a> {
    label: &'a str,
    rows: u64,
    arena: u64,
    materialized: u64,
}

#[derive(Debug, serde::Serialize)]
struct WorkspaceJson<'a> {
    family: &'a str,
    name: &'a str,
    bytes: u64,
    arena: bool,
    nodes: usize,
}

#[derive(Debug, serde::Serialize)]
struct TotalsJson {
    weights_stored: u64,
    weights_outside: u64,
    weights_derived: u64,
    weights_leaked: u64,
    activations_planned: u64,
    activations: u64,
    workspace: u64,
    states: u64,
    caches: u64,
    host: u64,
    driver: u64,
    device: u64,
    budget: u64,
    headroom: String,
}

#[derive(Debug, serde::Serialize)]
struct ConcurrencyJson {
    isl: u64,
    osl: u64,
    concurrency: Option<u64>,
}

#[derive(Debug, serde::Serialize)]
struct IslJson {
    concurrency: u64,
    osl: u64,
    isl: Option<u64>,
}

#[derive(Debug, serde::Serialize)]
struct KvPoolJson {
    blocks: u64,
    tokens: u64,
    needed_blocks: u64,
}

#[derive(Debug, serde::Serialize)]
struct InverseJson {
    max_concurrency: Option<ConcurrencyJson>,
    max_isl: Option<IslJson>,
    kv_pool: Option<KvPoolJson>,
}

/// 2026-10-02: The report as JSON: every node with memory, every state and cache term.
pub fn render_json<'a>(
    c: &'a Circuit,
    r: &'a MemoryReport,
    head: &'a [(String, String)],
    inverse: &Inverse,
) -> MemoryJson<'a> {
    let nodes = r
        .nodes
        .iter()
        .filter(|m| m.stored + m.derived + m.activations + m.workspace + m.state > 0)
        .map(|m| {
            let n = &c.nodes[m.node];
            let derived = r
                .weights
                .iter()
                .find(|w| w.node == m.node)
                .map(|w| {
                    w.derived
                        .iter()
                        .map(|d| DerivedJson {
                            rule: &d.rule,
                            bytes: d.bytes,
                            leaked: d.leaked,
                        })
                        .collect()
                })
                .unwrap_or_default();
            NodeJson {
                node: &n.id,
                op: n.op.name(),
                layer: n.layer,
                stored: m.stored,
                derived,
                activations: m.activations,
                workspace: m.workspace,
                state: m.state,
            }
        })
        .collect();
    let states = r
        .states
        .terms
        .iter()
        .map(|t| StateJson::Term {
            state: &t.state,
            holding: format!("{:?}", t.holding).to_lowercase(),
            lifetime: t.lifetime.name(),
            dtype: t.dtype.name(),
            units: t.units,
            bytes: t.bytes,
        })
        .chain(r.caches.iter().map(|t| StateJson::Cache {
            state: &t.state,
            kind: t.kind.name(),
            host: t.host,
            dtype: t.dtype.name(),
            units: t.units,
            bytes: t.bytes,
        }))
        .collect();
    let t = &r.totals;
    MemoryJson {
        schema: 1,
        header: header(head),
        nodes,
        states,
        runs: r
            .runs
            .iter()
            .map(|a| RunJson {
                label: &a.label,
                rows: a.rows,
                arena: a.arena,
                materialized: a.materialized,
            })
            .collect(),
        workspaces: r
            .workspaces
            .iter()
            .map(|w| WorkspaceJson {
                family: &w.family,
                name: &w.name,
                bytes: w.bytes,
                arena: w.arena,
                nodes: w.nodes.len(),
            })
            .collect(),
        totals: TotalsJson {
            weights_stored: t.weights_stored,
            weights_outside: t.weights_outside,
            weights_derived: t.weights_derived,
            weights_leaked: t.weights_leaked,
            activations_planned: t.activations_planned,
            activations: t.activations,
            workspace: t.workspace,
            states: t.states,
            caches: t.caches,
            host: t.host,
            driver: t.driver,
            device: t.device,
            budget: r.budget_bytes,
            headroom: r.headroom().to_string(),
        },
        inverse: InverseJson {
            max_concurrency: inverse.max_concurrency.map(|(isl, osl, concurrency)| {
                ConcurrencyJson {
                    isl,
                    osl,
                    concurrency,
                }
            }),
            max_isl: inverse.max_isl.map(|(concurrency, osl, isl)| IslJson {
                concurrency,
                osl,
                isl,
            }),
            kv_pool: inverse
                .kv_pool
                .map(|(blocks, tokens, needed_blocks)| KvPoolJson {
                    blocks,
                    tokens,
                    needed_blocks,
                }),
        },
    }
}
