// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: Weight bytes per node: what the checkpoint stores for it (its declared formats,
//! scales included) and every load-time copy the loader derives from it ([`super::copies`]).
//!
//! Owner: metrale-circuit (memory).
//! Invariants:
//! - A weight is `out x k`: `k` the node's first input edge's dim, `out` the sum of its output
//!   edges' dims. Routed experts hold `experts` such weights, whatever the rows routed to them.
//! - The embedding table is `vocab x hidden` at its output edge's format, counted once: the draft
//!   head reads the main table.
//! - Stored bytes are the declared circuit's (the checkpoint's own formats); served bytes appear
//!   only through copy rules, since the loader keeps every stored tensor resident.
//! - A tensor is stored once: a node whose binding another node already holds stores nothing,
//!   though copies may still derive from it. The draft head's `lm_head` reads the target's table
//!   (no checkpoint ships its own; Qwen's circuit binds the runtime copy as `mtp.lm_head`), so it
//!   stores nothing either.

use std::collections::BTreeMap;

use super::MemoryError;
use super::copies::CopyRule;
use crate::format::Format;
use crate::ir::{Circuit, Node, NodeIdx, OpKind, Section};

/// 2026-10-02: One derived copy of a node's weight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedCopy {
    /// 2026-10-02: The rule's id.
    pub rule: String,
    /// 2026-10-02: Bytes (all copies, every expert).
    pub bytes: u64,
    /// 2026-10-02: Nothing reads it.
    pub leaked: bool,
}

/// 2026-10-02: One node's weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeWeights {
    /// 2026-10-02: The node (in the served circuit).
    pub node: NodeIdx,
    /// 2026-10-02: Stored bytes at the declared formats.
    pub stored: u64,
    /// 2026-10-02: Derived copies.
    pub derived: Vec<DerivedCopy>,
}

/// 2026-10-02: Which section each node of `c` belongs to.
pub(crate) fn sections(c: &Circuit) -> Vec<Section> {
    let mut v = vec![Section::Main; c.nodes.len()];
    for b in &c.blocks {
        v[b.first..b.end].iter_mut().for_each(|s| *s = b.section);
    }
    v
}

/// 2026-10-02: `(out, k)` of a weight-reading node.
pub(crate) fn weight_shape(c: &Circuit, n: &Node) -> (u64, u64) {
    let k = n.inputs.first().map_or(0, |&e| c.edges[e].dim_value);
    let out = n.outputs.iter().map(|&e| c.edges[e].dim_value).sum();
    (out, k)
}

fn copies_per_node(c: &Circuit, n: &Node) -> Result<u64, MemoryError> {
    match n.op {
        OpKind::ExpertGateUp | OpKind::ExpertDown => {
            c.dims
                .get("experts")
                .copied()
                .ok_or_else(|| MemoryError::MissingDim {
                    node: n.id.clone(),
                    dim: "experts".into(),
                })
        }
        _ => Ok(1),
    }
}

fn bytes_of(n: &Node, what: &str, v: Option<u64>) -> Result<u64, MemoryError> {
    v.ok_or_else(|| MemoryError::Size {
        node: n.id.clone(),
        what: what.to_string(),
    })
}

/// 2026-10-02: Every weight of `served` (its embedding and every weight-reading node), stored at
/// the formats `declared` gives the node of the same id, with the copies `rules` derive under
/// `settings`.
pub fn node_weights(
    served: &Circuit,
    declared: &Circuit,
    rules: &[CopyRule],
    settings: &BTreeMap<String, String>,
) -> Result<Vec<NodeWeights>, MemoryError> {
    let secs = sections(served);
    let declared_of: BTreeMap<&str, &Node> =
        declared.nodes.iter().map(|n| (n.id.as_str(), n)).collect();
    let mut out = Vec::new();
    let mut bound: std::collections::BTreeSet<&[String]> = std::collections::BTreeSet::new();
    for (i, n) in served.nodes.iter().enumerate() {
        if n.op == OpKind::Embed {
            if secs[i] != Section::Main {
                continue;
            }
            let vocab = *served
                .dims
                .get("vocab")
                .ok_or_else(|| MemoryError::MissingDim {
                    node: n.id.clone(),
                    dim: "vocab".into(),
                })?;
            let e = n.outputs.first().map(|&e| &served.edges[e]);
            let table = e.and_then(|e| e.format.bytes(vocab, e.dim_value));
            out.push(NodeWeights {
                node: i,
                stored: bytes_of(n, "embedding table", table)?,
                derived: Vec::new(),
            });
            continue;
        }
        let Some(serve_fmt) = n.weight else { continue };
        let stored_fmt: Format = declared_of
            .get(n.id.as_str())
            .and_then(|d| d.weight)
            .unwrap_or(serve_fmt);
        let (o, k) = weight_shape(served, n);
        let copies = copies_per_node(served, n)?;
        let one = bytes_of(n, "stored weight", stored_fmt.weight_bytes(o, k))?;
        let draft_head = secs[i] == Section::Draft && n.op == OpKind::LmHead;
        let first = !draft_head && (n.binding.is_empty() || bound.insert(n.binding.as_slice()));
        let stored = match first {
            true => bytes_of(n, "stored weight", one.checked_mul(copies))?,
            false => 0,
        };
        let mut derived = Vec::new();
        for r in rules.iter().filter(|r| {
            r.matches(
                &served.arch,
                &n.op,
                secs[i],
                stored_fmt,
                serve_fmt,
                settings,
            )
        }) {
            let b = r
                .law
                .bytes(o, k)
                .and_then(|b| b.checked_mul(copies))
                .and_then(|b| b.checked_mul(r.count));
            derived.push(DerivedCopy {
                rule: r.id.clone(),
                bytes: bytes_of(n, &format!("copy `{}`", r.id), b)?,
                leaked: r.leaked,
            });
        }
        out.push(NodeWeights {
            node: i,
            stored,
            derived,
        });
    }
    Ok(out)
}
