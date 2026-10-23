// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The store's data contract: `kbp-card/v1` and `kbp-graph/v2`, derived from a
//! verified file by pure functions, and their JSON Schemas generated from the same types.
//!
//! Owner: blueprint.
//! Invariants: the `.kbp` stays the SSOT; a card or graph is never hand-edited, and the
//! committed `schemas/kbp-card.v1.json` and `schemas/kbp-graph.v2.json` equal [`card_schema`]
//! and [`graph_schema`] (golden test); `kbp-graph.v1.json` is kept as published.

pub mod card;
pub mod explain;
pub mod graph;
pub mod schema;

use alloc::string::String;
use alloc::vec::Vec;

use metrale_circuit::LayerKind;
use serde_json::Value;

use crate::types::Blueprint;

/// 2026-10-22: What the exporter knows beyond the file: who exports, when it verified, and
/// the file's length.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportContext {
    /// 2026-10-22: Engine version.
    pub exported_by: String,
    /// 2026-10-22: Unix ms of the verification.
    pub verified_at_ms: i64,
    /// 2026-10-22: The exported file's length.
    pub file_len: u64,
}

/// 2026-10-22: A layer kind as the exports spell it: the circuit's HF spelling, except
/// `sparse_attention` for `deepseek_sparse_attention` (the store names the mechanism, not the
/// checkpoint family).
pub fn layer_kind_spelling(k: LayerKind) -> &'static str {
    match k {
        LayerKind::SparseAttention => "sparse_attention",
        other => other.name(),
    }
}

/// 2026-10-22: `(weight, activation)` format spellings of every weight-reading node: the
/// weight format and the declared format of its first input.
pub fn linear_formats(bp: &Blueprint) -> Vec<(String, String)> {
    let c = &bp.circuit;
    c.nodes
        .iter()
        .filter_map(|n| {
            let w = n.weight?.name();
            let a = n
                .inputs
                .first()
                .and_then(|&e| c.edges.get(e))
                .map_or_else(|| String::from("-"), |e| e.format.name());
            Some((w, a))
        })
        .collect()
}

/// 2026-10-22: The `kbp-card/v1` JSON Schema.
pub fn card_schema() -> Value {
    schema::document::<card::Card>(
        "https://metrale.ai/schemas/kbp-card.v1.json",
        card::CARD_SCHEMA,
    )
}

/// 2026-10-22: The `kbp-graph/v2` JSON Schema (v1 stays committed as published, never edited).
pub fn graph_schema() -> Value {
    schema::document::<graph::Graph>(
        "https://metrale.ai/schemas/kbp-graph.v2.json",
        graph::GRAPH_SCHEMA,
    )
}
