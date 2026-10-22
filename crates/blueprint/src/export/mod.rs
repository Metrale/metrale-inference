// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The store's data contract: `kbp-card/v1` and `kbp-graph/v1`, derived from a
//! verified file by pure functions, and their JSON Schemas generated from the same types.
//!
//! Owner: blueprint.
//! Invariants: the `.kbp` stays the SSOT; a card or graph is never hand-edited, and the
//! committed `schemas/kbp-*.v1.json` equal [`card_schema`] and [`graph_schema`] (golden test).

pub mod card;
pub mod graph;
pub mod schema;

use alloc::string::String;
use alloc::vec::Vec;

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

/// 2026-10-22: `(weight, activation)` format spellings of every weight-reading node: the
/// weight format and the declared format of its first input.
pub fn linear_formats(bp: &Blueprint) -> Vec<(&str, &str)> {
    let c = &bp.circuit;
    c.nodes
        .iter()
        .filter_map(|n| {
            let w = n.weight.as_deref()?;
            let a = n
                .inputs
                .first()
                .and_then(|&e| c.edges.get(e as usize))
                .map_or("-", |e| e.format.as_str());
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

/// 2026-10-22: The `kbp-graph/v1` JSON Schema.
pub fn graph_schema() -> Value {
    schema::document::<graph::Graph>(
        "https://metrale.ai/schemas/kbp-graph.v1.json",
        graph::GRAPH_SCHEMA,
    )
}
