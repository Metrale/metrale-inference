// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: `met circuit plan --format explain`: the kernel-math table of one plan arm as
//! JSON, for exporters that read the planner (the store's preview converter) rather than a
//! signed `.kbp`. The table is built by `metrale_circuit::explain`, as the blueprint graph
//! export builds it, so both carry the same text.
//!
//! Owner: met CLI (circuit).
//! Invariants: deterministic (sorted maps); a family the plan launches that the manifest lacks is
//! an error, as in the graph export.

use std::collections::BTreeMap;

use anyhow::{Result, anyhow};
use metrale_circuit::Circuit;
use metrale_circuit::explain::{bytes_per_token, kernel_math, rows_per_token};
use metrale_circuit::hardware::OnePlan;

/// 2026-10-22: Per-tensor sizes of the arm, by circuit edge id.
#[derive(serde::Serialize)]
struct Tensor {
    format: String,
    dim_value: u64,
    rows_per_token: Option<u64>,
    bytes_per_token: Option<u64>,
}

#[derive(serde::Serialize)]
struct Doc {
    schema: &'static str,
    #[serde(flatten)]
    math: metrale_circuit::explain::KernelMath,
    tensors: BTreeMap<String, Tensor>,
}

/// 2026-10-22: The JSON text (one line per document, a trailing newline).
pub(crate) fn explain_json(c: &Circuit, one: &OnePlan) -> Result<String> {
    let manifest = &one.resolved.families;
    let plan = &one.planned.plan;
    let named: Vec<&str> = plan
        .groups
        .iter()
        .flat_map(|g| g.kernels.iter())
        .filter_map(|k| manifest.families.iter().find(|f| f.kernels.contains(k)))
        .map(|f| f.id.as_str())
        .collect();
    let formats = plan
        .edge_formats
        .iter()
        .copied()
        .chain(c.nodes.iter().filter_map(|n| n.weight));
    let math = kernel_math(manifest, named, formats).map_err(|e| anyhow!(e))?;
    let mut tensors = BTreeMap::new();
    for (i, e) in c.edges.iter().enumerate() {
        let stored = plan.edge_formats.get(i).copied().unwrap_or(e.format);
        tensors.insert(
            e.id.clone(),
            Tensor {
                format: stored.name(),
                dim_value: e.dim_value,
                rows_per_token: rows_per_token(c, e),
                bytes_per_token: bytes_per_token(c, e, stored),
            },
        );
    }
    let doc = Doc {
        schema: "kernel-math/v1",
        math,
        tensors,
    };
    Ok(serde_json::to_string(&doc)? + "\n")
}
