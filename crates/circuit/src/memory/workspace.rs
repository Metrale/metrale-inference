// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The workspaces of a plan: every kernel family a plan group runs on
//! ([`Families::of_group`]) with the workspaces it declares (`[[family.workspace]]`), evaluated at
//! the plan's rows. One buffer serves every launch of a family, so each (family, workspace) is
//! one term, attributed to every node whose group runs that family.
//!
//! Owner: metrale-circuit (memory).
//! Invariants:
//! - An expression reads the circuit's dims, `n` (the plan's rows), `k` (the node's input width)
//!   and `sm_count` (the device's); any other name is an error, never 0. The term is the largest
//!   value over the family's nodes.

use std::collections::BTreeMap;

use super::MemoryError;
use crate::fuser::FusionPlan;
use crate::ir::{Circuit, NodeIdx};
use crate::venn::families::Families;

/// 2026-10-02: One workspace of one family in a plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceTerm {
    /// 2026-10-02: Family id.
    pub family: String,
    /// 2026-10-02: Workspace name.
    pub name: String,
    /// 2026-10-02: Bytes at the plan's rows.
    pub bytes: u64,
    /// 2026-10-02: Held inside the legacy buffer arena.
    pub arena: bool,
    /// 2026-10-02: The nodes whose groups run the family.
    pub nodes: Vec<NodeIdx>,
}

/// 2026-10-02: One term per (family, workspace) over several plans' terms: the largest bytes,
/// the nodes of the plan that needed them.
pub(crate) fn widest(plans: Vec<Vec<WorkspaceTerm>>) -> Vec<WorkspaceTerm> {
    let mut by: BTreeMap<(String, String), WorkspaceTerm> = BTreeMap::new();
    for t in plans.into_iter().flatten() {
        let key = (t.family.clone(), t.name.clone());
        match by.get(&key) {
            Some(old) if old.bytes >= t.bytes => {}
            _ => {
                by.insert(key, t);
            }
        }
    }
    by.into_values().collect()
}

/// 2026-10-02: The workspaces of `plan` at `rows` rows on a device of `sm_count` SMs.
pub(crate) fn workspace_terms(
    c: &Circuit,
    plan: &FusionPlan,
    fams: &Families,
    rows: u64,
    sm_count: u64,
) -> Result<Vec<WorkspaceTerm>, MemoryError> {
    let mut dims = c.dims.clone();
    dims.insert("n".into(), rows);
    dims.insert("sm_count".into(), sm_count);
    let mut terms: BTreeMap<(String, String), WorkspaceTerm> = BTreeMap::new();
    for g in &plan.groups {
        for &n in &g.nodes {
            let node = &c.nodes[n];
            let Some(f) = fams.of_group(&g.kernels, &g.emitter, &node.op) else {
                continue;
            };
            dims.insert(
                "k".into(),
                node.inputs.first().map_or(0, |&e| c.edges[e].dim_value),
            );
            for w in &f.workspace {
                let mut bytes = 0;
                for e in &w.bytes {
                    bytes = bytes.max(e.eval(&dims).map_err(|e| MemoryError::Workspace {
                        family: f.id.clone(),
                        name: w.name.clone(),
                        detail: e.to_string(),
                    })?);
                }
                let t = terms
                    .entry((f.id.clone(), w.name.clone()))
                    .or_insert_with(|| WorkspaceTerm {
                        family: f.id.clone(),
                        name: w.name.clone(),
                        bytes: 0,
                        arena: w.arena,
                        nodes: Vec::new(),
                    });
                t.bytes = t.bytes.max(bytes);
                t.nodes.push(n);
            }
        }
    }
    Ok(terms.into_values().collect())
}
