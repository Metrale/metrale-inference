// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: A family's `workspace` entries into [`super::Workspace`]: device scratch one launch
//! of the family needs beyond its input and output edges (split partials, permute and gather
//! buffers, activation-quant scratch), as dim expressions over the circuit's dims, the launch's
//! rows `n`, the node's input width `k` and the device's `sm_count`. The memory model (`crate::memory`) sizes them.
//!
//! Owner: metrale-circuit (venn).
//! Invariants:
//! - `bytes` is a list of expressions whose largest value is the workspace (one buffer serves
//!   every use); a family's workspaces have distinct names.
//! - `arena = true` marks scratch the legacy buffer arena already holds
//!   (`gpu-runtime/src/buffers.rs`), so a budget over that arena does not count it twice.

use serde::Deserialize;

use super::Workspace;
use crate::dims::DimExpr;

/// 2026-10-02: One `[[family.workspace]]`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WorkspaceFile {
    name: String,
    bytes: Vec<String>,
    arena: bool,
    why: String,
}

/// 2026-10-02: The family's workspaces, or what is wrong with them.
pub(super) fn workspaces(files: Vec<WorkspaceFile>) -> Result<Vec<Workspace>, String> {
    let mut out: Vec<Workspace> = Vec::with_capacity(files.len());
    for w in files {
        if out.iter().any(|o| o.name == w.name) {
            return Err(format!("workspace `{}` is declared twice", w.name));
        }
        if w.bytes.is_empty() || w.why.trim().is_empty() {
            return Err(format!(
                "workspace `{}` needs at least one `bytes` expression and a `why`",
                w.name
            ));
        }
        let bytes = w
            .bytes
            .iter()
            .map(|b| DimExpr::parse(b).map_err(|e| format!("workspace `{}`: {e}", w.name)))
            .collect::<Result<Vec<_>, _>>()?;
        out.push(Workspace {
            name: w.name,
            bytes,
            arena: w.arena,
            why: w.why,
        });
    }
    Ok(out)
}
