// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Fuse a circuit that its rules cover only in part: every node no rule covers gets a
//! placeholder group (`emitter = "novel"`, rule id `novel.<op>`), and a rule that matches a node
//! whose edge format it cannot read is left out of the plan and listed. Split out of
//! `hardware/plan.rs` so `met circuit plan` and the Venn (a compared model that is not golden)
//! plan partial coverage the same way.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - A placeholder has the lowest priority, so it never displaces a real rule, and no kernels:
//!   it is never a kernel the plan's caller could run.
//! - Each pass either adds one placeholder or leaves out one rule, so the loop ends; a node
//!   still uncovered under its own placeholder is an error, never a silent gap.

use std::collections::{BTreeMap, BTreeSet};

use crate::format::Format;
use crate::fuser::{AvailableKernels, FuseError, FusionPlan, Policy, fuse};
use crate::ir::{Circuit, OpKind};
use crate::rules::{Mode, Numerics, PatternOp, Repeat, Rule};

/// 2026-09-30: The emitter of a placeholder group.
pub const NOVEL_EMITTER: &str = "novel";

/// 2026-10-10: Why no plan, not even a partial one, was produced.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CoverError {
    /// 2026-10-10: The fuser failed for a reason a placeholder cannot answer.
    #[error(transparent)]
    Fuse(#[from] FuseError),
    /// 2026-09-30: The fuser named an uncovered node the circuit does not have.
    #[error("uncovered node `{0}` not found")]
    Missing(String),
    /// 2026-09-30: A node stays uncovered under its own placeholder.
    #[error("node `{node}` stays uncovered with its placeholder `{placeholder}`")]
    Stuck {
        /// 2026-09-30: Node id.
        node: String,
        /// 2026-09-30: Placeholder rule id.
        placeholder: String,
    },
}

/// 2026-09-30: The placeholder for `op` reading `input`: a quantized read is stated, since the
/// fuser matches a pattern's quantized input format exactly (`fuser_match.rs`).
pub fn placeholder(op: &OpKind, input: Option<Format>) -> Rule {
    let input = input.filter(|f| !f.is_plain());
    Rule {
        id: match input {
            Some(f) => format!("novel.{}.{}", op.name(), f.name()),
            None => format!("novel.{}", op.name()),
        },
        pattern: vec![PatternOp {
            op: *op,
            roles: BTreeSet::new(),
            layer_kind: None,
            local: None,
            weight: None,
            input,
            writes: None,
            keep: false,
            stored: false,
            sibling: false,
            holds: None,
            steps: BTreeMap::new(),
            params: BTreeMap::new(),
        }],
        kernels: Vec::new(),
        repeat: Repeat::Once,
        copies: None,
        emitter: NOVEL_EMITTER.to_string(),
        rows: (1, u64::MAX),
        modes: Mode::ALL.into_iter().collect(),
        requires: BTreeSet::new(),
        when: Default::default(),
        numerics: Numerics::Reference,
        runs: Vec::new(),
        priority: i64::MIN,
        cite: "no rule of this class covers the op on this device".into(),
    }
}

/// 2026-10-10: Fuse `circuit` at `mode` and `rows` with `rules`, adding a placeholder to `rules`
/// for each node none covers and removing each rule that reads an edge in a format its producer
/// does not store (appended to `refused` with the reason). On return `rules` holds the rule set
/// the plan was fused with, so a caller that refuses more rules can fuse again from it.
pub fn fuse_covering(
    circuit: &Circuit,
    rules: &mut Vec<Rule>,
    available: &AvailableKernels,
    policy: &Policy,
    mode: Mode,
    rows: u64,
    refused: &mut Vec<(String, String)>,
) -> Result<FusionPlan, CoverError> {
    loop {
        match fuse(circuit, rules, available, policy, mode, rows) {
            Ok(plan) => return Ok(plan),
            Err(FuseError::Uncovered { node, .. }) => {
                let idx = circuit
                    .node(&node)
                    .ok_or_else(|| CoverError::Missing(node.clone()))?;
                let n = &circuit.nodes[idx];
                let p = placeholder(&n.op, n.inputs.first().map(|&e| circuit.edges[e].format));
                if rules.iter().any(|x| x.id == p.id) {
                    return Err(CoverError::Stuck {
                        node,
                        placeholder: p.id,
                    });
                }
                rules.push(p);
            }
            Err(e @ FuseError::FormatConflict { .. }) => {
                let FuseError::FormatConflict { rule, .. } = &e else {
                    unreachable!("matched above")
                };
                let id = rule.clone();
                rules.retain(|x| x.id != id);
                refused.push((id, e.to_string()));
            }
            Err(e) => return Err(e.into()),
        }
    }
}
