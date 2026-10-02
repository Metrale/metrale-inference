// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The tensor-core policy over a class's whole rule set, not only the plans a report
//! runs: every rule that could cover a policy op off tensor cores, at any mode and row count the
//! rule and the policy share, needs an exemption that lists the op, those modes, those rows and
//! every kernel of the rule. A new FUSIONS.toml rule or kernel that would violate the policy at a
//! row count no golden plan or report exercises fails here.
//!
//! Owner: metrale-circuit (hardware).
//! Invariants: pure; the answer is a list of messages, empty when the rule set complies.

use std::collections::BTreeSet;

use super::{TcPolicy, unit_of_group};
use crate::format::Format;
use crate::ir::OpKind;
use crate::rules::{Mode, Rule};
use crate::venn::families::Families;

/// 2026-10-02: Whether a requirement's weights can meet a pattern element's (`None`: any).
fn weights_meet(w: &super::Weights, pattern: Option<Format>) -> bool {
    match (w, pattern) {
        (super::Weights::Any, _) | (_, None) => true,
        (super::Weights::Only(set), Some(f)) => set.contains(&f),
    }
}

/// 2026-10-02: Every rule-level violation of `policy` in `rules`, one message each.
pub fn lint_rules(rules: &[Rule], families: &Families, policy: &TcPolicy) -> Vec<String> {
    let mut out = Vec::new();
    for rule in rules.iter().filter(|r| !r.kernels.is_empty()) {
        let unit = match unit_of_group(families, &rule.kernels) {
            Ok(u) => u,
            Err(e) => {
                out.push(format!("rule `{}`: {e}", rule.id));
                continue;
            }
        };
        if unit.as_ref().is_some_and(|u| u.is_tensor_core()) {
            continue;
        }
        for p in &rule.pattern {
            let ops: Vec<OpKind> = if p.roles.is_empty() {
                vec![p.op]
            } else {
                p.roles.iter().map(|r| OpKind::Linear(*r)).collect()
            };
            for op in ops {
                for req in &policy.require {
                    let modes: BTreeSet<Mode> =
                        rule.modes.intersection(&req.modes).copied().collect();
                    let (lo, hi) = (rule.rows.0.max(req.min_rows), rule.rows.1);
                    if modes.is_empty()
                        || lo > hi
                        || !req.ops.iter().any(|o| o.matches(&op))
                        || !weights_meet(&req.weights, p.weight)
                    {
                        continue;
                    }
                    let allowed = policy.exempt.iter().any(|e| {
                        e.ops.iter().any(|o| o.matches(&op))
                            && modes.is_subset(&e.modes)
                            && e.rows.0 <= lo
                            && hi <= e.rows.1
                            && rule.kernels.iter().all(|k| e.kernels.contains(k))
                    });
                    if !allowed {
                        let modes: Vec<&str> = modes.iter().map(|m| m.name()).collect();
                        out.push(format!(
                            "rule `{}` runs {} off tensor cores at {} rows {lo}-{hi}, and no exemption lists it",
                            rule.id,
                            op.name(),
                            modes.join("/")
                        ));
                    }
                }
            }
        }
    }
    out
}
