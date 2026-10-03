// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The "what we need" view (`met circuit precision`): the required pipeline of every
//! planned node whose id matches a glob, with the rule and kernels that run it on the device.
//!
//! Owner: metrale-circuit (pipeline).
//! Invariants:
//! - Pure: the model and its plan arrive as values; the text is a function of them.
//! - Every line is a node of the primary plan, in execution order; a node no kernel of the
//!   class covers still has its requirement, marked as a gap.

use std::fmt::Write as _;

use crate::hardware::plan::NOVEL_EMITTER;
use crate::hardware::{ModelUnderPlan, OnePlan};

/// 2026-10-02: The required pipelines of the nodes of `one` (a plan of `model`) whose ids match
/// `pattern` (`*` matches any run, dots included; `None` matches every node). An error when none
/// matches.
pub fn precision_text(
    model: &ModelUnderPlan,
    one: &OnePlan,
    pattern: Option<&str>,
) -> Result<String, String> {
    let c = &model.circuit;
    let plan = &one.planned.plan;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "# required node pipelines: {} on {} ({}), {} n={}",
        model.label,
        one.resolved.device.id,
        one.resolved.device.class,
        plan.mode.name(),
        plan.rows
    );
    let _ = writeln!(s, "# precision: {}", model.precision);
    for (k, v) in &one.header {
        if k == "settings" || k == "checkpoint" {
            let _ = writeln!(s, "# {k}: {v}");
        }
    }
    let _ = writeln!(
        s,
        "# node op: in -> [steps] -> out; (rule) marks what the routing states over the reference"
    );
    let mut matched = 0usize;
    for g in &plan.groups {
        let mut shown = false;
        for &n in &g.nodes {
            let node = &c.nodes[n];
            if pattern.is_some_and(|p| !crate::precision::glob(p, &node.id)) {
                continue;
            }
            let Some(line) = one.planned.pipelines.line(n) else {
                continue;
            };
            matched += 1;
            // 2026-10-02: A group's kernels once, at its first listed node; distinct, in launch
            // order.
            let ran = if g.emitter == NOVEL_EMITTER {
                format!("gap: no kernel of {} covers it", one.resolved.device.class)
            } else if shown {
                format!("{}: the launch group above", g.rule)
            } else if g.kernels.is_empty() {
                format!("{}: host ({})", g.rule, g.emitter)
            } else {
                let mut k: Vec<String> = Vec::new();
                for id in &g.kernels {
                    let id = id.to_string();
                    if !k.contains(&id) {
                        k.push(id);
                    }
                }
                format!("{}: {}", g.rule, k.join(" + "))
            };
            shown = true;
            let _ = writeln!(s, "{} {}: {line}\n    <- {ran}", node.id, node.op.name());
        }
    }
    if matched == 0 {
        return Err(format!(
            "no node of the {} n={} plan matches `{}`",
            plan.mode.name(),
            plan.rows,
            pattern.unwrap_or("*")
        ));
    }
    Ok(s)
}
