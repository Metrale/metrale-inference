// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: A small circuit, precision table and rule set the unit tests share: an embedding,
//! N layers of norm -> gate_up -> silu -> down -> residual add, and a head.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use crate::fuser::{AvailableKernels, FusionPlan, Policy, fuse};
use crate::ir::{ArchShape, Circuit, LayerKind};
use crate::precision::PrecisionTable;
use crate::rules::{Mode, Rule, parse_rules};

pub const CIRCUIT: &str = r#"
schema = 1
arch = "toy"
description = "toy"
layer_module = "layers.{i}"
dims = ["hidden", "inter", "vocab"]
prologue = ["embed"]
epilogue = ["head"]
draft = []

[layout]
kind = "list"

[layout.blocks]
linear_attention = ["ffn"]
full_attention = ["ffn"]

[block.embed]
stream_out = "h"

[[block.embed.node]]
id = "embed"
op = "embed"
out = [{ edge = "h", format = "bf16", shape = "n x hidden" }]

[block.ffn]
stream_in = "x"
stream_out = "y"

[[block.ffn.node]]
id = "norm"
op = "rms_norm"
in = ["x"]
out = [{ edge = "xn", format = "bf16", shape = "n x hidden" }]

[[block.ffn.node]]
id = "up"
op = "linear"
role = "gate_up"
in = ["xn"]
out = [{ edge = "gu", format = "bf16", shape = "n x inter*2" }]
binding = ["{L}.up"]

[[block.ffn.node]]
id = "act"
op = "silu_mul"
in = ["gu"]
out = [{ edge = "a", format = "bf16", shape = "n x inter" }]

[[block.ffn.node]]
id = "down"
op = "linear"
role = "down"
in = ["a"]
out = [{ edge = "d", format = "bf16", shape = "n x hidden" }]
binding = ["{L}.down"]

[[block.ffn.node]]
id = "add"
op = "residual_add"
in = ["x", "d"]
out = [{ edge = "y", format = "bf16", shape = "n x hidden" }]

[block.head]
stream_in = "x"
outputs = ["logits"]

[[block.head.node]]
id = "final_norm"
op = "final_norm"
in = ["x"]
out = [{ edge = "xn", format = "bf16", shape = "n x hidden" }]

[[block.head.node]]
id = "lm_head"
op = "lm_head"
in = ["xn"]
out = [{ edge = "logits", format = "bf16", shape = "n x vocab" }]
binding = ["lm_head"]
"#;

pub const PRECISION: &str = r#"
schema = 1
checkpoint = "toy"
tier = "nvfp4"

[[linear]]
match = "lm_head"
weight = "bf16"
activation = "bf16"

[[linear]]
match = "*"
weight = "nvfp4/g16"
activation = "bf16"
"#;

/// 2026-09-28: One rule per op, each covering every mode and row count.
pub const BASE_RULES: &str = r#"
schema = 1
"#;

/// 2026-09-28: A single-op rule for `op` (with `extra` pattern keys) at priority 1.
pub fn single(id: &str, op: &str, extra: &str) -> String {
    format!(
        r#"
[[rule]]
id = "{id}"
pattern = [{{ op = "{op}"{extra} }}]
kernels = [{{ module = "m", func = "{id}" }}]
repeat = "once"
emitter = "{id}"
rows = [1, 128]
modes = ["decode", "multi_seq", "verify"]
numerics = "reference"
priority = 1
cite = "test"
"#
    )
}

/// 2026-09-28: The base rules plus `extra` rule text.
pub fn rules(extra: &str) -> Vec<Rule> {
    let mut text = BASE_RULES.to_string();
    for (id, op, x) in [
        ("embed", "embed", ""),
        ("norm", "rms_norm", ""),
        ("up", "linear", ", role = \"gate_up\""),
        ("act", "silu_mul", ""),
        ("down", "linear", ", role = \"down\""),
        ("add", "residual_add", ""),
        ("final_norm", "final_norm", ""),
        ("lm_head", "lm_head", ""),
    ] {
        text.push_str(&single(id, op, x));
    }
    text.push_str(extra);
    parse_rules(&text).unwrap_or_else(|e| panic!("toy rules: {e}"))
}

pub fn shape(layers: usize) -> ArchShape {
    ArchShape {
        layer_kinds: vec![LayerKind::LinearAttention; layers],
        dims: BTreeMap::from([
            ("hidden".to_string(), 64),
            ("inter".to_string(), 128),
            ("vocab".to_string(), 256),
        ]),
    }
}

pub fn circuit(layers: usize) -> Circuit {
    let table = PrecisionTable::parse(PRECISION).unwrap();
    crate::instantiate(CIRCUIT, &shape(layers), &table).unwrap_or_else(|e| panic!("toy: {e}"))
}

pub fn policy() -> Policy {
    Policy {
        opt_in_levers: Default::default(),
        settings: BTreeMap::from([("kv".to_string(), "bf16".to_string())]),
    }
}

pub fn plan(c: &Circuit, rules: &[Rule], policy: &Policy, rows: u64) -> FusionPlan {
    fuse(
        c,
        rules,
        &AvailableKernels::all_named_by(rules),
        policy,
        Mode::Decode,
        rows,
    )
    .unwrap_or_else(|e| panic!("toy plan: {e}"))
}

/// 2026-09-28: `(rule, member local ids)` per group, for readable assertions.
pub fn groups(c: &Circuit, p: &FusionPlan) -> Vec<(String, Vec<String>)> {
    p.groups
        .iter()
        .map(|g| {
            let ids = g.nodes.iter().map(|&n| c.nodes[n].id.clone()).collect();
            (g.rule.clone(), ids)
        })
        .collect()
}

/// 2026-09-28: A reference rule over `pattern` (TOML array body) at `priority`.
pub fn fused(id: &str, pattern: &str, priority: i64) -> String {
    fused_with(id, pattern, priority, (1, 128), "numerics = \"reference\"")
}

/// 2026-09-28: A rule over `pattern` with explicit rows and numerics lines.
pub fn fused_with(
    id: &str,
    pattern: &str,
    priority: i64,
    rows: (u64, u64),
    numerics: &str,
) -> String {
    format!(
        r#"
[[rule]]
id = "{id}"
pattern = [{pattern}]
kernels = [{{ module = "m", func = "{id}" }}]
repeat = "once"
emitter = "{id}"
rows = [{}, {}]
modes = ["decode", "multi_seq", "verify"]
{numerics}
priority = {priority}
cite = "test"
"#,
        rows.0, rows.1
    )
}
