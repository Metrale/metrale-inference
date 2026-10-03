// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The memory model's formulas on synthetic circuits: weights are `out x k` at the
//! declared format with every derived copy, routed experts are counted once per expert whatever
//! the rows, activations come from the planner's peak run, a workspace is one buffer per family,
//! the KV demand keeps one block beyond the tokens, and the device total is the sum of its terms.
//!
//! Owner: metrale-circuit (memory).
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use super::*;
use crate::fuser::{AvailableKernels, fuse};
use crate::ir::ArchShape;
use crate::precision::PrecisionTable;
use crate::rules::{Mode, parse_rules};
use crate::state::StateDtype;
use crate::test_toy;

const TERMS: DriverTerms = DriverTerms {
    driver_fixed_bytes: 1000,
    driver_budget_per_mille: 10,
    unified: true,
    util_ceiling: 0.85,
};

fn state_inputs() -> StateInputs {
    StateInputs {
        formats: BTreeMap::from([("kv".to_string(), StateDtype::Bf16)]),
        slots: 1,
        verify: None,
        kv: None,
        draft_kv: None,
    }
}

struct Fixture {
    served: Circuit,
    declared: Circuit,
    settings: BTreeMap<String, String>,
}

fn toy(layers: usize, declared_table: &str) -> Fixture {
    let table = PrecisionTable::parse(declared_table).unwrap();
    Fixture {
        served: test_toy::circuit(layers),
        declared: crate::instantiate(test_toy::CIRCUIT, &[], &test_toy::shape(layers), &table)
            .unwrap(),
        settings: BTreeMap::new(),
    }
}

fn eval(
    f: &Fixture,
    copies: &[CopyRule],
    runs: &[ActivationRun<'_>],
    fams: Option<(&Families, u64)>,
    legacy: Option<u64>,
) -> MemoryReport {
    evaluate(&MemoryInputs {
        served: &f.served,
        declared: &f.declared,
        copies,
        settings: &f.settings,
        states: &state_inputs(),
        draft_kv_dtype: None,
        caches: &CacheInputs::default(),
        runs,
        families: fams,
        legacy_arena: legacy,
        driver: TERMS,
        budget_bytes: 1_000_000,
        chunk_slack: 7,
    })
    .unwrap()
}

#[test]
fn stored_weights_are_out_times_k_at_the_declared_format() {
    let f = toy(2, test_toy::PRECISION);
    let r = eval(&f, &[], &[], None, None);
    // 2026-10-02: embed 256x64 bf16; per layer up nvfp4 [256, 64] = 8192 packed + 1024 scales +
    // 4 global, down nvfp4 [64, 128] = 4096 + 512 + 4; lm_head bf16 [256, 64].
    let embed = 256 * 64 * 2;
    let layer = (8192 + 1024 + 4) + (4096 + 512 + 4);
    assert_eq!(r.totals.weights_stored, embed + 2 * layer + 256 * 64 * 2);
    assert_eq!(r.totals.weights_derived, 0);
    let up = f.served.node("l0.ffn.up").unwrap();
    assert_eq!(r.nodes[up].stored, 8192 + 1024 + 4);
}

const BF16_TABLE: &str = r#"
schema = 1
checkpoint = "toy"
tier = "declared"
[[linear]]
match = "*"
weight = "bf16"
activation = "bf16"
"#;

const RULES: &str = r#"
schema = 1
[[copy]]
id = "requant"
arch = ["toy"]
ops = ["linear:gate_up", "linear:down"]
section = "main"
stored = ["bf16"]
served = ["nvfp4/g16"]
law = "nvfp4/g16"
count = 1
site = "test"
[[copy]]
id = "twin"
arch = ["toy"]
ops = ["linear:gate_up"]
section = "main"
served = ["nvfp4/g16"]
when = { speculative = "on" }
law = "nvfp4/g16"
count = 2
leaked = true
site = "test"
"#;

#[test]
fn a_served_format_other_than_the_stored_one_is_a_derived_copy() {
    let mut f = toy(1, BF16_TABLE);
    let rules = parse_copies(RULES).unwrap();
    let r = eval(&f, &rules, &[], None, None);
    let up = f.served.node("l0.ffn.up").unwrap();
    // 2026-10-02: Stored at the declared bf16, requantized once to the served nvfp4.
    assert_eq!(r.nodes[up].stored, 256 * 64 * 2);
    assert_eq!(r.nodes[up].derived, 8192 + 1024 + 4);
    assert_eq!(r.totals.weights_leaked, 0);
    f.settings.insert("speculative".into(), "on".into());
    let r = eval(&f, &rules, &[], None, None);
    assert_eq!(r.nodes[up].derived, 3 * (8192 + 1024 + 4));
    assert_eq!(r.totals.weights_leaked, 2 * (8192 + 1024 + 4));
}

#[test]
fn copy_rules_with_unknown_spellings_are_refused() {
    for (from, to) in [
        (
            "law = \"nvfp4/g16\"\ncount = 1",
            "law = \"nvfp5\"\ncount = 1",
        ),
        ("\"linear:down\"]", "\"linear:sideways\"]"),
        ("\"linear:down\"]", "\"silu_mul\"]"),
        ("when = { speculative", "when = { mood"),
        ("section = \"main\"\nstored", "section = \"both\"\nstored"),
        ("count = 2", "count = 0"),
    ] {
        assert!(RULES.contains(from), "{from}");
        let text = RULES.replacen(from, to, 1);
        assert!(parse_copies(&text).is_err(), "`{to}` parsed");
    }
    let dup = format!("{RULES}{}", &RULES[RULES.find("[[copy]]").unwrap()..]);
    assert!(parse_copies(&dup).unwrap_err().0.contains("listed twice"));
}

#[test]
fn activations_come_from_the_peak_run_and_attribute_to_producers() {
    let f = toy(2, test_toy::PRECISION);
    let rules = test_toy::rules("");
    let policy = test_toy::policy();
    let (p1, p8) = (
        test_toy::plan(&f.served, &rules, &policy, 1),
        test_toy::plan(&f.served, &rules, &policy, 8),
    );
    let runs = [
        ActivationRun {
            label: "one",
            rows: 1,
            plan: &p1,
        },
        ActivationRun {
            label: "eight",
            rows: 8,
            plan: &p8,
        },
    ];
    let r = eval(&f, &[], &runs, None, None);
    let b8 = planner::plan_buffers(&f.served, &p8, 8).unwrap();
    assert_eq!(r.totals.activations_planned, b8.arena_bytes);
    assert_eq!(r.totals.activations, b8.arena_bytes);
    let per_node: u64 = r.nodes.iter().map(|n| n.activations).sum();
    assert_eq!(per_node, b8.materialized_bytes);
    assert!(r.runs[0].arena < r.runs[1].arena);
    // 2026-10-02: The gate_up output is 8 rows x 256 bf16.
    let up = f.served.node("l0.ffn.up").unwrap();
    assert_eq!(r.nodes[up].activations, 8 * 256 * 2);
    // 2026-10-02: A legacy arena replaces the planned one in the charge, not in the report.
    let r = eval(&f, &[], &runs, None, Some(123));
    assert_eq!(
        (r.totals.activations, r.totals.activations_planned),
        (123, b8.arena_bytes)
    );
}

const FAMILIES: &str = r#"
schema = 1
hardware = "toy"
[roofline]
dram_gbps = 1.0
bf16_tflops = 1.0
fp8_tflops = 1.0
nvfp4_tflops = 1.0
context_tokens = 16
[[family]]
id = "norm"
description = "toy"
compute = "memory"
kernels = ["m::norm", "m::final_norm"]
rows = [1, 128]
pipeline.rms_norm = { in = ["bf16"], compute = "f32", out = ["bf16"] }
pipeline.final_norm = { in = ["bf16"], compute = "f32", out = ["bf16"] }
op = [{ op = "rms_norm" }, { op = "final_norm" }]
[[family.point]]
values = {}
how = "instantiation"
files = ["toy.cu"]
[[family.workspace]]
name = "partials"
bytes = ["n*hidden*4", "sm_count*1000"]
arena = true
why = "test"
[[family.workspace]]
name = "scales"
bytes = ["n*hidden/32*4"]
arena = false
why = "test"
"#;

#[test]
fn a_family_workspace_is_one_buffer_at_the_largest_expression() {
    let f = toy(3, test_toy::PRECISION);
    let fams = crate::venn::parse_families(FAMILIES).unwrap();
    let rules = test_toy::rules("");
    let p = test_toy::plan(&f.served, &rules, &test_toy::policy(), 4);
    let runs = [ActivationRun {
        label: "four",
        rows: 4,
        plan: &p,
    }];
    let r = eval(&f, &[], &runs, Some((&fams, 2)), None);
    let by: BTreeMap<&str, (u64, usize)> = r
        .workspaces
        .iter()
        .map(|w| (w.name.as_str(), (w.bytes, w.nodes.len())))
        .collect();
    // 2026-10-02: max(4 x 64 x 4, 2 x 1000); 4 x ceil(64 / 32) x 4. Three layer norms and the
    // final norm share each buffer.
    assert_eq!(by["partials"], (2000, 4));
    assert_eq!(by["scales"], (32, 4));
    assert_eq!(r.totals.workspace, 2032);
    let norm = f.served.node("l1.ffn.norm").unwrap();
    assert_eq!(r.nodes[norm].workspace, 2000);
    // 2026-10-02: Over a legacy arena, the scratch it already holds is not charged again.
    let r = eval(&f, &[], &runs, Some((&fams, 2)), Some(10));
    assert_eq!(r.totals.workspace, 32);
    // 2026-10-02: A name the expression reads that no dim gives is an error.
    let bad = FAMILIES.replace("n*hidden/32*4", "n*heads*4");
    let fams = crate::venn::parse_families(&bad).unwrap();
    let e = evaluate(&MemoryInputs {
        served: &f.served,
        declared: &f.declared,
        copies: &[],
        settings: &f.settings,
        states: &state_inputs(),
        draft_kv_dtype: None,
        caches: &CacheInputs::default(),
        runs: &runs,
        families: Some((&fams, 2)),
        legacy_arena: None,
        driver: TERMS,
        budget_bytes: 1,
        chunk_slack: 0,
    });
    assert!(matches!(e, Err(MemoryError::Workspace { .. })), "{e:?}");
}

const MOE: &str = r#"
schema = 1
arch = "toy_moe"
description = "toy moe"
layer_module = "layers.{i}"
include = []
dims = ["hidden", "experts", "top_k", "moe_inter", "vocab"]
prologue = ["embed"]
epilogue = ["head"]
draft = []
[layout]
kind = "list"
[layout.blocks]
linear_attention = ["moe"]
full_attention = ["moe"]
[block.embed]
stream_out = "h"
[[block.embed.node]]
id = "embed"
op = "embed"
out = [{ edge = "h", format = "bf16", shape = "n x hidden" }]
[block.moe]
stream_in = "x"
stream_out = "y"
[[block.moe.node]]
id = "router"
op = "router"
in = ["x"]
out = [{ edge = "logits", format = "bf16", shape = "n x experts" }]
binding = ["{L}.gate"]
[[block.moe.node]]
id = "top_k"
op = "top_k"
in = ["logits"]
out = [
  { edge = "w", format = "f32", shape = "n x top_k" },
  { edge = "id", format = "i32", shape = "n x top_k" },
]
params = { top_k = "2", scoring = "softmax" }
[[block.moe.node]]
id = "gate_up"
op = "expert_gate_up"
in = ["x", "id"]
out = [{ edge = "gu", format = "bf16", shape = "n*top_k x moe_inter*2" }]
binding = ["{L}.experts.*.gate_up"]
[[block.moe.node]]
id = "act"
op = "silu_mul"
in = ["gu"]
out = [{ edge = "a", format = "bf16", shape = "n*top_k x moe_inter" }]
[[block.moe.node]]
id = "down"
op = "expert_down"
in = ["a", "id"]
out = [{ edge = "d", format = "bf16", shape = "n*top_k x hidden" }]
binding = ["{L}.experts.*.down"]
[[block.moe.node]]
id = "blend"
op = "blend"
in = ["d", "w"]
out = [{ edge = "f", format = "bf16", shape = "n x hidden" }]
[[block.moe.node]]
id = "add"
op = "residual_add"
in = ["x", "f"]
out = [{ edge = "y", format = "bf16", shape = "n x hidden" }]
[block.head]
stream_in = "x"
outputs = ["logits"]
[[block.head.node]]
id = "lm_head"
op = "lm_head"
in = ["x"]
out = [{ edge = "logits", format = "bf16", shape = "n x vocab" }]
binding = ["lm_head"]
"#;

#[test]
fn routed_experts_are_counted_once_per_expert_not_per_routed_row() {
    let shape = ArchShape {
        layer_kinds: vec![crate::ir::LayerKind::LinearAttention],
        dims: [
            ("hidden", 64),
            ("experts", 16),
            ("top_k", 2),
            ("moe_inter", 32),
            ("vocab", 100),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect(),
    };
    let table = PrecisionTable::parse(test_toy::PRECISION).unwrap();
    let c = crate::instantiate(MOE, &[], &shape, &table).unwrap();
    let f = Fixture {
        served: c.clone(),
        declared: c,
        settings: BTreeMap::new(),
    };
    let mut text = String::from("schema = 1\n");
    for (id, op) in [
        ("embed", "embed"),
        ("router", "router"),
        ("top_k", "top_k"),
        ("gate_up", "expert_gate_up"),
        ("act", "silu_mul"),
        ("down", "expert_down"),
        ("blend", "blend"),
        ("add", "residual_add"),
        ("lm_head", "lm_head"),
    ] {
        text.push_str(&test_toy::single(id, op, ""));
    }
    let rules = parse_rules(&text).unwrap();
    let avail = AvailableKernels::all_named_by(&rules);
    let pol = test_toy::policy();
    let plan = |rows| fuse(&f.served, &rules, &avail, &pol, Mode::Decode, rows).unwrap();
    let (p1, p64) = (plan(1), plan(64));
    let one = eval(
        &f,
        &[],
        &[ActivationRun {
            label: "1",
            rows: 1,
            plan: &p1,
        }],
        None,
        None,
    );
    let wide = eval(
        &f,
        &[],
        &[ActivationRun {
            label: "64",
            rows: 64,
            plan: &p64,
        }],
        None,
        None,
    );
    let gu = f.served.node("l0.moe.gate_up").unwrap();
    let dn = f.served.node("l0.moe.down").unwrap();
    // 2026-10-02: 16 experts x nvfp4 [64, 64] and 16 x nvfp4 [64, 32], whatever the rows.
    let expert_gu = 64 * 64 / 2 + 64 * 64 / 16 + 4;
    let expert_dn = 64 * 32 / 2 + 64 * 32 / 16 + 4;
    assert_eq!(one.nodes[gu].stored, 16 * expert_gu);
    assert_eq!(one.nodes[dn].stored, 16 * expert_dn);
    assert_eq!(one.totals.weights_stored, wide.totals.weights_stored);
    // 2026-10-02: The routed rows scale the activations, not the weights.
    assert_eq!(wide.nodes[gu].activations, 64 * 2 * 64 * 2);
    assert_eq!(one.nodes[gu].activations, 2 * 64 * 2);
}

#[test]
fn the_kv_demand_keeps_one_block_beyond_the_tokens_and_one_dummy() {
    // 2026-10-02: Scheduler admission reserves `tokens / block + 1` blocks per sequence.
    for (tokens, per_seq) in [
        (1, 1),
        (15, 1),
        (16, 2),
        (17, 2),
        (31, 2),
        (32, 3),
        (1152, 73),
    ] {
        assert_eq!(kv_blocks_for(1, tokens, 16), Some(per_seq + 1), "{tokens}");
        assert_eq!(
            kv_blocks_for(3, tokens, 16),
            Some(3 * per_seq + 1),
            "{tokens}"
        );
    }
    assert_eq!(kv_blocks_for(1, 10, 0), None);
    assert_eq!(kv_blocks_for(u64::MAX, 16, 16), None);
}

#[test]
fn the_device_total_is_the_sum_of_its_terms() {
    let f = toy(1, test_toy::PRECISION);
    let r = eval(&f, &[], &[], None, None);
    let t = r.totals;
    assert_eq!(t.driver, 1000 + 1_000_000 / 1000 * 10 + 7);
    assert_eq!(
        t.device,
        t.weights_stored
            + t.weights_derived
            + t.activations
            + t.workspace
            + t.states
            + t.caches
            + t.driver
    );
    assert_eq!(r.headroom(), 1_000_000 - i128::from(t.device));
}
