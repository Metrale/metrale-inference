// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The buffer planner over random DAGs (fixed seed) and the toy circuit.
//!
//! Owner: metrale-circuit.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use super::*;
use crate::fuser::{AvailableKernels, Policy, fuse};
use crate::ir::{ArchShape, LayerKind};
use crate::precision::PrecisionTable;
use crate::rules::{Mode, parse_rules};

/// 2026-09-28: A 64-bit LCG (Knuth's MMIX constants): deterministic, no dependency.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self, bound: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) % bound
    }
}

const PRECISION: &str = "schema = 1\ncheckpoint = \"r\"\ntier = \"t\"\n[[linear]]\nmatch = \"*\"\nweight = \"bf16\"\nactivation = \"bf16\"\n";

const RULES: &str = r#"
schema = 1

[[rule]]
id = "embed"
pattern = [{ op = "embed" }]
kernels = [{ module = "m", func = "embed" }]
repeat = "once"
emitter = "e"
rows = [1, 128]
modes = ["decode"]
numerics = "reference"
priority = 1
cite = "test"

[[rule]]
id = "copy"
pattern = [{ op = "copy" }]
kernels = [{ module = "m", func = "copy" }]
repeat = "once"
emitter = "e"
rows = [1, 128]
modes = ["decode"]
numerics = "reference"
priority = 1
cite = "test"

[[rule]]
id = "pair"
pattern = [{ op = "copy" }, { op = "copy" }]
kernels = [{ module = "m", func = "pair" }]
repeat = "once"
emitter = "e"
rows = [1, 128]
modes = ["decode"]
numerics = "reference"
priority = 10
cite = "test"
"#;

fn random_circuit(rng: &mut Lcg, nodes: usize) -> String {
    let mut s = String::from(
        "schema = 1\narch = \"rand\"\ndescription = \"r\"\nlayer_module = \"l.{i}\"\n\
         dims = [\"w\"]\nprologue = []\nepilogue = []\ndraft = []\n\
         [layout]\nkind = \"list\"\n[layout.blocks]\nlinear_attention = [\"b\"]\n[block.b]\n",
    );
    let mut readers = vec![0usize; nodes + 1];
    let mut body = String::from(
        "[[block.b.node]]\nid = \"n0\"\nop = \"embed\"\nout = [{ edge = \"e0\", format = \"bf16\", shape = \"n x w\" }]\n",
    );
    for j in 1..=nodes {
        let a = rng.next(j as u64) as usize;
        let mut ins = vec![a];
        if j > 1 && rng.next(2) == 0 {
            let b = rng.next(j as u64) as usize;
            if b != a {
                ins.push(b);
            }
        }
        for &i in &ins {
            readers[i] += 1;
        }
        let list: Vec<String> = ins.iter().map(|i| format!("\"e{i}\"")).collect();
        let fmt = if rng.next(3) == 0 { "f32" } else { "bf16" };
        body.push_str(&format!(
            "[[block.b.node]]\nid = \"n{j}\"\nop = \"copy\"\nin = [{}]\nout = [{{ edge = \"e{j}\", format = \"{fmt}\", shape = \"n x w*{}\" }}]\n",
            list.join(", "),
            1 + rng.next(4)
        ));
    }
    let outs: Vec<String> = (0..=nodes)
        .filter(|&i| readers[i] == 0)
        .map(|i| format!("\"e{i}\""))
        .collect();
    s.push_str(&format!("outputs = [{}]\n", outs.join(", ")));
    s.push_str(&body);
    s
}

#[test]
fn live_edges_never_share_bytes_over_random_dags() {
    let mut rng = Lcg(0x5eed_c1c0);
    let rules = parse_rules(RULES).unwrap();
    let table = PrecisionTable::parse(PRECISION).unwrap();
    let shape = ArchShape {
        layer_kinds: vec![LayerKind::LinearAttention],
        dims: BTreeMap::from([("w".to_string(), 256)]),
    };
    let mut fused_seen = 0;
    for case in 0..200 {
        let nodes = 2 + rng.next(40) as usize;
        let text = random_circuit(&mut rng, nodes);
        let c = crate::instantiate(&text, &shape, &table)
            .unwrap_or_else(|e| panic!("case {case}: {e}"));
        let avail = AvailableKernels::all_named_by(&rules);
        let plan = fuse(&c, &rules, &avail, &Policy::default(), Mode::Decode, 4).unwrap();
        fused_seen += plan
            .edge_states
            .iter()
            .filter(|s| matches!(s, Some(crate::fuser::EdgeState::Fused(_))))
            .count();
        let rows = 1 + rng.next(128);
        let b = plan_buffers(&c, &plan, rows).unwrap();
        let ranges = live_ranges(&c, &plan);
        assert_eq!(
            b.slots.len(),
            ranges.len(),
            "case {case}: every materialised edge is placed"
        );
        for (i, x) in b.slots.iter().enumerate() {
            assert_eq!(x.offset % ALIGN, 0);
            assert!(x.offset + x.bytes <= b.arena_bytes);
            assert_eq!(ranges[&x.edge], (x.start, x.end));
            for y in &b.slots[i + 1..] {
                let live_together = x.start <= y.end && y.start <= x.end;
                let bytes_overlap = x.offset < y.offset + y.bytes && y.offset < x.offset + x.bytes;
                assert!(
                    !(live_together && bytes_overlap),
                    "case {case}: edges {} and {} overlap while both live",
                    c.edges[x.edge].id,
                    c.edges[y.edge].id
                );
            }
        }
        assert!(b.arena_bytes <= b.materialized_bytes + ALIGN * b.slots.len() as u64);
    }
    assert!(
        fused_seen > 0,
        "the pair rule never fused an edge; the property ran on materialised edges only"
    );
}

#[test]
fn a_chain_reuses_the_bytes_of_edges_that_died() {
    let c = crate::test_toy::circuit(4);
    let r = crate::test_toy::rules("");
    let p = crate::test_toy::plan(&c, &r, &crate::test_toy::policy(), 8);
    let b = plan_buffers(&c, &p, 8).unwrap();
    assert!(
        b.arena_bytes < b.materialized_bytes / 2,
        "{} vs {}",
        b.arena_bytes,
        b.materialized_bytes
    );
    let logits = c.edge("head.logits").unwrap();
    let slot = b.slots.iter().find(|s| s.edge == logits).unwrap();
    assert_eq!(slot.end, p.groups.len() - 1, "an output lives to the end");
    assert_eq!(slot.bytes, 8 * 256 * 2);
}

#[test]
fn sizes_follow_the_row_expression_at_the_sized_rows() {
    let c = crate::test_toy::circuit(1);
    let r = crate::test_toy::rules("");
    let p = crate::test_toy::plan(&c, &r, &crate::test_toy::policy(), 2);
    let gu = c.edge("l0.ffn.gu").unwrap();
    for rows in [1u64, 7, 128] {
        let b = plan_buffers(&c, &p, rows).unwrap();
        let slot = b.slots.iter().find(|s| s.edge == gu).unwrap();
        assert_eq!(slot.bytes, rows * 256 * 2);
    }
}
