// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The buffer planner over random DAGs (fixed seed) and the toy circuit.
//!
//! Owner: metrale-circuit.
//! Invariants: none beyond the types.

use std::collections::{BTreeMap, BTreeSet};

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
        "schema = 1\narch = \"rand\"\ndescription = \"r\"\nlayer_module = \"l.{i}\"\ninclude = []\n\
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
        let c = crate::instantiate(&text, &[], &shape, &table)
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

fn toy(layers: usize) -> (Circuit, FusionPlan) {
    let c = crate::test_toy::circuit(layers);
    let r = crate::test_toy::rules("");
    let p = crate::test_toy::plan(&c, &r, &crate::test_toy::policy(), 8);
    (c, p)
}

fn slot(b: &BufferPlan, e: EdgeIdx) -> Slot {
    *b.slots.iter().find(|s| s.edge == e).unwrap()
}

#[test]
fn the_default_layout_places_exactly_what_plan_buffers_does() {
    let (c, p) = toy(3);
    assert_eq!(
        plan_buffers(&c, &p, 8).unwrap(),
        plan_buffers_with(&c, &p, 8, &Layout::default()).unwrap()
    );
}

#[test]
fn an_external_edge_is_not_placed() {
    let (c, p) = toy(2);
    let logits = c.edge("head.logits").unwrap();
    let layout = Layout {
        external: BTreeSet::from([logits]),
        ..Layout::default()
    };
    let with = plan_buffers_with(&c, &p, 8, &layout).unwrap();
    let without = plan_buffers(&c, &p, 8).unwrap();
    assert!(with.slots.iter().all(|s| s.edge != logits));
    assert_eq!(with.slots.len() + 1, without.slots.len());
    assert!(with.arena_bytes <= without.arena_bytes);
}

#[test]
fn an_alias_shares_the_input_bytes_and_holds_them_over_both_ranges() {
    let (c, p) = toy(1);
    let xn = c.edge("l0.ffn.xn").unwrap();
    let d = c.edge("l0.ffn.d").unwrap();
    let layout = Layout {
        aliases: vec![(d, xn)],
        ..Layout::default()
    };
    let b = plan_buffers_with(&c, &p, 8, &layout).unwrap();
    let (sx, sd) = (slot(&b, xn), slot(&b, d));
    assert_eq!(sx.offset, sd.offset);
    let ranges = live_ranges(&c, &p);
    let union = (
        ranges[&xn].0.min(ranges[&d].0),
        ranges[&xn].1.max(ranges[&d].1),
    );
    assert_eq!((sx.start, sx.end), union);
    for s in b.slots.iter().filter(|s| s.edge != xn && s.edge != d) {
        let live_together = s.start <= union.1 && union.0 <= s.end;
        let bytes_overlap = s.offset < sx.offset + sx.bytes && sx.offset < s.offset + s.bytes;
        assert!(
            !(live_together && bytes_overlap),
            "`{}` sits on the alias class while it is live",
            c.edges[s.edge].id
        );
    }
}

#[test]
fn a_pack_lays_its_edges_back_to_back_in_order() {
    let (c, p) = toy(1);
    let gu = c.edge("l0.ffn.gu").unwrap();
    let a = c.edge("l0.ffn.a").unwrap();
    for pack in [vec![gu, a], vec![a, gu]] {
        let layout = Layout {
            packs: vec![pack.clone()],
            ..Layout::default()
        };
        let b = plan_buffers_with(&c, &p, 8, &layout).unwrap();
        let (first, second) = (slot(&b, pack[0]), slot(&b, pack[1]));
        assert_eq!(first.offset % ALIGN, 0);
        assert_eq!(second.offset, first.offset + first.bytes);
        assert_eq!((first.start, first.end), (second.start, second.end));
    }
}

#[test]
fn a_row_pack_interleaves_its_edges_within_each_row() {
    let (c, p) = toy(1);
    let gu = c.edge("l0.ffn.gu").unwrap();
    let a = c.edge("l0.ffn.a").unwrap();
    let rows = 8u64;
    for pack in [vec![gu, a], vec![a, gu]] {
        let layout = Layout {
            row_packs: vec![RowPack {
                members: pack.clone(),
                over: None,
            }],
            ..Layout::default()
        };
        let b = plan_buffers_with(&c, &p, rows, &layout).unwrap();
        let (first, second) = (slot(&b, pack[0]), slot(&b, pack[1]));
        let row = |s: Slot| s.bytes / rows;
        assert_eq!(first.offset % ALIGN, 0);
        assert_eq!(second.offset, first.offset + row(first));
        assert_eq!(first.row_stride, row(first) + row(second));
        assert_eq!(second.row_stride, first.row_stride);
        let (lo, hi) = (first.offset, first.offset + rows * first.row_stride);
        for s in b.slots.iter().filter(|s| s.edge != gu && s.edge != a) {
            let mut dims = c.dims.clone();
            dims.insert("n".into(), rows);
            let own_rows = c.edges[s.edge].rows.eval(&dims);
            assert_eq!(s.row_stride * own_rows.unwrap().max(1), s.bytes);
            let live_together = s.start <= first.end && first.start <= s.end;
            let bytes_overlap = s.offset < hi && lo < s.offset + s.bytes;
            assert!(
                !(live_together && bytes_overlap),
                "`{}` sits inside the row pack while it is live",
                c.edges[s.edge].id
            );
        }
    }
}

#[test]
fn a_row_pack_over_an_edge_takes_its_bytes_and_an_alias_takes_its_stride() {
    let (c, p) = toy(2);
    let e = |id: &str| c.edge(id).unwrap();
    let (gu, a0, a1, d0) = (e("l0.ffn.gu"), e("l0.ffn.a"), e("l1.ffn.a"), e("l0.ffn.d"));
    let rows = 8u64;
    let layout = Layout {
        row_packs: vec![RowPack {
            members: vec![a0, a1],
            over: Some(gu),
        }],
        ..Layout::default()
    };
    let b = plan_buffers_with(&c, &p, rows, &layout).unwrap();
    let (sg, s0, s1) = (slot(&b, gu), slot(&b, a0), slot(&b, a1));
    assert_eq!(sg.offset, s0.offset);
    assert_eq!(s1.offset, s0.offset + s0.bytes / rows);
    assert_eq!(sg.row_stride, sg.bytes / rows);
    assert_eq!(
        (s0.row_stride, s1.row_stride),
        (sg.row_stride, sg.row_stride)
    );
    let span = (sg.start.min(s1.start), sg.end.max(s1.end));
    assert_eq!((sg.start, sg.end), span);
    for s in b.slots.iter().filter(|s| ![gu, a0, a1].contains(&s.edge)) {
        let live_together = s.start <= span.1 && span.0 <= s.end;
        let bytes_overlap = s.offset < sg.offset + sg.bytes && sg.offset < s.offset + s.bytes;
        assert!(
            !(live_together && bytes_overlap),
            "`{}`",
            c.edges[s.edge].id
        );
    }
    let with_alias = Layout {
        aliases: vec![(d0, e("l0.ffn.xn"))],
        row_packs: vec![RowPack {
            members: vec![e("l0.ffn.xn"), a0],
            over: None,
        }],
        ..Layout::default()
    };
    let b = plan_buffers_with(&c, &p, rows, &with_alias).unwrap();
    assert_eq!(slot(&b, d0).offset, slot(&b, e("l0.ffn.xn")).offset);
    assert_eq!(slot(&b, d0).row_stride, slot(&b, a0).row_stride);
}

#[test]
fn layouts_the_plan_cannot_honour_are_refused() {
    let (c, p) = toy(1);
    let e = |id: &str| c.edge(id).unwrap();
    let refused = |layout: Layout, want: &str| {
        let err = plan_buffers_with(&c, &p, 8, &layout)
            .unwrap_err()
            .to_string();
        assert!(err.contains(want), "{err} lacks `{want}`");
    };
    refused(
        Layout {
            aliases: vec![(e("l0.ffn.a"), e("l0.ffn.gu"))],
            ..Layout::default()
        },
        "sizes differ",
    );
    refused(
        Layout {
            external: BTreeSet::from([e("head.logits")]),
            aliases: vec![(e("head.logits"), e("head.xn"))],
            ..Layout::default()
        },
        "external",
    );
    refused(
        Layout {
            external: BTreeSet::from([e("l0.ffn.gu")]),
            packs: vec![vec![e("l0.ffn.gu"), e("l0.ffn.a")]],
            ..Layout::default()
        },
        "external",
    );
    refused(
        Layout {
            packs: vec![vec![e("l0.ffn.gu")], vec![e("l0.ffn.gu"), e("l0.ffn.a")]],
            ..Layout::default()
        },
        "packed twice",
    );
    let row_pack = |members: Vec<usize>, over: Option<usize>| RowPack { members, over };
    refused(
        Layout {
            packs: vec![vec![e("l0.ffn.gu")]],
            row_packs: vec![row_pack(vec![e("l0.ffn.gu"), e("l0.ffn.a")], None)],
            ..Layout::default()
        },
        "packed twice",
    );
    refused(
        Layout {
            row_packs: vec![row_pack(vec![e("l0.ffn.a")], Some(e("l0.ffn.gu")))],
            ..Layout::default()
        },
        "holds",
    );
    refused(
        Layout {
            external: BTreeSet::from([e("l0.ffn.a")]),
            row_packs: vec![row_pack(vec![e("l0.ffn.gu"), e("l0.ffn.a")], None)],
            ..Layout::default()
        },
        "external",
    );
    let up_act = r#"{ op = "linear", role = "gate_up" }, { op = "silu_mul" }"#;
    let fused_rules = crate::test_toy::rules(&crate::test_toy::fused("up_act", up_act, 10));
    let fp = crate::test_toy::plan(&c, &fused_rules, &crate::test_toy::policy(), 8);
    let err = plan_buffers_with(
        &c,
        &fp,
        8,
        &Layout {
            external: BTreeSet::from([e("l0.ffn.gu")]),
            ..Layout::default()
        },
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("not materialised"), "{err}");
}
