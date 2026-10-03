// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The LoRA overlay over the toy circuit: where the adapter nodes land, how readers
//! are rewired, the graph's index invariants after the rewrite, and the refusals.
//!
//! Owner: metrale-circuit (FEATURES workstream).
//! Invariants: none beyond the types.

use std::collections::BTreeSet;

use super::*;
use crate::test_toy;

fn spec(roles: &[LinearRole]) -> LoraSpec {
    LoraSpec::uniform(16, [0, 1], roles)
}

/// 2026-10-03: Producers, consumers and block ranges agree with the nodes, and nodes stay in
/// a topological order.
fn assert_consistent(c: &Circuit) {
    for (n, node) in c.nodes.iter().enumerate() {
        for &o in &node.outputs {
            assert_eq!(c.edges[o].producer, Some(n), "edge {}", c.edges[o].id);
        }
        for &i in &node.inputs {
            assert!(c.edges[i].consumers.contains(&n), "edge {}", c.edges[i].id);
            if let Some(p) = c.edges[i].producer {
                assert!(
                    p < n,
                    "`{}` reads `{}` before it is written",
                    node.id,
                    c.edges[i].id
                );
            }
        }
    }
    for e in &c.edges {
        for &n in &e.consumers {
            assert!(c.nodes[n].inputs.contains(&c.edge(&e.id).unwrap()));
        }
    }
    let mut next = 0;
    for b in &c.blocks {
        assert_eq!(b.first, next, "block {} starts at {}", b.template, b.first);
        for n in b.first..b.end {
            assert_eq!(c.nodes[n].block, b.template, "node {}", c.nodes[n].id);
        }
        next = b.end;
    }
    assert_eq!(next, c.nodes.len());
}

#[test]
fn each_adapted_projection_is_followed_by_its_shrink_and_expand_and_readers_move() {
    let base = test_toy::circuit(2);
    let c = adapt(&base, &spec(&[LinearRole::Down])).unwrap();
    assert_consistent(&c);
    assert_eq!(c.nodes.len(), base.nodes.len() + 2 * 2);
    assert_eq!(c.dims[RANK_DIM], 16);
    for layer in 0..2 {
        let down = c.node(&format!("l{layer}.ffn.down")).unwrap();
        let (shrink, expand) = (&c.nodes[down + 1], &c.nodes[down + 2]);
        assert_eq!(
            (shrink.op, expand.op),
            (OpKind::LoraShrink, OpKind::LoraExpand)
        );
        assert_eq!(shrink.local, "down_lora_a");
        assert_eq!(expand.local, "down_lora_b");
        let d = &c.nodes[down];
        assert_eq!(
            shrink.inputs,
            vec![d.inputs[0]],
            "the shrink reads the projection's input"
        );
        assert_eq!(expand.inputs, vec![shrink.outputs[0], d.outputs[0]]);
        let y = d.outputs[0];
        assert_eq!(
            c.edges[y].consumers,
            vec![down + 2],
            "only the expand reads the base output"
        );
        let adapted = expand.outputs[0];
        assert_eq!(c.edges[adapted].format, c.edges[y].format);
        assert_eq!(c.edges[adapted].dim_value, c.edges[y].dim_value);
        let readers_before: Vec<&str> = base.edges[base.edge(&c.edges[y].id).unwrap()]
            .consumers
            .iter()
            .map(|&n| base.nodes[n].id.as_str())
            .collect();
        let readers_after: Vec<&str> = c.edges[adapted]
            .consumers
            .iter()
            .map(|&n| c.nodes[n].id.as_str())
            .collect();
        assert_eq!(readers_after, readers_before);
        assert_eq!(c.edges[shrink.outputs[0]].dim_value, 16);
    }
    let up = c.node("l0.ffn.up").unwrap();
    assert!(
        c.nodes.iter().all(|n| !n.id.starts_with("l0.ffn.up_lora")),
        "an unadapted role gets no adapter"
    );
    assert_ne!(c.nodes[up + 1].op, OpKind::LoraShrink);
}

#[test]
fn the_gate_up_shrink_holds_both_adapters_ranks() {
    let c = adapt(
        &test_toy::circuit(1),
        &LoraSpec::uniform(16, [0], &[LinearRole::GateUp]),
    )
    .unwrap();
    assert_consistent(&c);
    let shrink = c.node("l0.ffn.up_lora_a").unwrap();
    let xa = c.nodes[shrink].outputs[0];
    assert_eq!(c.edges[xa].dim_value, 32);
    assert_eq!(c.edges[xa].dim.text(), "lora_rank*2");
}

#[test]
fn the_refusals_name_their_cause() {
    let c = test_toy::circuit(2);
    assert_eq!(
        adapt(&c, &spec(&[])),
        Err(LoraError::Empty { rank: 16, roles: 0 })
    );
    let zero = LoraSpec::uniform(0, [0], &[LinearRole::Down]);
    assert!(matches!(
        adapt(&c, &zero),
        Err(LoraError::Empty { rank: 0, .. })
    ));
    assert_eq!(
        adapt(&c, &spec(&[LinearRole::Qkvz])),
        Err(LoraError::Role("qkvz"))
    );
    assert_eq!(
        adapt(&c, &spec(&[LinearRole::Q])),
        Err(LoraError::Absent("q", 0))
    );
    assert_eq!(
        adapt(&c, &LoraSpec::uniform(16, [5], &[LinearRole::Down])),
        Err(LoraError::Absent("down", 5))
    );
    let once = adapt(&c, &spec(&[LinearRole::Down])).unwrap();
    assert_eq!(
        adapt(&once, &spec(&[LinearRole::Down])),
        Err(LoraError::Twice)
    );
}

#[test]
fn a_projection_reading_a_quantized_activation_is_refused() {
    let mut c = test_toy::circuit(2);
    let down = c.node("l0.ffn.down").unwrap();
    let x = c.nodes[down].inputs[0];
    c.edges[x].format = Format::Nvfp4 { group: 16 };
    assert!(matches!(
        adapt(&c, &spec(&[LinearRole::Down])),
        Err(LoraError::QuantizedInput(id, _)) if id == "l0.ffn.down"
    ));
}

#[test]
fn only_the_listed_layers_are_adapted() {
    let base = test_toy::circuit(3);
    let spec = LoraSpec {
        rank: 8,
        layers: BTreeMap::from([(1, BTreeSet::from([LinearRole::Down]))]),
    };
    let c = adapt(&base, &spec).unwrap();
    assert_consistent(&c);
    let adapted: Vec<&str> = c
        .nodes
        .iter()
        .filter(|n| n.op == OpKind::LoraShrink)
        .map(|n| n.id.as_str())
        .collect();
    assert_eq!(adapted, ["l1.ffn.down_lora_a"]);
}

#[test]
fn the_pool_holds_every_slots_a_and_b_at_the_padded_rank() {
    let base = test_toy::circuit(2);
    let roles = [LinearRole::GateUp, LinearRole::Down];
    let c = adapt(&base, &LoraSpec::uniform(8, [0, 1], &roles)).unwrap();
    let (h, i) = (base.dims["hidden"], base.dims["inter"]);
    // 2026-10-03: Per layer: gate and up (A 2 x [8, h], B [2i, 8]) and down (A [8, i], B [h, 8]).
    let per_layer = (2 * 8 * h + 2 * i * 8) + (8 * i + h * 8);
    assert_eq!(pool_bytes(&c, 3), 3 * 2 * per_layer * 2);
    assert_eq!(
        pool_bytes(&base, 3),
        0,
        "an unadapted circuit holds no pool"
    );
}
