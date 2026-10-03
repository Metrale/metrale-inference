// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: A tensor-parallel rank's shape and reduces over the toy circuit (its `down`
//! projection standing in for a row-parallel `o`), and the refusals.
//!
//! Owner: metrale-circuit (FEATURES workstream).
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use super::*;
use crate::ir::LayerKind;
use crate::test_toy;

fn shape() -> ArchShape {
    ArchShape {
        layer_kinds: vec![LayerKind::FullAttention],
        dims: BTreeMap::from([
            ("q_heads".to_string(), 24),
            ("kv_heads".to_string(), 4),
            ("lin_v_heads".to_string(), 48),
            ("hidden".to_string(), 5120),
            ("inter".to_string(), 17408),
        ]),
    }
}

#[test]
fn a_rank_holds_its_share_of_the_heads_and_all_of_the_rest() {
    let s = rank_shape(&shape(), TpRank { rank: 1, world: 2 }).unwrap();
    assert_eq!(s.dims["q_heads"], 12);
    assert_eq!(s.dims["kv_heads"], 2);
    assert_eq!(s.dims["lin_v_heads"], 24);
    assert_eq!(
        s.dims["hidden"], 5120,
        "the residual stream is whole on every rank"
    );
    assert_eq!(s.dims["inter"], 17408, "the dense FFN is replicated");
    assert!(
        !s.dims.contains_key("lin_k_heads"),
        "an absent dim stays absent"
    );
}

#[test]
fn the_rank_refusals_name_their_cause() {
    assert_eq!(
        rank_shape(&shape(), TpRank { rank: 0, world: 1 }),
        Err(TpError::Rank { rank: 0, world: 1 })
    );
    assert_eq!(
        rank_shape(&shape(), TpRank { rank: 2, world: 2 }),
        Err(TpError::Rank { rank: 2, world: 2 })
    );
    assert_eq!(
        rank_shape(&shape(), TpRank { rank: 0, world: 8 }),
        Err(TpError::Indivisible {
            dim: "kv_heads".into(),
            value: 4,
            world: 8
        })
    );
}

/// 2026-10-03: The toy circuit with its `down` projections made row-parallel `o`s.
fn row_parallel(layers: usize) -> Circuit {
    let mut c = test_toy::circuit(layers);
    for n in &mut c.nodes {
        if n.op == OpKind::Linear(LinearRole::Down) {
            n.op = OpKind::Linear(LinearRole::O);
        }
    }
    c
}

#[test]
fn each_row_parallel_output_is_reduced_before_anything_reads_it() {
    let base = row_parallel(2);
    let c = with_reduces(&base, TpRank { rank: 1, world: 2 }).unwrap();
    assert_eq!(
        c.nodes.len(),
        base.nodes.len() + 2 + 1,
        "two reduces and the head's"
    );
    for l in 0..2 {
        let o = c.node(&format!("l{l}.ffn.down")).unwrap();
        let r = &c.nodes[o + 1];
        assert_eq!(r.op, OpKind::AllReduce);
        let y = c.nodes[o].outputs[0];
        assert_eq!(r.inputs, vec![y]);
        assert_eq!(
            c.edges[y].consumers,
            vec![o + 1],
            "only the reduce reads the partial sum"
        );
        let summed = r.outputs[0];
        assert_eq!(c.edges[summed].dim_value, c.edges[y].dim_value);
        let base_y = base.edge(&c.edges[y].id).unwrap();
        let before: Vec<&str> = base.edges[base_y]
            .consumers
            .iter()
            .map(|&n| base.nodes[n].id.as_str())
            .collect();
        let after: Vec<&str> = c.edges[summed]
            .consumers
            .iter()
            .map(|&n| c.nodes[n].id.as_str())
            .collect();
        assert_eq!(after, before);
        assert_eq!(c.edges[summed].producer, Some(o + 1));
    }
    let blocks_cover: usize = c.blocks.iter().map(|b| b.end - b.first).sum();
    assert_eq!(blocks_cover, c.nodes.len());
}

#[test]
fn a_draft_head_is_refused() {
    let mut c = row_parallel(1);
    c.blocks.last_mut().unwrap().section = Section::Draft;
    assert_eq!(
        with_reduces(&c, TpRank { rank: 0, world: 2 }),
        Err(TpError::DraftHead)
    );
}

#[test]
fn a_bf16_head_runs_its_vocabulary_slice_and_sums_the_logits_in_place() {
    let base = test_toy::circuit(1);
    let c = shard_head(&base, TpRank { rank: 1, world: 4 }).unwrap();
    let h = c.nodes.iter().position(|n| n.op == OpKind::LmHead).unwrap();
    let head = &c.nodes[h];
    assert_eq!(head.local, SHARDED_HEAD);
    assert_eq!(
        head.params["vocab_begin"], "64",
        "rank 1 of 4 over 256 rows"
    );
    assert_eq!(head.params["vocab_len"], "64");
    let r = &c.nodes[h + 1];
    assert_eq!(r.op, OpKind::AllReduce);
    assert_eq!(r.local, "lm_head_all_reduce");
    assert_eq!(r.inputs, head.outputs, "it sums the logits");
    assert!(
        r.outputs.is_empty(),
        "in place: the logits stay the declared output"
    );
    assert!(c.edges[head.outputs[0]].is_output);
    let odd = shard_head(&base, TpRank { rank: 0, world: 3 }).unwrap();
    assert_eq!(
        odd, base,
        "a vocabulary that does not divide is not sharded"
    );
}

/// 2026-10-03: The toy FFN's `down` made a MoE blend over (expert outputs, weights, shared
/// output, shared gate): four inputs the toy already has.
fn moe_like() -> Circuit {
    let mut c = test_toy::circuit(1);
    let down = c.node("l0.ffn.down").unwrap();
    let up = c.node("l0.ffn.up").unwrap();
    let act = c.node("l0.ffn.act").unwrap();
    let norm = c.node("l0.ffn.norm").unwrap();
    let inputs = vec![
        c.nodes[act].outputs[0],
        c.nodes[up].outputs[0],
        c.nodes[norm].outputs[0],
        c.nodes[act].outputs[0],
    ];
    c.nodes[down].op = OpKind::Blend;
    c.nodes[down].inputs = inputs;
    c
}

#[test]
fn expert_parallelism_reduces_the_routed_sum_and_adds_the_shared_expert_once() {
    let base = moe_like();
    let c = with_expert_reduces(&base, TpRank { rank: 0, world: 2 }).unwrap();
    let b = c.node("l0.ffn.down").unwrap();
    let (routed, reduce, shared) = (&c.nodes[b], &c.nodes[b + 1], &c.nodes[b + 2]);
    let bb = &base.nodes[base.node("l0.ffn.down").unwrap()];
    assert_eq!(routed.op, OpKind::Blend);
    assert_eq!(routed.params["ep"], "routed");
    assert_eq!(
        routed.inputs,
        bb.inputs[..2].to_vec(),
        "experts and weights only"
    );
    assert_eq!(reduce.op, OpKind::EpReduce);
    assert_eq!(reduce.inputs, routed.outputs);
    assert_eq!(shared.params["ep"], "shared");
    assert_eq!(
        shared.inputs,
        vec![reduce.outputs[0], bb.inputs[2], bb.inputs[3]],
        "the summed routed output, then the shared expert and its gate"
    );
    assert_eq!(
        shared.outputs, bb.outputs,
        "the block's FFN output is unchanged"
    );
    let f = shared.outputs[0];
    assert_eq!(c.edges[f].producer, Some(b + 2));
    assert!(matches!(
        with_expert_reduces(&test_toy::circuit(1), TpRank { rank: 0, world: 2 }),
        Err(TpError::NoExperts)
    ));
}
