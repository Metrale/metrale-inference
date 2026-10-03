// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Tensor and expert parallelism of a circuit's target (LIFECYCLE-DESIGN.md 15.10):
//! each rank plans its own circuit, so every rank's plan has its own digest.
//! - TP: instantiated at its share of the heads ([`rank_shape`]), with an `all_reduce` after
//!   every row-parallel projection ([`with_reduces`]).
//! - EP: every expert is routed globally and a rank runs its own; each MoE `blend` is split into
//!   the routed sum, an `ep_reduce` over the ranks and the gated shared expert added once
//!   ([`with_expert_reduces`]), as legacy (`moe/forward.rs`, `moe/forward/ep_reduce.rs`).
//! - Both: a BF16 head is vocabulary-parallel, as legacy's `lmhead_vocab_shard` over the
//!   communicator's world: the rank computes its slice of the logits into a zeroed buffer and an
//!   in-place `all_reduce` (a node with no output) sums the slices ([`shard_head`]).
//!
//! Owner: metrale-circuit (FEATURES workstream).
//! Invariants:
//! - The sharding is legacy's (`serve_phases/topology.rs`): attention and GatedDeltaNet heads are
//!   divided by the world, the dense FFN is replicated, and only the attention `o` and the
//!   GatedDeltaNet `out_proj` outputs are partial sums (the all-reduce sites of
//!   `qwen3_attention/trait_impl/decode_inner.rs` and `qwen3_ssm/trait_prefill_helper.rs`).
//! - A dim that does not divide, a rank outside the world, and a draft head (legacy runs it
//!   replicated over the divided config, which this plan cannot describe) are refused.

use std::collections::{BTreeMap, BTreeSet};

use crate::follow::{Followers, push_edge};
use crate::ir::{ArchShape, Circuit, Edge, EdgeIdx, LinearRole, Node, NodeIdx, OpKind, Section};

/// 2026-10-03: The dims each rank holds a share of.
pub const SHARDED_DIMS: [&str; 4] = ["q_heads", "kv_heads", "lin_k_heads", "lin_v_heads"];

/// 2026-10-03: The projections whose outputs are partial sums under TP.
pub const REDUCED: [LinearRole; 2] = [LinearRole::O, LinearRole::GdnOut];

/// 2026-10-03: The head node's id once vocabulary-parallel; its rules match it by this local.
pub const SHARDED_HEAD: &str = "lm_head_vocab_shard";

/// 2026-10-03: How a process's circuit is parallel: its rank of a tensor- or an expert-parallel
/// world (the two together are not modelled).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parallel {
    Tensor(TpRank),
    Expert(TpRank),
}

/// 2026-10-03: One rank of a tensor- or expert-parallel world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TpRank {
    pub rank: u64,
    pub world: u64,
}

/// 2026-10-03: Why a rank's circuit could not be planned.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TpError {
    /// 2026-10-03: A world below 2, or a rank outside it.
    #[error("rank {rank} of a world of {world}")]
    Rank { rank: u64, world: u64 },
    /// 2026-10-03: A sharded dim the world does not divide.
    #[error("`{dim}` = {value} does not divide over {world} ranks")]
    Indivisible { dim: String, value: u64, world: u64 },
    /// 2026-10-03: A circuit with a draft head.
    #[error(
        "a draft head under tensor parallelism (legacy runs it replicated over the divided heads)"
    )]
    DraftHead,
    /// 2026-10-03: Expert parallelism over a circuit without MoE blends.
    #[error("expert parallelism over a circuit with no MoE `blend`")]
    NoExperts,
    /// 2026-10-03: A `blend` that is not (expert outputs, weights, shared output, shared gate).
    #[error("`{0}` is not a routed-plus-shared blend; expert parallelism cannot split it")]
    BlendShape(String),
    /// 2026-10-03: A row-parallel projection whose output leaves the program.
    #[error("`{0}`'s output is read outside the program")]
    External(String),
}

/// 2026-10-03: `shape` as rank `tp` of its world holds it.
pub fn rank_shape(shape: &ArchShape, tp: TpRank) -> Result<ArchShape, TpError> {
    if tp.world < 2 || tp.rank >= tp.world {
        return Err(TpError::Rank {
            rank: tp.rank,
            world: tp.world,
        });
    }
    let mut out = shape.clone();
    for d in SHARDED_DIMS {
        if let Some(v) = out.dims.get_mut(d) {
            if *v % tp.world != 0 {
                return Err(TpError::Indivisible {
                    dim: d.to_string(),
                    value: *v,
                    world: tp.world,
                });
            }
            *v /= tp.world;
        }
    }
    Ok(out)
}

/// 2026-10-03: `circuit` (instantiated at rank `tp`'s shape) with an `all_reduce` after every
/// row-parallel projection of the target, and its head sharded ([`shard_head`]).
pub fn with_reduces(circuit: &Circuit, tp: TpRank) -> Result<Circuit, TpError> {
    if circuit.blocks.iter().any(|b| b.section == Section::Draft) {
        return Err(TpError::DraftHead);
    }
    let chosen = main_where(
        circuit,
        |n| matches!(n.op, OpKind::Linear(r) if REDUCED.contains(&r)),
    );
    shard_head(&crate::follow::follow(circuit, &chosen, reduce_of)?, tp)
}

/// 2026-10-03: The target's nodes `pick` selects.
fn main_where(c: &Circuit, pick: impl Fn(&Node) -> bool) -> BTreeSet<NodeIdx> {
    let main = crate::follow::main_nodes(c);
    (0..c.nodes.len())
        .filter(|&n| main[n] && pick(&c.nodes[n]))
        .collect()
}

/// 2026-10-03: `circuit` with each MoE `blend` of the target split for expert parallelism: the
/// routed sum (the `blend` node, its shared inputs dropped, `ep = "routed"`), `<blend>_ep_reduce`
/// over the ranks, and `<blend>_ep_shared` adding the gated shared expert once (`ep =
/// "shared"`); and its head sharded ([`shard_head`]). The draft head stays replicated (legacy
/// proposes without the communicator).
pub fn with_expert_reduces(circuit: &Circuit, ep: TpRank) -> Result<Circuit, TpError> {
    if ep.world < 2 || ep.rank >= ep.world {
        return Err(TpError::Rank {
            rank: ep.rank,
            world: ep.world,
        });
    }
    let chosen = main_where(circuit, |n| n.op == OpKind::Blend);
    if chosen.is_empty() {
        return Err(TpError::NoExperts);
    }
    shard_head(&crate::follow::follow(circuit, &chosen, split_blend)?, ep)
}

/// 2026-10-03: The routed sum, the reduce and the shared add a `blend` becomes.
fn split_blend(out: &mut Circuit, node: &Node) -> Result<Followers, TpError> {
    let (routed_in, shared_in) = match node.inputs.as_slice() {
        [e, w, s, g] => ([*e, *w], [*s, *g]),
        _ => return Err(TpError::BlendShape(node.id.clone())),
    };
    let f = node.outputs[0];
    let base = out.edges[f].clone();
    let partial = push_edge(
        out,
        Edge {
            id: format!("{}.ep_partial", base.id),
            producer: None,
            consumers: Vec::new(),
            is_output: false,
            binds: None,
            ..base.clone()
        },
    );
    let summed = push_edge(
        out,
        Edge {
            id: format!("{}.ep_sum", base.id),
            producer: None,
            consumers: Vec::new(),
            is_output: false,
            binds: None,
            ..base
        },
    );
    let part = |p: &str| BTreeMap::from([("ep".to_string(), p.to_string())]);
    let mut leader = node.clone();
    leader.inputs = routed_in.to_vec();
    leader.outputs = vec![partial];
    leader.params.extend(part("routed"));
    let follower = |suffix: &str, op, inputs: Vec<EdgeIdx>, outputs, params| Node {
        id: format!("{}_{suffix}", node.id),
        local: format!("{}_{suffix}", node.local),
        op,
        inputs,
        outputs,
        weight: None,
        binding: Vec::new(),
        params,
        layer: node.layer,
        block: node.block.clone(),
        state: Vec::new(),
    };
    Ok(Followers {
        nodes: vec![
            follower(
                "ep_reduce",
                OpKind::EpReduce,
                vec![partial],
                vec![summed],
                BTreeMap::new(),
            ),
            follower(
                "ep_shared",
                OpKind::Blend,
                vec![summed, shared_in[0], shared_in[1]],
                vec![f],
                part("shared"),
            ),
        ],
        replaces: None,
        leader: Some(leader),
    })
}

/// 2026-10-03: `circuit` with its BF16 head vocabulary-parallel over `world` (rank's slice
/// `[rank, rank + 1) x vocab / world`) when the vocabulary divides; unchanged otherwise, as
/// legacy's `lmhead_vocab_shard` (an FP8 or NVFP4 head is never sharded).
pub fn shard_head(circuit: &Circuit, world: TpRank) -> Result<Circuit, TpError> {
    let vocab = circuit.dims.get("vocab").copied().unwrap_or(0);
    if vocab == 0 || vocab % world.world != 0 {
        return Ok(circuit.clone());
    }
    let len = vocab / world.world;
    let heads = main_where(circuit, |n| {
        n.op == OpKind::LmHead && n.weight == Some(crate::format::Format::Bf16)
    });
    crate::follow::follow(circuit, &heads, |_, node| {
        let logits = *node
            .outputs
            .first()
            .ok_or_else(|| TpError::External(node.id.clone()))?;
        let mut leader = node.clone();
        leader.local = SHARDED_HEAD.to_string();
        leader.params.extend([
            ("vocab_begin".to_string(), (world.rank * len).to_string()),
            ("vocab_len".to_string(), len.to_string()),
        ]);
        Ok(Followers {
            nodes: vec![Node {
                id: format!("{}_all_reduce", node.id),
                local: format!("{}_all_reduce", node.local),
                op: OpKind::AllReduce,
                inputs: vec![logits],
                outputs: Vec::new(),
                weight: None,
                binding: Vec::new(),
                params: BTreeMap::from([("in_place".to_string(), "true".to_string())]),
                layer: node.layer,
                block: node.block.clone(),
                state: Vec::new(),
            }],
            replaces: None,
            leader: Some(leader),
        })
    })
}

/// 2026-10-03: The all-reduce that follows row-parallel `node`.
fn reduce_of(out: &mut Circuit, node: &Node) -> Result<Followers, TpError> {
    let y = *node
        .outputs
        .first()
        .ok_or_else(|| TpError::External(node.id.clone()))?;
    let base = out.edges[y].clone();
    let stream = out
        .blocks
        .iter()
        .any(|b| b.stream_in == Some(y) || b.stream_out == Some(y));
    if base.is_output || stream {
        return Err(TpError::External(node.id.clone()));
    }
    let summed = push_edge(
        out,
        Edge {
            id: format!("{}.reduced", node.id),
            producer: None,
            consumers: Vec::new(),
            ..base
        },
    );
    Ok(Followers {
        nodes: vec![Node {
            id: format!("{}_all_reduce", node.id),
            local: format!("{}_all_reduce", node.local),
            op: OpKind::AllReduce,
            inputs: vec![y],
            outputs: vec![summed],
            weight: None,
            binding: Vec::new(),
            params: Default::default(),
            layer: node.layer,
            block: node.block.clone(),
            state: Vec::new(),
        }],
        replaces: Some((y, summed)),
        leader: None,
    })
}

#[cfg(test)]
#[path = "parallel_tests.rs"]
mod parallel_tests;
