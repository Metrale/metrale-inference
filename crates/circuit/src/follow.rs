// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The one rewrite the feature overlays share (FEATURES workstream): new nodes placed
//! right after chosen nodes of an instantiated circuit, every other reader of a chosen node's
//! output moved to the edge the new nodes produce in its place. The LoRA overlay follows an
//! adapted projection with its shrink and expand ([`crate::lora`]); tensor parallelism follows a
//! row-parallel projection with its all-reduce ([`crate::parallel`]).
//!
//! Owner: metrale-circuit (FEATURES workstream).
//! Invariants:
//! - Nodes stay in a topological order: the followers of node `n` come right after it, and read
//!   only `n`'s inputs and outputs or edges they produce.
//! - Producers, consumers and block ranges are rebuilt from the nodes, so they cannot disagree
//!   with them; a follower belongs to its leader's block.

use std::collections::{BTreeMap, BTreeSet};

use crate::ir::{Circuit, EdgeIdx, Node, NodeIdx};

/// 2026-10-03: What follows one chosen node: the new nodes (their edges already pushed into the
/// circuit being built), the replacement `(old, new)` of the leader's output for every other
/// reader, and a rewritten leader (its inputs or outputs changed) to stand in its place.
pub(crate) struct Followers {
    pub nodes: Vec<Node>,
    pub replaces: Option<(EdgeIdx, EdgeIdx)>,
    pub leader: Option<Node>,
}

/// 2026-10-03: `circuit` with `make`'s followers after each node in `chosen`. `make` receives the
/// circuit being built (to push the followers' edges) and the leader.
pub(crate) fn follow<E>(
    circuit: &Circuit,
    chosen: &BTreeSet<NodeIdx>,
    mut make: impl FnMut(&mut Circuit, &Node) -> Result<Followers, E>,
) -> Result<Circuit, E> {
    let mut out = circuit.clone();
    out.nodes.clear();
    let mut moved = vec![0usize; circuit.nodes.len()];
    let mut followers = vec![0usize; circuit.nodes.len()];
    let mut is_follower = Vec::with_capacity(circuit.nodes.len());
    let mut redirect: BTreeMap<EdgeIdx, EdgeIdx> = BTreeMap::new();
    for (n, node) in circuit.nodes.iter().enumerate() {
        moved[n] = out.nodes.len();
        out.nodes.push(node.clone());
        is_follower.push(false);
        if chosen.contains(&n) {
            let f = make(&mut out, node)?;
            followers[n] = f.nodes.len();
            if let Some(leader) = f.leader {
                *out.nodes.last_mut().expect("the leader was pushed") = leader;
            }
            if let Some((old, new)) = f.replaces {
                redirect.insert(old, new);
            }
            for m in f.nodes {
                out.nodes.push(m);
                is_follower.push(true);
            }
        }
    }
    for (node, &f) in out.nodes.iter_mut().zip(&is_follower) {
        if !f {
            for i in &mut node.inputs {
                if let Some(&to) = redirect.get(i) {
                    *i = to;
                }
            }
        }
    }
    let at = |n: usize| moved.get(n).copied().unwrap_or(out.nodes.len());
    for (b, sb) in out.blocks.iter_mut().zip(&circuit.blocks) {
        b.first = at(sb.first);
        b.end = match sb.end.checked_sub(1).filter(|&l| l >= sb.first) {
            Some(last) => moved[last] + 1 + followers[last],
            None => b.first,
        };
    }
    for e in &mut out.edges {
        e.producer = None;
        e.consumers.clear();
    }
    for (n, node) in out.nodes.iter().enumerate() {
        for &o in &node.outputs {
            out.edges[o].producer = Some(n);
        }
        for &i in &node.inputs {
            if !out.edges[i].consumers.contains(&n) {
                out.edges[i].consumers.push(n);
            }
        }
    }
    Ok(out)
}

/// 2026-10-03: Which nodes of `c` belong to the target's forward.
pub(crate) fn main_nodes(c: &Circuit) -> Vec<bool> {
    let mut main = vec![false; c.nodes.len()];
    for b in c
        .blocks
        .iter()
        .filter(|b| b.section == crate::ir::Section::Main)
    {
        main[b.first..b.end].fill(true);
    }
    main
}

/// 2026-10-03: Push `e` into the circuit being built; its index.
pub(crate) fn push_edge(out: &mut Circuit, e: crate::ir::Edge) -> EdgeIdx {
    out.edges.push(e);
    out.edges.len() - 1
}
