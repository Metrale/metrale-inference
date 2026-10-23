// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Pattern matching and group ordering for the fuser.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - A chain element after the first joins by reading an output of an earlier member, or, when
//!   marked `sibling`, by sitting in the first member's block. Among candidates the lowest
//!   node index wins, so matching is deterministic.
//! - A chain is refused when a member's output leaves the group without `keep`, or when an
//!   outside node lies on a path from one member to another (fusing would need a cycle).
//! - Groups execute in a topological order of the group graph, ties broken by their lowest
//!   node index; a node with no outputs (the KV write) orders before later readers of its
//!   inputs.

use crate::prelude::*;

use alloc::collections::{BTreeSet, VecDeque};

use super::{AvailableKernels, Policy, applies, block_index, section_of};
use crate::ir::{Circuit, Node, NodeIdx, OpKind};
use crate::rules::{Mode, PatternOp, Rule};

/// 2026-10-23: One place a rule matches: its id and the chain it forms, in pattern order.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Match {
    /// 2026-10-23: Rule id.
    pub rule: String,
    /// 2026-10-23: Member nodes, in pattern order.
    pub nodes: Vec<NodeIdx>,
}

/// 2026-10-23: Every match of every rule that may form a group of this plan, each formed over a
/// circuit with no node claimed yet: the matches the greedy chooses among, in its own order
/// (rule priority, then start node). The matcher is the fuser's own ([`Matcher::chain_at`]).
pub fn match_instances(
    circuit: &Circuit,
    rules: &[Rule],
    available: &AvailableKernels,
    policy: &Policy,
    (mode, rows): (Mode, u64),
) -> Vec<Match> {
    let (in_scope, block_of) = scope_of(circuit, mode);
    let m = Matcher {
        circuit,
        in_scope: &in_scope,
        block_of: &block_of,
    };
    let free = vec![None; circuit.nodes.len()];
    let mut out = Vec::new();
    for rule in applicable_order(circuit, rules, available, policy, (mode, rows, None)) {
        for s in (0..circuit.nodes.len()).filter(|&s| in_scope[s]) {
            if let Some(nodes) = m.chain_at(rule, s, &free) {
                out.push(Match {
                    rule: rule.id.clone(),
                    nodes,
                });
            }
        }
    }
    out
}


/// 2026-10-01: Node `n` of `c` fits pattern element `p`: op (any listed linear role), local id,
/// weight format, quantized input format and layer kind. The fuser matches chains with it, and
/// the hardware planner asks which rules could run a node at all. 2026-10-08: And every param
/// the element states, with the node's value.
pub(crate) fn fits(c: &Circuit, p: &PatternOp, n: &Node) -> bool {
    let op = match n.op {
        OpKind::Linear(r) if !p.roles.is_empty() => p.roles.contains(&r),
        _ => p.op == n.op,
    };
    op && p.local.as_deref().is_none_or(|l| l == n.local)
        && p.block.as_deref().is_none_or(|b| b == n.block)
        && p.weight.is_none_or(|w| n.weight == Some(w))
        && reads_quantized_as(c, p, n)
        && p.layer_kind
            .is_none_or(|k| n.layer.is_some_and(|i| c.layer_kinds[i] == k))
        && p.params.iter().all(|(k, v)| n.params.get(k) == Some(v))
        && p.ranks
            .is_none_or(|r| group_size(n).is_some_and(|g| r.holds(g)))
}

/// 2026-10-23: The size of a collective's communicator group (its `group` param, the ranks
/// comma-separated, as `instantiate::shard` writes it on every collective it derives).
fn group_size(n: &Node) -> Option<u64> {
    n.params
        .get("group")
        .and_then(|g| u64::try_from(g.split(',').count()).ok())
}

/// 2026-09-30: A node that reads a quantized activation (an `act_quant` output) matches only
/// a pattern element that names that format as its `input`; a pattern naming a quantized
/// input matches only such a node. A kernel for 16-bit activations never takes a W4A4 or
/// W8A8 projection.
fn reads_quantized_as(c: &Circuit, p: &PatternOp, n: &Node) -> bool {
    let read = n
        .inputs
        .first()
        .map(|&e| c.edges[e].format)
        .filter(|f| !f.is_plain());
    match (read, p.input.filter(|f| !f.is_plain())) {
        (None, None) => true,
        (Some(r), Some(w)) => r == w,
        _ => false,
    }
}

/// 2026-10-23: Whether each node is in `mode`'s section, and each node's block.
pub(super) fn scope_of(circuit: &Circuit, mode: Mode) -> (Vec<bool>, Vec<usize>) {
    let section = section_of(mode);
    let block_of = block_index(circuit);
    let in_scope = block_of
        .iter()
        .map(|&b| circuit.blocks[b].section == section)
        .collect();
    (in_scope, block_of)
}

/// 2026-10-23: The rules that may form a group of this plan, in the order the greedy tries
/// them: priority, highest first, then id.
pub(super) fn applicable_order<'a>(
    circuit: &Circuit,
    rules: &'a [Rule],
    available: &AvailableKernels,
    policy: &Policy,
    (mode, rows, table): (Mode, u64, Option<&crate::runs::RowTable>),
) -> Vec<&'a Rule> {
    let mut order: Vec<&Rule> = rules
        .iter()
        .filter(|r| applies(r, available, policy, mode, rows))
        .filter(|r| crate::rules::dims::all_hold(&r.dims, &circuit.dims) == Ok(true))
        .filter(|r| {
            r.runs.is_empty()
                || table.is_some_and(|t| crate::runs::resolve_runs(t, &r.runs).is_some())
        })
        .collect();
    order.sort_by(|a, b| b.priority.cmp(&a.priority).then_with(|| a.id.cmp(&b.id)));
    order
}

pub(super) struct Matcher<'a> {
    pub circuit: &'a Circuit,
    pub in_scope: &'a [bool],
    pub block_of: &'a [usize],
}

impl Matcher<'_> {
    fn fits(&self, p: &PatternOp, n: &Node) -> bool {
        fits(self.circuit, p, n)
    }

    fn free(&self, n: NodeIdx, owner: &[Option<usize>], chain: &[NodeIdx]) -> bool {
        self.in_scope[n] && owner[n].is_none() && !chain.contains(&n)
    }

    /// 2026-09-28: The chain `rule` forms starting at node `s`, if it forms one.
    pub fn chain_at(
        &self,
        rule: &Rule,
        s: NodeIdx,
        owner: &[Option<usize>],
    ) -> Option<Vec<NodeIdx>> {
        let c = self.circuit;
        if !self.fits(&rule.pattern[0], &c.nodes[s]) {
            return None;
        }
        let mut chain = vec![s];
        for p in &rule.pattern[1..] {
            let candidates: BTreeSet<NodeIdx> = if p.sibling {
                let blk = &c.blocks[self.block_of[s]];
                (blk.first..blk.end).collect()
            } else {
                chain
                    .iter()
                    .flat_map(|&m| c.nodes[m].outputs.iter())
                    .flat_map(|&e| c.edges[e].consumers.iter().copied())
                    .collect()
            };
            let next = candidates
                .into_iter()
                .find(|&n| self.free(n, owner, &chain) && self.fits(p, &c.nodes[n]))?;
            chain.push(next);
        }
        (self.outputs_legal(rule, &chain) && !self.needs_cycle(&chain)).then_some(chain)
    }

    fn outputs_legal(&self, rule: &Rule, chain: &[NodeIdx]) -> bool {
        let c = self.circuit;
        let last = chain.len() - 1;
        chain.iter().enumerate().all(|(j, &m)| {
            j == last
                || rule.pattern[j].keep
                || c.nodes[m].outputs.iter().all(|&e| {
                    let edge = &c.edges[e];
                    (!edge.is_output || rule.pattern[j].elides)
                        && edge.consumers.iter().all(|x| chain.contains(x))
                })
        })
    }

    fn needs_cycle(&self, chain: &[NodeIdx]) -> bool {
        let c = self.circuit;
        let hi = *chain.iter().max().unwrap_or(&0);
        let mut seen = BTreeSet::new();
        let mut queue: VecDeque<NodeIdx> = VecDeque::new();
        let push_readers = |n: NodeIdx, q: &mut VecDeque<NodeIdx>| {
            for &e in &c.nodes[n].outputs {
                q.extend(c.edges[e].consumers.iter().copied());
            }
        };
        for &m in chain {
            push_readers(m, &mut queue);
        }
        while let Some(n) = queue.pop_front() {
            if n > hi || !seen.insert(n) {
                continue;
            }
            if chain.contains(&n) {
                // 2026-09-28: Reached a member only through an outside node when `n` was
                // queued from one; a member reading a member directly is the chain itself.
                continue;
            }
            if c.nodes[n]
                .outputs
                .iter()
                .any(|&e| c.edges[e].consumers.iter().any(|x| chain.contains(x)))
            {
                return true;
            }
            push_readers(n, &mut queue);
        }
        false
    }
}

/// 2026-09-28: Group indices in execution order.
pub(super) fn execution_order(
    circuit: &Circuit,
    in_scope: &[bool],
    owner: &[Option<usize>],
    groups: usize,
) -> Vec<usize> {
    let mut first = vec![usize::MAX; groups];
    for (n, o) in owner.iter().enumerate() {
        if let Some(g) = o {
            first[*g] = first[*g].min(n);
        }
    }
    let mut succ: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); groups];
    let mut add = |from: Option<usize>, to: Option<usize>| {
        if let (Some(a), Some(b)) = (from, to)
            && a != b
        {
            succ[a].insert(b);
        }
    };
    for e in &circuit.edges {
        let Some(p) = e.producer.filter(|&p| in_scope[p]) else {
            continue;
        };
        for &r in e.consumers.iter().filter(|&&r| in_scope[r]) {
            add(owner[p], owner[r]);
        }
    }
    for (s, node) in circuit.nodes.iter().enumerate() {
        if !in_scope[s] || !node.outputs.is_empty() {
            continue;
        }
        for &e in &node.inputs {
            for &r in circuit.edges[e]
                .consumers
                .iter()
                .filter(|&&r| r > s && in_scope[r])
            {
                add(owner[s], owner[r]);
            }
        }
    }
    let mut indeg = vec![0usize; groups];
    for s in &succ {
        for &b in s {
            indeg[b] += 1;
        }
    }
    let mut ready: BTreeSet<(usize, usize)> = (0..groups)
        .filter(|&g| indeg[g] == 0)
        .map(|g| (first[g], g))
        .collect();
    let mut out = Vec::with_capacity(groups);
    while let Some((_, g)) = ready.pop_first() {
        out.push(g);
        for &b in &succ[g] {
            indeg[b] -= 1;
            if indeg[b] == 0 {
                ready.insert((first[b], b));
            }
        }
    }
    out
}
