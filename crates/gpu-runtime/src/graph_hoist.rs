// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The NCCL host-node hoist, as a plan over a captured graph's edges.
//!
//! A collective captured over a network transport adds a host-callback node (the proxy
//! upload) that runs only after the kernel producing the collective's input, and the
//! collective's kernel waits for that callback. Each collective therefore pays a host callback
//! on its critical path: about 3.8 us before each of the 91 collectives of a GLM-5.3 decode
//! step (rank-0 nsys, 2026-10-10).
//!
//! The hoist rewires the host nodes into a chain of their own, each depending only on the
//! previous host node, so the callbacks run ahead in their original order. Each node that
//! depended on a host node first gets a direct edge from every node that host node gave up,
//! so a kernel still waits for everything it waited for before.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - For any two non-host nodes, if one reached the other before the plan, it still does.
//! - Host nodes depend only on host nodes, and each reaches the next one in topological order.

use std::collections::{BTreeMap, BTreeSet};

/// 2026-10-10: An edge `(from, to)`: `to` depends on `from`. Node ids are indices.
pub type Edge = (usize, usize);

/// 2026-10-10: The edges to remove and to add, in that order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HoistPlan {
    pub remove: Vec<Edge>,
    pub add: Vec<Edge>,
}

/// 2026-10-10: A topological order of `n` nodes under `edges` (Kahn, lowest ready index
/// first, so the order is deterministic). Errors on a cycle.
pub fn topo_order(n: usize, edges: &[Edge]) -> anyhow::Result<Vec<usize>> {
    let mut indeg = vec![0usize; n];
    let mut outs: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(a, b) in edges {
        anyhow::ensure!(a < n && b < n, "edge ({a}, {b}) outside {n} nodes");
        indeg[b] += 1;
        outs[a].push(b);
    }
    let mut ready: BTreeSet<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(i) = ready.pop_first() {
        order.push(i);
        for &j in &outs[i] {
            indeg[j] -= 1;
            if indeg[j] == 0 {
                ready.insert(j);
            }
        }
    }
    anyhow::ensure!(order.len() == n, "the graph has a cycle");
    Ok(order)
}

/// 2026-10-10: The hoist plan for `n` nodes, `edges`, and which nodes are host callbacks.
pub fn plan_hoist(n: usize, edges: &[Edge], is_host: &[bool]) -> anyhow::Result<HoistPlan> {
    anyhow::ensure!(
        is_host.len() == n,
        "{} host flags for {n} nodes",
        is_host.len()
    );
    let order = topo_order(n, edges)?;
    let mut live: BTreeSet<Edge> = edges.iter().copied().collect();
    let deps = |live: &BTreeSet<Edge>, x: usize| -> Vec<usize> {
        live.iter().filter(|e| e.1 == x).map(|e| e.0).collect()
    };
    let mut plan = HoistPlan::default();
    let mut added: BTreeMap<Edge, ()> = BTreeMap::new();
    let mut prev: Option<usize> = None;
    for &h in order.iter().filter(|&&i| is_host[i]) {
        let gave_up = deps(&live, h);
        let dependents: Vec<usize> = live.iter().filter(|e| e.0 == h).map(|e| e.1).collect();
        for &x in &dependents {
            for &d in &gave_up {
                if live.insert((d, x)) {
                    plan.add.push((d, x));
                    added.insert((d, x), ());
                }
            }
        }
        for d in gave_up {
            live.remove(&(d, h));
            // 2026-10-10: An edge this plan added and now removes is dropped from both lists.
            if added.remove(&(d, h)).is_some() {
                plan.add.retain(|e| *e != (d, h));
            } else {
                plan.remove.push((d, h));
            }
        }
        if let Some(p) = prev
            && live.insert((p, h))
        {
            plan.add.push((p, h));
            added.insert((p, h), ());
        }
        prev = Some(h);
    }
    Ok(plan)
}

#[cfg(test)]
#[path = "graph_hoist_tests.rs"]
mod tests;
