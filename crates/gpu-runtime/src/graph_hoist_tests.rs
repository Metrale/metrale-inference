// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Tests of the host-node hoist plan on synthetic captured graphs.
//!
//! Owner: gpu-runtime.
//! Invariants: none beyond the types.
//!
//! The properties: every reachability between non-host nodes survives, host nodes end up
//! depending only on host nodes, and each host node reaches the next in topological order. A
//! naive rewire (dropping the host node's dependencies without handing them to its dependents)
//! is the negative control: the reachability check rejects it on the NCCL-shaped chain.

use super::{Edge, HoistPlan, plan_hoist, topo_order};
use std::collections::BTreeSet;

fn apply(edges: &[Edge], plan: &HoistPlan) -> Vec<Edge> {
    let mut live: BTreeSet<Edge> = edges.iter().copied().collect();
    for e in &plan.remove {
        assert!(live.remove(e), "plan removes a missing edge {e:?}");
    }
    for e in &plan.add {
        assert!(live.insert(*e), "plan adds an existing edge {e:?}");
    }
    live.into_iter().collect()
}

fn reach(n: usize, edges: &[Edge]) -> Vec<Vec<bool>> {
    let mut r = vec![vec![false; n]; n];
    let order = topo_order(n, edges).unwrap();
    for &v in order.iter().rev() {
        for &(a, b) in edges.iter().filter(|e| e.0 == v) {
            let below = r[b].clone();
            r[a][b] = true;
            for (dst, src) in r[a].iter_mut().zip(below) {
                *dst |= src;
            }
        }
    }
    r
}

/// 2026-10-10: `Err` names the first property the rewired `after` breaks.
fn check(n: usize, before: &[Edge], after: &[Edge], is_host: &[bool]) -> Result<(), String> {
    let (rb, ra) = (reach(n, before), reach(n, after));
    for a in (0..n).filter(|&i| !is_host[i]) {
        for b in (0..n).filter(|&i| !is_host[i]) {
            if rb[a][b] && !ra[a][b] {
                return Err(format!("{a} no longer reaches {b}"));
            }
        }
    }
    for &(a, b) in after {
        if is_host[b] && !is_host[a] {
            return Err(format!("host {b} still depends on non-host {a}"));
        }
    }
    let hosts: Vec<usize> = topo_order(n, before)
        .unwrap()
        .into_iter()
        .filter(|&i| is_host[i])
        .collect();
    for w in hosts.windows(2) {
        if !ra[w[0]][w[1]] {
            return Err(format!("host {} does not reach host {}", w[0], w[1]));
        }
    }
    Ok(())
}

/// 2026-10-10: `k` collectives as NCCL captures them: producer -> host -> collective kernel,
/// the collective kernel feeding the next producer. Node ids: 3i producer, 3i+1 host,
/// 3i+2 kernel.
fn nccl_chain(k: usize) -> (usize, Vec<Edge>, Vec<bool>) {
    let n = 3 * k;
    let mut e = Vec::new();
    for i in 0..k {
        e.push((3 * i, 3 * i + 1));
        e.push((3 * i + 1, 3 * i + 2));
        if i + 1 < k {
            e.push((3 * i + 2, 3 * i + 3));
        }
    }
    let host = (0..n).map(|v| v % 3 == 1).collect();
    (n, e, host)
}

#[test]
fn the_nccl_chain_hoists_and_keeps_every_kernel_behind_its_producer() {
    let (n, e, host) = nccl_chain(4);
    let plan = plan_hoist(n, &e, &host).unwrap();
    let after = apply(&e, &plan);
    check(n, &e, &after, &host).unwrap();
    // 2026-10-10: Host 1 has no dependency left; host 4 hangs off host 1 alone.
    assert!(after.iter().all(|&(_, b)| b != 1));
    assert_eq!(
        after
            .iter()
            .filter(|e| e.1 == 4)
            .copied()
            .collect::<Vec<_>>(),
        vec![(1, 4)]
    );
    // 2026-10-10: Kernel 2 waits for producer 0 directly.
    assert!(after.contains(&(0, 2)));
}

#[test]
fn a_naive_rewire_fails_the_reachability_check() {
    let (n, e, host) = nccl_chain(3);
    let mut naive: Vec<Edge> = e.iter().copied().filter(|&(_, b)| !host[b]).collect();
    naive.push((1, 4));
    naive.push((4, 7));
    let err = check(n, &e, &naive, &host).unwrap_err();
    assert!(err.contains("no longer reaches"), "{err}");
}

#[test]
fn random_dags_keep_every_property() {
    let mut state = 0x9E37_79B9u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for _ in 0..400 {
        let n = 3 + (next() % 14) as usize;
        let mut e = Vec::new();
        for b in 1..n {
            for a in 0..b {
                if next() % 4 == 0 {
                    e.push((a, b));
                }
            }
        }
        let host: Vec<bool> = (0..n).map(|_| next() % 3 == 0).collect();
        let plan = plan_hoist(n, &e, &host).unwrap();
        let after = apply(&e, &plan);
        check(n, &e, &after, &host).unwrap();
    }
}

#[test]
fn a_graph_without_host_nodes_is_left_alone() {
    let e = vec![(0, 1), (1, 2), (0, 2)];
    assert_eq!(
        plan_hoist(3, &e, &[false; 3]).unwrap(),
        HoistPlan::default()
    );
}
