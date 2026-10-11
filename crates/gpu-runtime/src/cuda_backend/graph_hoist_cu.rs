// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Applies the NCCL host-node hoist (`crate::graph_hoist`) to a captured graph
//! before it is instantiated, when `METRALE_NCCL_HOST_HOIST=1`.
//!
//! The graph's nodes and edges are read with the driver's graph queries, the plan is made over
//! node indices, and the edge removals and additions are applied with
//! `cuGraphRemoveDependencies` / `cuGraphAddDependencies`. Any edge with non-default edge data
//! (the query then answers `CUDA_ERROR_LOSSY_QUERY`) leaves the graph as captured.
//!
//! Owner: gpu-runtime (CUDA backend).
//! Invariants:
//! - Off unless the lever is `1`; a graph without host nodes is never touched.

use anyhow::{Result, bail};

/// 2026-10-10: The lever.
const ENV: &str = "METRALE_NCCL_HOST_HOIST";

/// 2026-10-10: `CU_GRAPH_NODE_TYPE_HOST`.
const NODE_TYPE_HOST: u32 = 3;
/// 2026-10-10: `CUDA_ERROR_LOSSY_QUERY`.
const LOSSY_QUERY: i32 = 402;

// 2026-10-11: The graph queries go through cudarc's dynamically loaded driver table, not a link-time
// `extern` block: the CUDA 13 link stub (`lib64/stubs/libcuda.so`) does not export them, so a
// static reference fails to link on boxes whose linker resolves `-lcuda` to that stub.
use cudarc::driver::sys;

unsafe fn graph_get_nodes(h_graph: u64, nodes: *mut u64, num_nodes: *mut usize) -> i32 {
    unsafe { sys::cuGraphGetNodes(h_graph as sys::CUgraph, nodes.cast(), num_nodes) as i32 }
}

unsafe fn node_get_type(h_node: u64, ty: *mut u32) -> i32 {
    let mut t = std::mem::MaybeUninit::<sys::CUgraphNodeType>::uninit();
    let rc = unsafe { sys::cuGraphNodeGetType(h_node as sys::CUgraphNode, t.as_mut_ptr()) };
    if rc == sys::CUresult::CUDA_SUCCESS {
        // SAFETY: the driver wrote a valid node type on success.
        unsafe { *ty = t.assume_init() as u32 };
    }
    rc as i32
}

unsafe fn node_get_dependencies(
    h_node: u64,
    deps: *mut u64,
    edge_data: *mut std::ffi::c_void,
    num: *mut usize,
) -> i32 {
    unsafe {
        sys::cuGraphNodeGetDependencies_v2(
            h_node as sys::CUgraphNode,
            deps.cast(),
            edge_data.cast(),
            num,
        ) as i32
    }
}

unsafe fn add_dependencies(
    h_graph: u64,
    from: *const u64,
    to: *const u64,
    edge_data: *const std::ffi::c_void,
    num: usize,
) -> i32 {
    unsafe {
        sys::cuGraphAddDependencies_v2(
            h_graph as sys::CUgraph,
            from.cast(),
            to.cast(),
            edge_data.cast(),
            num,
        ) as i32
    }
}

unsafe fn remove_dependencies(
    h_graph: u64,
    from: *const u64,
    to: *const u64,
    edge_data: *const std::ffi::c_void,
    num: usize,
) -> i32 {
    unsafe {
        sys::cuGraphRemoveDependencies_v2(
            h_graph as sys::CUgraph,
            from.cast(),
            to.cast(),
            edge_data.cast(),
            num,
        ) as i32
    }
}

/// 2026-10-10: Whether the lever is on (read once per process).
pub(super) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var(ENV).as_deref() == Ok("1"))
}

fn check(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("{what} failed: status {rc}");
    }
    Ok(())
}

/// 2026-10-10: Hoist the host nodes of `graph` (a captured, not yet instantiated `CUgraph`).
/// Returns how many host nodes were chained; 0 when the graph had none or carries edge data
/// this rewiring does not reproduce.
pub(super) fn hoist(graph: u64) -> Result<usize> {
    let mut n = 0usize;
    check(
        unsafe { graph_get_nodes(graph, std::ptr::null_mut(), &mut n) },
        "cuGraphGetNodes",
    )?;
    let mut nodes = vec![0u64; n];
    check(
        unsafe { graph_get_nodes(graph, nodes.as_mut_ptr(), &mut n) },
        "cuGraphGetNodes",
    )?;
    nodes.truncate(n);
    let index: std::collections::HashMap<u64, usize> =
        nodes.iter().enumerate().map(|(i, &h)| (h, i)).collect();
    let mut is_host = vec![false; n];
    let mut edges = Vec::new();
    for (i, &h) in nodes.iter().enumerate() {
        let mut ty = 0u32;
        check(unsafe { node_get_type(h, &mut ty) }, "cuGraphNodeGetType")?;
        is_host[i] = ty == NODE_TYPE_HOST;
        let mut k = 0usize;
        let rc =
            unsafe { node_get_dependencies(h, std::ptr::null_mut(), std::ptr::null_mut(), &mut k) };
        if rc == LOSSY_QUERY {
            tracing::warn!("NCCL host-node hoist skipped: the graph has edge data");
            return Ok(0);
        }
        check(rc, "cuGraphNodeGetDependencies")?;
        let mut deps = vec![0u64; k];
        if k > 0 {
            check(
                unsafe {
                    node_get_dependencies(h, deps.as_mut_ptr(), std::ptr::null_mut(), &mut k)
                },
                "cuGraphNodeGetDependencies",
            )?;
        }
        for d in deps.into_iter().take(k) {
            let from = *index
                .get(&d)
                .ok_or_else(|| anyhow::anyhow!("a dependency outside the graph's node list"))?;
            edges.push((from, i));
        }
    }
    let hosts = is_host.iter().filter(|&&h| h).count();
    if hosts == 0 {
        return Ok(0);
    }
    let plan = crate::graph_hoist::plan_hoist(n, &edges, &is_host)?;
    for &(a, b) in &plan.remove {
        check(
            unsafe { remove_dependencies(graph, &nodes[a], &nodes[b], std::ptr::null(), 1) },
            "cuGraphRemoveDependencies",
        )?;
    }
    for &(a, b) in &plan.add {
        check(
            unsafe { add_dependencies(graph, &nodes[a], &nodes[b], std::ptr::null(), 1) },
            "cuGraphAddDependencies",
        )?;
    }
    Ok(hosts)
}
