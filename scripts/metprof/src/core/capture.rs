// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: one device's window from the raw rows of a trace (pure).
//!
//! The adapters hand over every row of every device; this selects a device,
//! orders the rows, and names the copy kinds. An absent device is an error
//! that lists the devices the capture does have.

use std::collections::BTreeSet;

use crate::core::error::Result;
use crate::core::model::{ApiCall, Copy, Kernel};
use crate::fail;

/// A kernel row as stored: the device it ran on and the kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelRow {
    pub device: i64,
    pub kernel: Kernel,
}

/// A memcpy row as stored: `kind` is the CUPTI copy-kind number.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CopyRow {
    pub device: i64,
    pub start: i64,
    pub end: i64,
    pub kind: i64,
    pub nbytes: i64,
}

/// CUpti_ActivityMemcpyKind values as nsys exports them.
pub fn copy_kind(kind: i64) -> String {
    match kind {
        1 => "HtoD".to_string(),
        2 => "DtoH".to_string(),
        8 => "DtoD".to_string(),
        n => format!("kind{n}"),
    }
}

pub fn devices(rows: &[KernelRow]) -> Vec<i64> {
    rows.iter()
        .map(|r| r.device)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Kernels of one device, ordered by (start, end).
pub fn device_kernels(label: &str, rows: Vec<KernelRow>, device: i64) -> Result<Vec<Kernel>> {
    let have = devices(&rows);
    if !have.contains(&device) {
        let list: Vec<String> = have.iter().map(i64::to_string).collect();
        fail!(
            "{label}: device {device} has no kernels (devices present: [{}])",
            list.join(", ")
        );
    }
    let mut out: Vec<Kernel> = rows
        .into_iter()
        .filter(|r| r.device == device)
        .map(|r| r.kernel)
        .collect();
    out.sort_by_key(|k| (k.start, k.end));
    Ok(out)
}

/// Memcpy activity of one device, ordered by (start, end).
pub fn device_copies(rows: Vec<CopyRow>, device: i64) -> Vec<Copy> {
    let mut sel: Vec<CopyRow> = rows.into_iter().filter(|r| r.device == device).collect();
    sel.sort_by_key(|r| (r.start, r.end));
    sel.into_iter()
        .map(|r| Copy {
            start: r.start,
            end: r.end,
            kind: copy_kind(r.kind),
            nbytes: r.nbytes,
        })
        .collect()
}

/// API calls ordered by (start, end).
pub fn ordered_calls(mut calls: Vec<ApiCall>) -> Vec<ApiCall> {
    calls.sort_by_key(|a| (a.start, a.end));
    calls
}

/// True when any kernel (of any device) carries a graph node id, i.e. the
/// capture was taken with `--cuda-graph-trace=node`.
pub fn graph_node_traced(rows: &[KernelRow]) -> bool {
    rows.iter().any(|r| r.kernel.in_graph())
}
