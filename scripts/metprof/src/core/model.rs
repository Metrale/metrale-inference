// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: row types handed from the adapters to the pure core. Times are
//! integer ns, as the profiler stores them.

/// One kernel launch of the selected device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kernel {
    pub start: i64,
    pub end: i64,
    pub stream: i64,
    pub correlation: i64,
    pub graph_node: Option<i64>,
    pub name: String,
    /// The full (demangled) name; `name` is the short name.
    pub demangled: String,
    pub grid: Grid,
    pub launch: Launch,
}

pub type Grid = (i64, i64, i64);

/// What CUPTI records about a launch besides its grid: the block shape,
/// registers per thread and shared memory in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Launch {
    pub block: Grid,
    pub registers_per_thread: i64,
    pub static_shared_bytes: i64,
    pub dynamic_shared_bytes: i64,
}

/// Kernel identity: the name alone is not enough (one name runs at several
/// shapes), so the grid is part of the key.
pub type KernelKey = (String, Grid);

impl Kernel {
    pub fn key(&self) -> KernelKey {
        (self.name.clone(), self.grid)
    }

    pub fn dur(&self) -> i64 {
        self.end - self.start
    }

    /// A kernel launched as a node of a CUDA graph (node-level trace only).
    pub fn in_graph(&self) -> bool {
        matches!(self.graph_node, Some(n) if n != 0)
    }
}

/// One memcpy of the selected device; `kind` is `HtoD`, `DtoH`, `DtoD` or
/// `kind<N>` for any other copy kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Copy {
    pub start: i64,
    pub end: i64,
    pub kind: String,
    pub nbytes: i64,
}

/// One host-side CUDA runtime/driver API call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiCall {
    pub start: i64,
    pub end: i64,
    pub name: String,
    pub correlation: i64,
}

pub const NS_PER_MS: f64 = 1e6;

pub fn ms(ns: f64) -> f64 {
    ns / NS_PER_MS
}

/// A ms command-line value as integer ns, rounded half to even like the
/// thresholds have always been rounded.
pub fn ms_to_ns(value_ms: f64) -> i64 {
    (value_ms * NS_PER_MS).round_ties_even() as i64
}
