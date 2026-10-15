// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the I/O ports. The pure core never reads a file; the CLI asks
//! these traits for bytes and rows and hands plain values to the core. The
//! real implementations live in `adapters`; tests implement them in memory.

use crate::core::capture::{CopyRow, KernelRow};
use crate::core::error::Result;
use crate::core::model::ApiCall;

/// Opens a capture by path. A missing path is `capture not found: PATH` and
/// nothing is ever created or written.
pub trait TraceSource {
    fn open(&self, path: &str) -> Result<Box<dyn Trace>>;
}

/// One opened capture: every row of every device, unordered.
pub trait Trace {
    /// The kernel table joined with its names. `graph_node` is `None` when
    /// the export has no graph-node column.
    fn kernel_rows(&self) -> Result<Vec<KernelRow>>;
    fn copy_rows(&self) -> Result<Vec<CopyRow>>;
    /// Host-side CUDA runtime API calls with their names.
    fn call_rows(&self) -> Result<Vec<ApiCall>>;
}

/// Whole-file reads for configs, records and environments.
pub trait Files {
    fn read(&self, path: &str) -> std::io::Result<Vec<u8>>;
}
